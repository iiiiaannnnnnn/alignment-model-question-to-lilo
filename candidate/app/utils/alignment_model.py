"""
LILO Alignment Module  --  Recommended Composite Pipeline (v2)
=============================================================
Implements the full recommended alignment model pipeline:

  Step 1   Text Normalization     (lowercase, punct, stopwords, suffix stem)
  Step 2   Bloom Level Detection  (regex keyword map, 6 levels)
  Step 3   Keyword Extraction     (KeyBERT with fallback to stemmed tokens)
  Step 4   Embedding Generation   (multi-qa-mpnet-base-dot-v1, 768-dim, local/free)
  Step 5   Four Scoring Channels  (computed in parallel per LILO candidate):
             Channel 1 -- Cosine Similarity           (semantic, w=0.45)
             Channel 2 -- Keyword Jaccard Similarity  (topic coverage, w=0.20)
             Channel 3 -- Content Block Jaccard       (syllabus context, w=0.15)
             Channel 4 -- Bloom Match Score           (cognitive level, w=0.20)
  Step 6   Composite Score Fusion:
             FinalScore = 0.45·cosine + 0.20·jaccard + 0.15·content + 0.20·bloom
  Step 7   Threshold Gate        (FinalScore >= 0.30, or 0.28 when the LILOs are
                                   already scoped to <= 2 content blocks)
  Step 8   Top-3 Selection       (sort candidates, keep top 3)
  Step 9   Close-Score Decision  (if top-2 differ by < 0.15 -> LLM validation,
                                   skipped entirely for ILO-driven questions)
  Step 10  LLM Validation        (Gemini, conditional, rate-limited)
  Step 11  Final LILO Alignment   (per question output with full breakdown)

Formulas:
  Cosine Similarity:
      cos(A,B) = (A . B) / (||A|| * ||B||)

  Bloom Match Score (graduated cognitive proximity):
      bloom_match = 1.00  if same level
                  = 0.75  if 1 step apart on the taxonomy
                  = 0.50  if 2 steps apart
                  = 0.25  if 3+ steps apart

  Keyword Jaccard Score:
      jaccard = |Q_kw ∩ LILO_kw| / |Q_kw ∪ LILO_kw|

  Composite Score (weights empirically tuned, sum = 1.0):
      FinalScore = 0.45·cosine + 0.20·jaccard + 0.15·content + 0.20·bloom

  Softmax Confidence (tau = 0.10, for reporting only -- not the gate):
      conf = exp(s_max / tau) / Σ_j exp(s_j / tau)

ILO-Driven Quiz Generation:
  When a question carries ``source_ilo_id`` (set by the ILO-driven generator),
  that ILO is used as-is as the authoritative tag and the LLM tiebreak is
  skipped entirely (saving API calls and latency).

Backward Compatibility:
  All public function signatures are unchanged so existing routes/lilo.py,
  routes/assessment.py, and utils/algorithms.py continue to work without edits.

Model: multi-qa-mpnet-base-dot-v1  (Sentence Transformer, local, ~420 MB, 768-dim)
         Trained on question-answer pairs — better Q-to-LILO semantic matching
         than all-MiniLM-L6-v2.

Keyword Extraction: KeyBERT (BERT-based, falls back to stemmed tokens if
         keybert is not installed).  KeyBERT extracts semantically meaningful
         keywords using BERT embeddings instead of raw stemmed tokens, making
         the Jaccard channel more accurate.
"""

from typing import List, Dict, Optional, Tuple
import re
import time
import os
import hashlib
import json
import logging
import threading
from app.utils.gemini_key_pool import get_next_key, all_keys, key_count

# Import settings lazily to avoid circular imports at module load time
def _get_settings():
    from app.config.settings import settings
    return settings

logger = logging.getLogger(__name__)

# In-memory enrichment cache (keyed by LILO text hash, cleared on restart)
_ENRICHMENT_CACHE: Dict[str, List[Dict]] = {}

# ---------------------------------------------------------------------------
# Bloom's Taxonomy keyword map
# ---------------------------------------------------------------------------

# Verb tables aligned with UARK Bloom's Taxonomy reference
# (https://tips.uark.edu/using-blooms-taxonomy)
# Plus common verbs found in Philippine CHED / ISU syllabi
BLOOMS_VERBS: Dict[str, List[str]] = {
    "remember":   [
        # UARK canonical
        "list", "recite", "outline", "define", "name", "match",
        "quote", "recall", "identify", "label", "recognize",
        # Additional / CHED common
        "state", "memorize", "repeat", "select", "choose", "locate",
        "enumerate", "give", "show", "tell", "write",
        "what is", "what are", "who", "when", "where",
    ],
    "understand": [
        # UARK canonical
        "describe", "explain", "paraphrase", "restate", "give original examples of",
        "summarize", "contrast", "interpret", "discuss",
        # Additional / CHED common
        "classify", "outline", "illustrate", "give example", "indicate",
        "review", "relate", "predict", "infer", "translate", "convert",
        "compare", "express", "comprehend", "distinguish",
        # Common CHED verbs implying understanding
        "gain", "acquire", "appreciate", "understand",
    ],
    "apply":      [
        # UARK canonical
        "calculate", "predict", "apply", "solve", "illustrate",
        "use", "demonstrate", "determine", "model", "perform", "present",
        # Additional / CHED common
        "compute", "construct", "show", "operate", "practice", "implement",
        "execute", "complete", "carry out", "utilize", "employ",
        "make use", "work out",
    ],
    "analyze":    [
        # UARK canonical
        "classify", "break down", "categorize", "analyze", "diagram",
        "illustrate", "criticize", "simplify", "associate",
        # Additional / CHED common
        "differentiate", "examine", "distinguish", "contrast", "deconstruct",
        "separate", "question", "test", "investigate", "find",
        "identify the", "what causes", "why does", "how does",
        "conduct", "attribute", "compare and contrast",
        "analyse",  # British/Filipino spelling variant
    ],
    "evaluate":   [
        # UARK canonical
        "choose", "support", "relate", "determine", "defend", "judge",
        "grade", "argue", "justify", "convince",
        "select", "evaluate",
        # Additional / CHED common
        "critique", "assess", "recommend", "prioritize", "decide",
        "measure", "rate", "rank", "conclude", "verify",
        "which is best",
    ],
    "create":     [
        # UARK canonical
        "design", "formulate", "build", "invent", "create",
        "compose", "generate", "derive", "modify", "develop",
        # Additional / CHED common
        "propose", "produce", "plan", "construct", "make", "write",
        "assemble", "organize", "prepare", "setup", "devise",
        "draft", "compile",
        # Common in ISU / CHED syllabi
        "cultivate", "foster",
    ],
}

_BLOOMS_ORDER = ["create", "evaluate", "analyze", "apply", "understand", "remember"]

# Ascending taxonomy order (lowest → highest cognitive demand).
_BLOOM_ASC = ["remember", "understand", "apply", "analyze", "evaluate", "create"]

# ---------------------------------------------------------------------------
# Bloom ceiling per objective question type
# ---------------------------------------------------------------------------
# The highest Bloom level each objective format can validly assess. Used by the
# quiz/exam generator to PREFER Bloom-compatible LILOs (soft model: prefer +
# warn + allow — never removes a LILO). Higher-order outcomes (evaluate/create)
# are better measured by rubric-scored Activities.
QUESTION_TYPE_BLOOM_CEILING: Dict[str, str] = {
    "enumeration":    "remember",    # "enumerate/list" is itself a remember verb
    "identification": "understand",  # recall + recognize a term/concept
    "trueOrFalse":    "understand",  # judge one proposition's truth
    "multipleChoice": "analyze",     # well-built stems can reach analyze
}


# The same four formats travel under two spellings: the frontend flag dict and
# the ceiling table above use camelCase ("multipleChoice"), while a generated
# question's ``type`` field is snake_case ("multiple_choice"). Map both onto the
# canonical key so a ceiling lookup never depends on which spelling arrived.
_QUESTION_TYPE_ALIASES: Dict[str, str] = {
    "multiplechoice":  "multipleChoice",
    "multiple_choice": "multipleChoice",
    "mcq":             "multipleChoice",
    "trueorfalse":     "trueOrFalse",
    "true_false":      "trueOrFalse",
    "truefalse":       "trueOrFalse",
    "identification":  "identification",
    "enumeration":     "enumeration",
}


def question_type_ceiling(question_type: Optional[str]) -> Optional[str]:
    """Highest Bloom level a SINGLE question format can validly assess.

    Accepts either spelling ("multipleChoice" or "multiple_choice"). Returns
    ``None`` — not a default — when the type is missing or unrecognised, so a
    caller must decide for itself what "unknown" means. ``tag_questions_with_lilos``
    reports it as unknown; ``build_type_plans`` treats it as permissive so the
    pre-generation check never warns about a format it cannot identify.
    """
    key = (question_type or "").strip()
    if not key:
        return None
    canonical = _QUESTION_TYPE_ALIASES.get(key.lower())
    return QUESTION_TYPE_BLOOM_CEILING.get(canonical) if canonical else None


def assessment_bloom_ceiling(question_types: Optional[Dict[str, bool]]) -> str:
    """Highest Bloom ceiling among the selected objective question types.

    ``question_types`` is the frontend flag dict, e.g.
    ``{"multipleChoice": True, "enumeration": False, ...}``.
    Defaults to ``"analyze"`` (the most permissive objective ceiling) when no
    recognised type is selected, so the check never over-warns by accident.
    """
    selected = [
        t for t, on in (question_types or {}).items()
        if on and t in QUESTION_TYPE_BLOOM_CEILING
    ]
    if not selected:
        return "analyze"
    return max(
        (QUESTION_TYPE_BLOOM_CEILING[t] for t in selected),
        key=lambda b: _BLOOM_ASC.index(b),
    )


def bloom_within_ceiling(bloom_level: Optional[str], ceiling: str) -> bool:
    """True when ``bloom_level`` is at or below ``ceiling`` on the taxonomy.

    Unknown / unrecognised levels return True (treated as compatible — never
    warned about) so parsing quirks don't trigger false mismatch warnings.
    """
    try:
        return _BLOOM_ASC.index((bloom_level or "understand").lower()) <= _BLOOM_ASC.index(ceiling)
    except ValueError:
        return True


# ---------------------------------------------------------------------------
# Flat verb set for fast "line starts with Bloom's verb" detection
# Used by parse_lilos to identify ILO lines in CHED-format syllabi
# ---------------------------------------------------------------------------

_BLOOM_ACTION_VERBS: frozenset = frozenset({
    # Remember
    "define", "list", "recall", "identify", "name", "state", "label",
    "memorize", "recite", "quote", "enumerate", "recognize", "match",
    "locate", "give", "tell",
    # Understand
    "explain", "describe", "summarize", "discuss", "paraphrase",
    "interpret", "compare", "illustrate", "restate", "review", "relate",
    "predict", "infer", "translate", "contrast", "exemplify",
    "distinguish", "express", "comprehend", "classify", "indicate",
    "gain", "acquire", "appreciate",
    # Apply
    "solve", "use", "demonstrate", "compute", "calculate", "apply",
    "construct", "show", "operate", "perform", "practice", "implement",
    "execute", "complete", "utilize", "employ", "present", "model",
    # Analyze
    "differentiate", "examine", "analyze", "analyse", "deconstruct",
    "separate", "categorize", "question", "test", "diagram",
    "investigate", "criticize", "simplify", "associate", "conduct",
    "attribute",
    # Evaluate
    "justify", "critique", "assess", "judge", "defend", "evaluate",
    "recommend", "argue", "prioritize", "decide", "measure", "rate",
    "rank", "conclude", "support", "verify", "determine", "grade",
    "select", "choose",
    # Create
    "design", "develop", "formulate", "propose", "create", "generate",
    "produce", "plan", "compose", "invent", "build", "make", "write",
    "assemble", "prepare", "devise", "draft", "compile", "derive",
    "modify",
    # CHED-specific common verbs
    "cultivate", "foster", "demonstrate",
    # Computing-syllabus verbs. These lead real outcomes in ISU IT/CS syllabi
    # ("Trace recursive execution…", "Represent, traverse, implement, and test
    # binary trees…", "Configure the approved programming environment") and were
    # silently dropped: `classify_bloom` already assigns them a level, but the
    # leading-verb check did not know them, so every line-based priority skipped
    # the outcome entirely.
    "trace", "represent", "configure", "maintain", "validate", "specify",
    "convert", "refactor", "debug", "integrate", "organize", "adapt",
    "optimize", "simulate", "visualize", "communicate", "deploy", "document",
})


def _starts_with_bloom_verb(line: str) -> bool:
    """Return True if line begins with a recognized Bloom's action verb.
    Handles inflected forms (demonstrates→demonstrate, applies→apply, etc.).
    """
    words = line.split()
    if not words or len(words) < 3:
        return False
    # PDF text keeps the punctuation glued to the word. The first word of
    # "Implement, test, and apply linear data structures, ..." is "implement,"
    # and matched nothing, so three of one syllabus's four course outcomes were
    # dropped on every parse while the comma-less first one came through.
    first = words[0].lower().strip('.,;:!?()[]{}"‘’“”\'')
    if not first:
        return False
    # Exact match
    if first in _BLOOM_ACTION_VERBS:
        return True
    # Strip common inflectional suffixes: -s, -es, -ing, -ed
    for suffix in ("ing", "ed", "es", "s"):
        if first.endswith(suffix) and len(first) > len(suffix) + 2:
            stem = first[: -len(suffix)]
            if stem in _BLOOM_ACTION_VERBS:
                return True
            # e.g. "applies" → strip "ies" + add "y" = "apply"
            if suffix == "es" and first.endswith("ies"):
                if first[:-3] + "y" in _BLOOM_ACTION_VERBS:
                    return True
    return False

# ---------------------------------------------------------------------------
# Step 1: English stopwords (no external dependency)
# ---------------------------------------------------------------------------

_STOPWORDS = frozenset({
    'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'on',
    'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
    'through', 'during', 'before', 'after', 'above', 'below', 'from',
    'up', 'down', 'out', 'off', 'over', 'under', 'and', 'but', 'or',
    'nor', 'so', 'yet', 'both', 'either', 'neither', 'not', 'no', 'it',
    'its', 'this', 'that', 'these', 'those', 'i', 'me', 'my', 'we',
    'our', 'you', 'your', 'he', 'she', 'they', 'them', 'their', 'what',
    'which', 'who', 'whom', 'how', 'when', 'where', 'why', 'all', 'each',
    'every', 'any', 'few', 'more', 'most', 'other', 'some', 'such',
    'than', 'too', 'very', 's', 't', 'just', 'also', 'as',
})

# ---------------------------------------------------------------------------
# Suffix stemming rules (no NLTK / spaCy needed)
# ---------------------------------------------------------------------------

_STEM_SUFFIXES = [
    ('ational', 'ate'), ('tional', 'tion'), ('enci', 'ence'), ('anci', 'ance'),
    ('izer', 'ize'), ('izing', 'ize'), ('ising', 'ise'),
    ('ation', 'ate'), ('ator', 'ate'), ('alism', 'al'), ('iveness', 'ive'),
    ('fulness', 'ful'), ('ousness', 'ous'),
    ('ments', 'ment'), ('ment', ''), ('nesses', 'ness'), ('ness', ''),
    ('alities', 'al'), ('ality', 'al'), ('ative', ''), ('alize', 'al'),
    ('ically', 'ic'), ('ical', 'ic'),
    ('ings', 'ing'), ('ing', ''), ('edly', 'ed'), ('edness', 'ed'),
    ('ers', 'er'), ('er', ''),
    ('ies', 'y'), ('ied', 'y'), ('es', ''), ('ed', ''), ('s', ''),
]


def _stem(word: str) -> str:
    """Apply suffix stripping to approximate word stem."""
    if len(word) < 4:
        return word
    for suffix, replacement in _STEM_SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[:-len(suffix)] + replacement
    return word


def _normalize_text(text: str) -> str:
    """
    Step 1 -- Text Normalization.

    1. Lowercase
    2. Remove punctuation
    3. Remove stopwords
    4. Suffix stemming
    Returns a space-joined string of normalized tokens.
    """
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in _STOPWORDS and len(t) >= 2]
    tokens = [_stem(t) for t in tokens]
    return ' '.join(tokens)


def _combine_alignment_parts(*parts: Optional[str]) -> str:
    """Join non-empty text parts while avoiding exact duplicates."""
    combined: List[str] = []
    seen = set()
    for part in parts:
        cleaned = (part or '').strip()
        if cleaned and cleaned not in seen:
            combined.append(cleaned)
            seen.add(cleaned)
    return ' '.join(combined)


def _lilo_alignment_text(lilo: Dict) -> str:
    """Text bundle used for semantic matching: outcome + topic block context."""
    return _combine_alignment_parts(
        lilo.get('enriched_text'),
        lilo.get('text'),
        lilo.get('topic_title'),
        lilo.get('learning_content'),
    )


def _lilo_content_text(lilo: Dict) -> str:
    """
    Syllabus course-plan context for Channel 3 (content Jaccard).

    Includes the LILO's own outcome text so that ISU OBE syllabi (where
    topic_title is a short bullet header) still produce meaningful keyword
    overlap with generated questions that use domain-specific vocabulary.
    """
    return _combine_alignment_parts(
        lilo.get('text'),
        lilo.get('topic_title'),
        lilo.get('learning_content'),
    )


def _content_block_key(item: Dict) -> str:
    return "||".join([
        (item.get("topic_title") or "").strip(),
        (item.get("learning_content") or "").strip(),
        (item.get("week") or "").strip(),
    ])


def _content_block_alignment_text(block: Dict) -> str:
    return _combine_alignment_parts(
        block.get("topic_title"),
        block.get("learning_content"),
        block.get("ilo_summary"),
    )


def build_content_blocks_from_lilos(lilos: List[Dict]) -> List[Dict]:
    """
    Build weekly Course Plan content blocks from parsed ILO metadata.

    Each block corresponds to one syllabus row/week topic and contains the
    topic title, learning content, PO coverage, and the ILOs that belong to it.
    Also annotates each ILO with ``content_block_id`` in-place.
    """
    if not lilos:
        return []

    blocks_by_key: Dict[str, Dict] = {}
    block_counter = 1
    used_block_ids = {
        str(lilo.get("content_block_id")).strip()
        for lilo in lilos
        if str(lilo.get("content_block_id") or "").strip()
    }

    def _next_block_id() -> str:
        nonlocal block_counter
        while True:
            candidate = f"CONTENT-{block_counter}"
            block_counter += 1
            if candidate not in used_block_ids:
                used_block_ids.add(candidate)
                return candidate

    for lilo in lilos:
        key = _content_block_key(lilo)
        if key not in blocks_by_key:
            existing_block_id = str(lilo.get("content_block_id") or "").strip()
            block_id = existing_block_id or _next_block_id()
            if existing_block_id:
                used_block_ids.add(existing_block_id)
            blocks_by_key[key] = {
                "id": block_id,
                "topic_title": lilo.get("topic_title"),
                "learning_content": lilo.get("learning_content"),
                "week": lilo.get("week"),
                "po_codes": set(),
                "ga_codes": set(),
                "co_ids": set(),
                "ilos": [],
            }

        block = blocks_by_key[key]
        lilo["content_block_id"] = block["id"]
        block["ilos"].append({
            "id": lilo.get("id"),
            "text": lilo.get("text"),
            "bloom_level": lilo.get("bloom_level"),
            "co_id": lilo.get("co_id"),
        })
        block["po_codes"].update(lilo.get("po_codes", []))
        block["ga_codes"].update(lilo.get("ga_codes", []))
        if lilo.get("co_id"):
            block["co_ids"].add(lilo["co_id"])

    blocks: List[Dict] = []
    for block in blocks_by_key.values():
        block["po_codes"] = sorted(block["po_codes"])
        block["ga_codes"] = sorted(block["ga_codes"])
        block["co_ids"] = sorted(block["co_ids"])
        block["ilo_ids"] = [ilo["id"] for ilo in block["ilos"] if ilo.get("id")]
        block["ilo_summary"] = " ".join(ilo.get("text", "") for ilo in block["ilos"])
        blocks.append(block)

    # Observability (this function previously had NO logging): show how LILOs
    # were grouped into content blocks, so an edited/realigned syllabus is
    # traceable in the parse/align flow.
    logger.info("build_content_blocks_from_lilos: %d LILOs -> %d content blocks",
                len(lilos or []), len(blocks))
    for b in blocks:
        logger.debug("  block %s topic=%r LILOs=%s COs=%s",
                     b.get("id"), (b.get("topic_title") or "")[:50],
                     b.get("ilo_ids"), b.get("co_ids"))
    return blocks


def select_content_blocks_for_topic(
    content_blocks: List[Dict],
    topic: str,
    source_text: str = "",
    top_n: int = 1,
    threshold: float = 0.25,
) -> List[Dict]:
    """
    Match an uploaded quiz topic/PPT against syllabus Course Plan blocks.

    For a quiz the caller should always get the SINGLE best-matching block
    (top_n=1 by default).  Only blocks within a tight score margin of the top
    winner are included, preventing unrelated blocks from leaking in.

    Returns at most ``top_n`` blocks.  If no block clears the threshold the
    single best candidate is still returned so the quiz always has a scope.
    """
    if not topic or not content_blocks:
        return content_blocks

    topic_sig = _topic_tokens(topic)
    logger.info(
        f"[select_content_blocks] topic='{topic}' tokens={sorted(topic_sig)} "
        f"blocks={len(content_blocks)}"
    )

    # ── Pass 1: token overlap on topic_title / learning_content / week ──────
    # Score every block; keep only the top-N winners within a tight margin.
    # A single-token topic is unreliable — skip pass 1 for it.
    if len(topic_sig) >= 2:
        scored_p1 = []
        for block in content_blocks:
            candidate_text = _combine_alignment_parts(
                block.get("topic_title"),
                block.get("learning_content"),
                block.get("week"),
            )
            if not candidate_text:
                continue
            candidate_sig = _topic_tokens(candidate_text)
            overlap = (len(topic_sig & candidate_sig) / max(len(topic_sig), 1)) if topic_sig else 0.0
            logger.debug(
                f"  [pass1] {block['id']:12s} overlap={overlap:.2f} "
                f"topic_title={str(block.get('topic_title',''))[:40]!r}"
            )
            if overlap >= 0.60:
                scored_p1.append((overlap, block))

        if scored_p1:
            scored_p1.sort(key=lambda x: x[0], reverse=True)
            best_score = scored_p1[0][0]
            # Only include blocks within 10 % of the top score
            margin = 0.10
            matched_blocks = [
                b for s, b in scored_p1
                if s >= best_score - margin
            ][:top_n]
            logger.info(f"[select_content_blocks] Pass1 matched: {[b['id'] for b in matched_blocks]}")
            return matched_blocks

    # ── Pass 2: token overlap on ilo_summary ─────────────────────────────────
    scored_p2 = []
    for block in content_blocks:
        ilo_text = (block.get("ilo_summary") or "").strip()
        if not ilo_text:
            continue
        candidate_sig = _topic_tokens(ilo_text)
        overlap = (len(topic_sig & candidate_sig) / max(len(topic_sig), 1)) if topic_sig else 0.0
        logger.debug(
            f"  [pass2] {block['id']:12s} overlap={overlap:.2f} "
            f"ilo_summary_start={ilo_text[:40]!r}"
        )
        if overlap >= 0.80:
            scored_p2.append((overlap, block))

    if scored_p2:
        scored_p2.sort(key=lambda x: x[0], reverse=True)
        best_score = scored_p2[0][0]
        matched_blocks = [
            b for s, b in scored_p2
            if s >= best_score - 0.10
        ][:top_n]
        logger.info(f"[select_content_blocks] Pass2 matched: {[b['id'] for b in matched_blocks]}")
        return matched_blocks

    import numpy as np

    # ── Pass 3: embedding fallback — topic-only query ────────────────────────
    logger.info("[select_content_blocks] Pass1+2 failed, using embedding fallback (topic-only query)")
    query = topic.strip()

    model = _get_model()
    query_vec = model.encode([query], normalize_embeddings=True)[0]

    scored = []
    for block in content_blocks:
        block_text = _content_block_alignment_text(block)
        block_vec = model.encode([block_text], normalize_embeddings=True)[0]
        sim = float(np.dot(query_vec, block_vec))
        scored.append((sim, block))

    scored.sort(key=lambda x: x[0], reverse=True)
    for sim, block in scored[:5]:
        logger.info(f"  [pass3-emb] {block['id']:12s} sim={sim:.4f} title={str(block.get('topic_title',''))[:40]!r}")

    best_score = scored[0][0] if scored else 0.0
    # For the embedding pass use a tighter margin (5 %) so a clear winner wins alone
    margin = 0.05
    relevant = [
        b for s, b in scored
        if s >= max(threshold, best_score - margin)
    ][:top_n]

    # Always return at least one block
    if not relevant and scored:
        relevant = [scored[0][1]]

    selected_ids = [b["id"] for b in relevant]
    logger.info(f"[select_content_blocks] Pass3 selected: {selected_ids}")
    return [block for block in content_blocks if block["id"] in selected_ids]


def _topic_tokens(text: str) -> set:
    """Normalized token set used for syllabus-topic matching."""
    return {tok for tok in _normalize_text(text).split() if len(tok) >= 3}


def _extract_po_codes(line: str) -> List[str]:
    """Extract PO codes like `PO1, PO2, PO6` from syllabus lines."""
    return [code.upper().replace(' ', '') for code in re.findall(r'PO\s*\d+', line or '', re.IGNORECASE)]


# ISU Graduate Attribute normalisation map.
# "IF" is a mammoth text-run split artifact for "IFK" (cell text split across runs).
_GA_NORMALISE: Dict[str, str] = {
    'IF': 'IFK', 'IFK': 'IFK', 'CEL': 'CEL', 'CM': 'Cm', 'CP': 'Cp', 'LL': 'LL',
}


def _extract_ga_codes(line: str) -> List[str]:
    """Extract ISU Graduate Attribute codes (Cm, IFK, Cp, CEL, LL) from a syllabus line.

    Handles the mammoth text-run split artifact where IFK appears as IF.
    """
    codes = re.findall(r'IFK?|CEL|Cm|Cp|LL', line or '', re.IGNORECASE)
    seen: set = set()
    result: List[str] = []
    for c in codes:
        norm = _GA_NORMALISE.get(c.upper(), c)
        if norm not in seen:
            seen.add(norm)
            result.append(norm)
    return result


def parse_program_outcomes(lilo_text: str) -> List[Dict]:
    """
    Extract program outcomes from the syllabus header section.

    Expected format in CHED-style syllabi:
      PROGRAM OUTCOMES
      ...
      PO1 Apply ...
      PO2 Identify ...
      ...
    """
    if not lilo_text or not lilo_text.strip():
        return []

    lines = [l.strip() for l in lilo_text.splitlines() if l.strip()]
    in_po_section = False
    program_outcomes: List[Dict] = []
    current: Optional[Dict] = None

    # Expanded PO section header patterns
    _PO_SECTION_HEADER = re.compile(
        r'^(?:program(?:me)?\s+outcomes?'             # PROGRAM OUTCOMES
        r'|program(?:me)?\s+learning\s+outcomes?'     # PROGRAM LEARNING OUTCOMES
        r'|intended\s+program(?:me)?\s+outcomes?'     # INTENDED PROGRAM OUTCOMES
        r'|graduate\s+attributes?'                    # GRADUATE ATTRIBUTES
        r'|graduate\s+outcomes?'                      # GRADUATE OUTCOMES
        r')(?:\s*\([^)]*\))?\s*:?$',
        re.IGNORECASE,
    )
    # PO/GA entry: "PO1 text", "GA2 – text", "GA3: text", "PO 4. text"
    _PO_ENTRY = re.compile(
        r'^((?:PO|GA|IO)\s*\d+)\s*[-–:.\)]?\s+(.+)$',
        re.IGNORECASE,
    )

    for line in lines:
        if not in_po_section:
            if _PO_SECTION_HEADER.match(line):
                in_po_section = True
            continue

        if re.match(r'^(?:course\s+description|course\s+outcomes?|course\s+plan|prerequisite|references?|course\s+objectives?)', line, re.IGNORECASE):
            break

        match = _PO_ENTRY.match(line)
        if match:
            if current:
                program_outcomes.append(current)
            current = {
                'id': re.sub(r'\s+', '', match.group(1).upper()),
                'text': match.group(2).strip(),
            }
            continue

        if current and len(line.split()) >= 3 and not re.match(r'^[A-Z][A-Z\s]+$', line):
            current['text'] += f" {line}"

    if current:
        program_outcomes.append(current)

    # Strip GA* entries — Graduate Attributes are institutional-level, not course-level POs.
    # Only PO* (and IO* if present) belong in the program outcomes list.
    program_outcomes = [po for po in program_outcomes if re.match(r'^PO\d+$', po['id'], re.IGNORECASE)]

    return program_outcomes


# ---------------------------------------------------------------------------
# KeyBERT model (lazy-loaded, optional)
# ---------------------------------------------------------------------------

_keybert_model = None
_keybert_available: Optional[bool] = None  # None = not yet checked


def _get_keybert():
    """Lazy-load KeyBERT. Returns None if keybert is not installed."""
    global _keybert_model, _keybert_available
    if _keybert_available is False:
        return None
    if _keybert_model is None:
        with _model_lock:  # reuse the model lock — serialise the one-time load
            if _keybert_model is None and _keybert_available is not False:
                try:
                    from keybert import KeyBERT
                    # Reuse the same sentence-transformer model for consistency
                    _keybert_model = KeyBERT(model=_get_model())
                    _keybert_available = True
                    logger.info("KeyBERT loaded (using same sentence-transformer backbone)")
                except ImportError:
                    _keybert_available = False
                    logger.warning(
                        "keybert not installed — falling back to stemmed-token Jaccard. "
                        "Run: pip install keybert  for better keyword extraction."
                    )
    return _keybert_model


def _extract_keywords(text: str) -> set:
    """
    Step 3 -- Keyword / Concept Extraction.

    Preferred: KeyBERT (BERT-based semantic keyword extraction).
    Fallback:  Stemmed tokens after stopword removal (original method).

    Returns a set of keyword strings (deduplicated).
    """
    kb = _get_keybert()
    if kb is not None:
        try:
            # Extract up to 10 keywords; keyphrase_ngram_range=(1,2) catches
            # compound concepts like "merge sort" or "binary search".
            keyphrases = kb.extract_keywords(
                text,
                keyphrase_ngram_range=(1, 2),
                stop_words='english',
                top_n=10,
                use_mmr=True,       # Maximal Marginal Relevance — diverse set
                diversity=0.5,
            )
            # keyphrases is [(phrase, score), ...]; return just the phrases
            kw_set = {phrase.lower() for phrase, _ in keyphrases if phrase}
            if kw_set:
                return kw_set
        except Exception as e:
            logger.warning(f"KeyBERT extraction failed, falling back: {e}")

    # Fallback: original stemmed-token method
    normalized = _normalize_text(text)
    return set(normalized.split())


def _expand_phrases(keywords: set) -> set:
    """
    Add the individual words of every multi-word keyphrase to the set.

    KeyBERT is called with keyphrase_ngram_range=(1, 2), so it does emit some
    unigrams — but it selects with MMR at diversity=0.5, which deliberately
    favours *diverse* phrases and therefore rarely returns both a bigram and
    its component words. "frequent itemsets" then misses a question that says
    "frequent" and "itemsets" separately, because Jaccard compares strings.
    The question side and Channel 2 already expand; Channel 3 did not.

    Measured over the 12 saved assessments (285 questions, 392 distinct texts):
    expansion changes 99% of extracted sets, adding ~14 terms to a mean set of
    10, and drops Channel 3's zero-score rate from 55% to 7%. The other three
    channels are unaffected.
    """
    expanded = set(keywords)
    for kw in keywords:
        expanded.update(kw.split())
    return expanded


def _jaccard_score(set_a: set, set_b: set) -> float:
    """Keyword Jaccard similarity: |A cap B| / |A cup B|."""
    if not set_a and not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union else 0.0


def _bloom_match_score(q_bloom: str, lilo_bloom: str) -> float:
    """
    Bloom Match Score with cognitive proximity.
      1.0  if same level (exact match)
      0.75 if adjacent level (1 step apart on the taxonomy)
      0.5  if 2 levels apart
      0.25 if 3+ levels apart
    Proximity reflects that adjacent cognitive demands are more related.
    """
    try:
        diff = abs(_BLOOM_ASC.index(q_bloom) - _BLOOM_ASC.index(lilo_bloom))
        if diff == 0: return 1.0
        if diff == 1: return 0.75
        if diff == 2: return 0.5
        return 0.25
    except ValueError:
        return 0.5


def _bloom_gap(q_bloom: str, lilo_bloom: str) -> Optional[int]:
    """Signed taxonomy distance from the LILO's Bloom level to the question's.

      0   question sits at the level the LILO targets
     -1   question is ONE level BELOW target (e.g. remember vs understand)
     +2   question is TWO levels ABOVE target (e.g. evaluate vs apply)
      None  either level is missing or unrecognised

    Sign matters for the report: a question that undershoots its outcome is a
    coverage problem, one that overshoots is a fairness problem, and
    ``_bloom_match_score`` collapses both into the same number.
    """
    try:
        return _BLOOM_ASC.index(q_bloom) - _BLOOM_ASC.index(lilo_bloom)
    except ValueError:
        return None


def _format_ceiling_flags(
    q: Dict, question_bloom: Optional[str]
) -> Tuple[Optional[str], Optional[bool]]:
    """(format_ceiling, bloom_above_ceiling) for one question.

    ``bloom_above_ceiling`` answers a different question from ``bloom_aligned``:
    not "does the question sit at the level its outcome targets" but "can this
    question FORMAT carry the level detected from its wording at all". A
    True/False item phrased as an analyze task is above its ceiling even when it
    matches its LILO exactly — the two flags are meant to coexist.

    Both are ``None`` when the type or the level is unrecognised. ``False`` there
    would assert "this item is fine", which is a claim we cannot make about
    something we did not measure — same convention as ``bloom_gap``.
    """
    ceiling = question_type_ceiling(q.get("type"))
    if ceiling is None or (question_bloom or "").lower() not in _BLOOM_ASC:
        return ceiling, None
    return ceiling, not bloom_within_ceiling(question_bloom, ceiling)


# ---------------------------------------------------------------------------
# Step 2: Bloom level detection
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Framing-phrase disambiguation for classify_bloom
#
# In "Develop the ability to identify patterns", "develop" is only a FRAMING
# verb — the real cognitive demand is the inner verb ("identify").  Without
# this, every "Develop the ability to / proficiency in / skills in …" outcome
# is mis-tagged "create" (because "develop" is a create verb).
#
# We treat "develop/build/gain/…" as framing ONLY when it is followed by a
# *framing noun* (ability/proficiency/skill/understanding/…).  Direct objects
# like "Develop a recommender system" have no framing noun, do NOT match here,
# and keep their normal "create" classification below.
# ---------------------------------------------------------------------------
_FRAMING_RE = re.compile(
    r"^(?:develop|build|gain|acquire|enhance|improve|strengthen|cultivate|foster|demonstrate|show)\s+"
    r"(?:(?:a|an|the|their|its|your|our|deep|advanced|basic|strong|solid|comprehensive|broad|good|thorough|sound|in-depth)\s+)*"
    r"(?P<noun>ability|abilities|proficiency|skill|skills|competenc\w*|capacity|capabilit\w*|"
    r"understanding|comprehension|knowledge|expertise|mastery|familiarity)\s+"
    r"(?:to|in|of|with|on)\s+(?P<inner>.+)$",
    re.IGNORECASE,
)

# Safe gerund → base-verb map.  NEVER blindly strip "ing" (using→us, mining→min,
# analyzing→analyz are all wrong).  Unknown gerunds are left unchanged.
_GERUND_MAP: Dict[str, str] = {
    "using": "use", "applying": "apply", "analyzing": "analyze",
    "identifying": "identify", "processing": "process", "creating": "create",
    "designing": "design", "evaluating": "evaluate", "building": "build",
    "modeling": "model", "interpreting": "interpret", "implementing": "implement",
    "extracting": "extract", "computing": "compute", "detecting": "detect",
    "developing": "develop", "constructing": "construct", "formulating": "formulate",
    "solving": "solve", "calculating": "calculate", "performing": "perform",
}

# Outcome-only purpose-clause elevation (is_outcome=True).
# In "Use generative models to generate …" the first verb is only an ENABLER;
# for a learning outcome the deliverable lives in the "to <verb>" purpose clause.
# Gated to enabler first-verbs + STRONG higher-order purpose verbs — weak create
# verbs (make/plan/write/organize) are intentionally excluded to avoid
# over-elevating phrases like "use a checklist to make decisions".
# "apply" is deliberately NOT an enabler: unlike "use/utilize", applying a
# technique IS the cognitive demand of an outcome — "Apply mutual information
# to assess information gain" tests APPLY, the purpose clause just names what
# the technique measures. Elevating it produced false evaluate/create levels.
_ENABLER_FIRST_VERBS = {
    "use", "utilize", "employ", "demonstrate", "perform", "operate",
}
_STRONG_PURPOSE_LEVEL: Dict[str, str] = {
    "design": "create", "create": "create", "develop": "create",
    "generate": "create", "formulate": "create", "construct": "create",
    "compose": "create", "invent": "create", "devise": "create",
    "produce": "create", "build": "create", "model": "create",
    "evaluate": "evaluate", "assess": "evaluate", "critique": "evaluate",
    "judge": "evaluate", "appraise": "evaluate", "justify": "evaluate",
}


def _framing_noun_default(noun: str) -> str:
    """Conservative level when the inner clause has no recognizable action verb.

    Knowledge-based framing → understand; skill-based framing → apply.
    Object nouns (system/model) never reach here — they don't match _FRAMING_RE.
    """
    n = noun.lower()
    if n.startswith(("understand", "comprehension", "knowledge", "familiar")):
        return "understand"
    # ability / skill / proficiency / competence / capacity / capability /
    # expertise / mastery → skill-development context → apply
    return "apply"


def _starts_with_level(text: str, level: str) -> bool:
    """True if the first (one- or two-word) verb of `text` belongs to `level`."""
    words = text.lower().split()
    if not words:
        return False
    first = words[0]
    for verb in BLOOMS_VERBS[level]:
        if ' ' not in verb and re.match(r'^' + re.escape(verb) + r'(?:s|es|ed|ing|ies)?$', first):
            return True
    if len(words) >= 2:
        two = words[0] + ' ' + words[1]
        for verb in BLOOMS_VERBS[level]:
            if ' ' in verb and two.startswith(verb):
                return True
    return False


# Contexts that license a VERB in the position that follows them. Anything else
# preceding a candidate token means the token is sitting inside a noun phrase.
#
# A whitelist rather than a determiner blacklist, because the false triggers
# measured across the 285-question corpus are compound nouns — "LDA model",
# "network measure", "frame rate", "minimum support threshold" — where the
# determiner is several words back and only the immediate modifier is visible.
_VERB_LICENSING_WORDS = frozenset({
    # coordinators / subordinators / infinitive marker
    "and", "or", "then", "but", "also", "to", "that", "thus", "hence", "so",
    # wh-words that front a clause
    "which", "what", "who", "how", "why", "if", "when", "while", "whether",
    # auxiliaries, modals, copulas
    "does", "do", "did", "will", "would", "can", "could", "should", "must",
    "may", "might", "shall", "is", "are", "was", "were", "be", "been", "being",
    "has", "have", "had", "not", "cannot",
    # subject pronouns
    "you", "we", "they", "it", "he", "she", "i", "one", "student", "students",
    # sequencing / imperative lead-ins
    "first", "next", "now", "please", "best", "most",
})

# Words that are Bloom cues only in a genuine interrogative ("When DID Dijkstra…",
# "Who IS the author…"). Elsewhere they open a subordinate clause — "When analyzing
# a dataset, …" — and carry no cognitive signal at all. The discriminator is the
# next word: a real question puts an auxiliary or copula there, a subordinate
# clause puts a participle or subject.
_STEM_ONLY_CUES = frozenset({"when", "where", "who"})
_INTERROGATIVE_AUX = frozenset({
    "did", "do", "does", "is", "are", "was", "were",
    "will", "can", "could", "should", "would", "has", "have", "had",
})


def _is_inert_cue(text_lower: str, pos: int, base: str) -> bool:
    """True when a stem-only cue at `pos` is NOT introducing a question.

    Guards the single biggest false trigger on generated question stems: every
    "When analyzing a large dataset, …" was classifying as `remember` off the
    conjunction alone.
    """
    if base not in _STEM_ONLY_CUES:
        return False
    nxt = text_lower[pos + len(base):].lstrip().split(' ', 1)[0].strip('.,;:?!')
    return nxt not in _INTERROGATIVE_AUX


def _is_noun_usage(text_lower: str, pos: int, token: str = "") -> bool:
    """True when the token at `pos` reads as a noun rather than a Bloom verb.

    A cheap stand-in for POS tagging — good enough to keep domain vocabulary out
    of the verb scan without pulling in a tagger. Verb position is: start of
    text, right after punctuation, after a licensing word, or after an adverb.
    Everything else is inside a noun phrase.

    Known residual: a sentence-initial noun ("Model X yields …") still reads as a
    verb, because suppressing that position would break every imperative stem.
    """
    if token and text_lower[pos + len(token):pos + len(token) + 1] == "(":
        return True  # "support(milk) = 0.8" — a function call, never a verb
    before = text_lower[:pos].rstrip()
    if not before or before[-1] in '.?!;:,()[]{}"\'':
        return False  # sentence start or post-punctuation → verb position
    prev = before.split()[-1].strip('.,;:?!()[]{}"\'')
    if prev in _VERB_LICENSING_WORDS or prev.endswith('ly'):
        return False  # licensing word or adverb → verb position
    return True


def _contains_bloom_verb(text: str) -> bool:
    """True if any recognized Bloom verb appears anywhere in `text`."""
    tl = text.lower()
    for level in _BLOOMS_ORDER:
        for verb in BLOOMS_VERBS[level]:
            if re.search(r'\b' + re.escape(verb) + r'\b', tl):
                return True
    return False


def classify_bloom(text: str, _strip_framing: bool = True, is_outcome: bool = False) -> str:
    """
    Step 2 -- Bloom Level Detection.

    Strategy:
      0. Framing-phrase disambiguation — "Develop the ability to <verb> …" is
         classified by the INNER verb, never by the framing verb "develop".
      0b. Purpose-clause elevation (OUTCOMES only) — "Use X to generate …" is
         classified by the higher-order purpose verb, not the enabler "use".
      1. Primary verb check — look at the FIRST action verb in the text.
         For an ILO/question, the first verb is almost always the cognitive
         demand verb ("Apply...", "Analyze...", "Design...").
      2. If no recognized verb starts the sentence, scan the whole text
         for the HIGHEST Bloom level present (descending create→remember).
      3. Fall back to interrogative-pattern heuristics, then "understand".

    `_strip_framing` is an internal guard flipped to False on the recursive
    call so the inner clause is not re-stripped (prevents infinite recursion).
    """
    text_lower = text.lower().strip()
    words = text_lower.split()

    # ── Strategy 0: framing-phrase disambiguation ──────────────────────────
    # "Develop the ability to <verb> …" / "Develop proficiency in using …" —
    # classify by the INNER cognitive verb.  A framing phrase may be "create"
    # ONLY if the inner clause itself starts with a genuine create verb; it
    # must never inherit "create" from the framing verb.
    if _strip_framing:
        _fm = _FRAMING_RE.match(text_lower)
        if _fm:
            noun = _fm.group("noun")
            inner = _fm.group("inner").strip()
            _iw = inner.split()
            if _iw and _iw[0] in _GERUND_MAP:
                _iw[0] = _GERUND_MAP[_iw[0]]
                inner = " ".join(_iw)
            if _contains_bloom_verb(inner):
                level = classify_bloom(inner, _strip_framing=False)
                if level == "create" and not _starts_with_level(inner, "create"):
                    level = _framing_noun_default(noun)
                return level
            # No recognizable action verb in the inner clause → noun-based default
            return _framing_noun_default(noun)

    # ── Strategy 0b: purpose-clause elevation (OUTCOMES only) ──────────────
    # "Use generative models to generate …" — the first verb is just an enabler;
    # for a learning outcome the real demand is the "to <strong higher-order
    # verb>" purpose clause.  Never lowers — only elevates to create/evaluate.
    # Questions keep first-verb authority (is_outcome stays False for them).
    if is_outcome and words:
        first = words[0]
        first_cands = {first, _GERUND_MAP.get(first, ""), re.sub(r'(?:s|es|ed|ing)$', '', first)}
        if first_cands & _ENABLER_FIRST_VERBS:
            for _m in re.finditer(r'\bto\s+([a-z]+)', text_lower):
                pv = _m.group(1)
                pv_base = pv[:-3] if pv.endswith("ing") else pv
                lvl = _STRONG_PURPOSE_LEVEL.get(pv) or _STRONG_PURPOSE_LEVEL.get(pv_base)
                if lvl:
                    return lvl

    # ── Strategy 1: primary verb (first word or first two-word phrase) ──
    if words:
        first = words[0]
        # Check single-word first verb against every level (remember→create order
        # so the lowest true match wins — primary verb is authoritative)
        for level in reversed(_BLOOMS_ORDER):  # remember up to create
            for verb in BLOOMS_VERBS[level]:
                # Only single-word verbs for primary match
                if ' ' not in verb and re.match(r'^' + re.escape(verb) + r'(?:s|es|ed|ing|ies)?$', first):
                    if _is_inert_cue(text_lower, 0, verb):
                        continue  # "When analyzing …" — a conjunction, not the demand
                    return level
        # Two-word primary phrase (e.g. "break down", "give example")
        if len(words) >= 2:
            two = words[0] + ' ' + words[1]
            for level in reversed(_BLOOMS_ORDER):
                for verb in BLOOMS_VERBS[level]:
                    if ' ' in verb and two.startswith(verb):
                        return level

    # ── Strategy 2: highest-level verb anywhere in text ──────────────────
    # Exclude verbs that appear *only* in an infinitive/purpose context
    # (e.g. "wants to analyze…", "needs to apply…") because those describe
    # the *goal* of the scenario, not the cognitive demand of the question.
    _verbs_only_in_infinitive: set = set()
    for m in re.finditer(r'\bto\s+([a-z]{4,})\b', text_lower):
        v = m.group(1)
        all_occ = list(re.finditer(r'\b' + re.escape(v) + r'\b', text_lower))
        inf_occ = [o for o in all_occ if text_lower[max(0, o.start() - 3):o.start()] == 'to ']
        if len(all_occ) == len(inf_occ):
            _verbs_only_in_infinitive.add(v)

    for level in _BLOOMS_ORDER:  # create → remember (highest wins)
        for verb in BLOOMS_VERBS[level]:
            base = verb.split()[0]  # use first word for single-word matching
            if base in _verbs_only_in_infinitive:
                continue  # skip — only appears as purpose context
            for _m in re.finditer(r'\b' + re.escape(verb) + r'\b', text_lower):
                if _is_noun_usage(text_lower, _m.start(), verb):
                    continue  # skip — "minimum support", "LDA model": a noun
                if _is_inert_cue(text_lower, _m.start(), base):
                    continue  # skip — "When analyzing…" is a conjunction, not a cue
                return level

    # ── Strategy 3: interrogative heuristics ─────────────────────────────
    if re.search(r'\bwhy\b', text_lower):
        return "analyze"
    if re.search(r'\bwhich\b.*\bbest\b|\bbest\b.*\bwhich\b', text_lower):
        return "evaluate"
    # "Which of the following is NOT…" / "which is not" → analysis
    # (identifying an incorrect item requires distinguishing, not mere recall)
    if re.search(r'\bwhich\b.{0,50}\bnot\b', text_lower):
        return "analyze"
    # "Which of the following…" — further disambiguate before returning "remember".
    # If the question contains application/analysis signal words, classify higher.
    if re.search(r'\bwhich\b', text_lower):
        # Application signals: apply, use, select, choose, determine, compute, calculate
        if re.search(r'\b(apply|applies|applied|use|select|choose|determine|compute|calculate|identify the (best|most|correct)|most (effective|appropriate|suitable))\b', text_lower):
            return "apply"
        # Analysis signals: compare, distinguish, differentiate, analyze, break down, examine
        if re.search(r'\b(compare|distinguish|differentiat|examin|break\s+down|investigat|decompos)\b', text_lower):
            return "analyze"
        return "remember"
    if re.search(r'\bhow\b', text_lower):
        return "understand"
    if re.search(r'\bwhat (is|are|was|were)\b', text_lower):
        return "remember"
    if re.search(r'\bwhat\b', text_lower):
        return "understand"

    return "understand"


# ---------------------------------------------------------------------------
# Step 4: Embedding model (lazy-loaded)
# ---------------------------------------------------------------------------

_model = None

# Guards the lazy singleton loads (B6): without it, two concurrent first-requests
# could both enter and load the ~420MB model twice (a transient RAM/CPU spike).
#
# Must be re-entrant: _get_keybert() holds this lock while it builds KeyBERT, and
# KeyBERT is constructed with _get_model() as its backbone — so the same thread
# re-enters here. With a plain Lock that is a self-deadlock that hangs forever.
#
# The served app does not hit it today only because main.py's lifespan preloads
# the model, so _get_model() returns at its `_model is None` check without ever
# taking the lock. That preload is best-effort (its failure is caught and
# logged), and tag_questions_with_lilos extracts keywords at Step 3 before
# calling _get_model() at Step 4 — so a skipped preload would turn the first
# tagging request into a permanent hang. RLock keeps that a slow load instead.
_model_lock = threading.RLock()


# Model name — change here to swap globally
_MODEL_NAME = 'multi-qa-mpnet-base-dot-v1'


def _get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:  # double-checked: another thread may have loaded it
                try:
                    from sentence_transformers import SentenceTransformer
                    _model = SentenceTransformer(_MODEL_NAME)
                    logger.info(f"Sentence transformer model loaded: {_MODEL_NAME}")
                except ImportError:
                    raise RuntimeError(
                        "sentence-transformers is not installed. "
                        "Run: pip install sentence-transformers"
                    )
    return _model


# ---------------------------------------------------------------------------
# Softmax confidence (reporting only -- not used as the threshold gate)
# ---------------------------------------------------------------------------

def _softmax_confidence(scores, temperature: float = 0.10) -> float:
    """
    Temperature-scaled softmax confidence of the winning composite score.
    tau = 0.10 gives a sharp distribution:
      > 0.70  clear best LILO  |  0.40-0.70  moderate  |  < 0.40  ambiguous
    """
    import numpy as np
    s = np.array(scores, dtype=float) / temperature
    s -= s.max()
    exp_s = np.exp(s)
    return float(exp_s.max() / exp_s.sum())


# ---------------------------------------------------------------------------
# Step 9/10: Conditional LLM validation (Gemini, rate-limited)
# ---------------------------------------------------------------------------

_LLM_LAST_CALL_TIME: float = 0.0
_LLM_MIN_INTERVAL: float = 5.0   # seconds between calls (max 12/min free tier)


def _llm_validate_alignment(
    question_text: str,
    candidates: List[Dict],
) -> Optional[str]:
    """
    Step 9/10 -- Conditional LLM Validation.

    Called ONLY when top-2 composite scores differ by < close_score_delta.
    Sends question + top-3 LILO candidates to Gemini (free tier, REST API).
    Returns the chosen lilo id string, or None on failure.
    Rate-limited: minimum 5 s gap between consecutive calls.
    """
    global _LLM_LAST_CALL_TIME

    elapsed = time.time() - _LLM_LAST_CALL_TIME
    if elapsed < _LLM_MIN_INTERVAL:
        time.sleep(_LLM_MIN_INTERVAL - elapsed)

    api_key = get_next_key()
    if not api_key:
        logger.warning("[GeminiKeyPool] No API keys configured -- skipping LLM tiebreaker validation")
        return None

    cand_lines = "\n".join(
        f"  {c['lilo_id']}: \"{c['lilo_text']}\"  "
        f"(cosine={c['cosine']:.3f}, jaccard={c['jaccard']:.3f}, "
        f"bloom={c['bloom_match']:.1f}, composite={c['composite']:.3f})"
        for c in candidates
    )

    prompt = (
        "You are an expert curriculum alignment evaluator.\n\n"
        f"Question: \"{question_text}\"\n\n"
        "Candidate lilo alignments (top 3 by composite score):\n"
        f"{cand_lines}\n\n"
        "Task: Choose the single best LILO for this question based on:\n"
        "  1. Topic / concept similarity\n"
        "  2. Cognitive level (Bloom Taxonomy)\n"
        "  3. Learning objective purpose\n\n"
        "Respond with ONLY the lilo id (e.g. LILO-2). Nothing else."
    )

    models_to_try = [
        ('gemini-2.5-flash',      'v1beta'),
        ('gemini-2.5-flash-lite', 'v1beta'),
        ('gemini-1.5-flash',      'v1beta'),
    ]

    try:
        import requests
        _n_keys = max(key_count(), 1)
        for _ki in range(_n_keys):
            for model_name, api_version in models_to_try:
                url = (
                    f"https://generativelanguage.googleapis.com/{api_version}"
                    f"/models/{model_name}:generateContent?key={api_key}"
                )
                payload = {
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": 0.1, "maxOutputTokens": 32},
                }
                try:
                    resp = requests.post(url, json=payload, timeout=30)
                    _LLM_LAST_CALL_TIME = time.time()
                    if resp.status_code == 200:
                        result = resp.json()
                        text = (
                            result.get("candidates", [{}])[0]
                            .get("content", {})
                            .get("parts", [{}])[0]
                            .get("text", "")
                            .strip()
                        )
                        m = re.search(r'LILO-\d+', text, re.IGNORECASE)
                        if m:
                            logger.info(f"LLM chose: {m.group(0)} for: {question_text[:60]}")
                            return m.group(0).upper()
                        return None
                    elif resp.status_code in (429, 403):
                        logger.warning(f"Tiebreaker: key#{_ki+1} {model_name} → {resp.status_code}, rotating key")
                        break  # try next key
                except Exception as e:
                    logger.warning(f"LLM model {model_name} failed: {e}")
                    continue
            api_key = get_next_key(failed_key=api_key)
    except ImportError:
        logger.warning("requests not installed -- cannot call Gemini for LLM validation")

    return None


# ---------------------------------------------------------------------------
# lilo Enrichment — single batch LLM call per syllabus (Gemini → Grok fallback)
# ---------------------------------------------------------------------------

def _hash_lilos(lilos: List[Dict]) -> str:
    """MD5 of all LILO texts — used as a cache key."""
    combined = '|'.join(l['text'] for l in lilos)
    return hashlib.md5(combined.encode('utf-8')).hexdigest()


def _call_gemini_enrich(prompt: str) -> Optional[list]:
    """Single Gemini REST call for batch LILO enrichment. Returns a JSON list or None."""
    api_key = get_next_key()
    if not api_key:
        logger.warning("[GeminiKeyPool] No API keys configured — skipping Gemini enrichment")
        return None
    models_to_try = [
        ('gemini-2.5-flash',      'v1beta'),
        ('gemini-2.5-flash-lite', 'v1beta'),
        ('gemini-1.5-flash',      'v1beta'),
    ]
    try:
        import requests
        _n_keys = max(key_count(), 1)
        for _ki in range(_n_keys):
            for model_name, api_version in models_to_try:
                url = (
                    f"https://generativelanguage.googleapis.com/{api_version}"
                    f"/models/{model_name}:generateContent?key={api_key}"
                )
                payload = {
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": 0.1, "maxOutputTokens": 8192},
                }
                try:
                    resp = requests.post(url, json=payload, timeout=60)
                    if resp.status_code == 200:
                        _result = resp.json()
                        # Record EXACT tokens + cost for the enrichment call so the
                        # per-run total includes it. Best-effort, never fatal.
                        try:
                            from app.utils.token_tracker import TokenTracker
                            _pt, _ct, _tot = TokenTracker.extract_gemini_usage(_result)
                            if _tot:
                                _pr = TokenTracker._price_for(model_name)
                                _cost = (_pt / 1_000_000) * _pr['in'] + (_ct / 1_000_000) * _pr['out']
                                logger.info(f"Enrich [{model_name}] in={_pt} out={_ct} "
                                            f"total={_tot} tok → ${_cost:.6f}")
                                TokenTracker.log_api_call(
                                    model=model_name, prompt_tokens=_pt,
                                    completion_tokens=_ct, question_id=0,
                                    question_quality_before=0, question_quality_after=100)
                        except Exception:
                            pass
                        raw = (
                            _result
                            .get("candidates", [{}])[0]
                            .get("content", {})
                            .get("parts", [{}])[0]
                            .get("text", "")
                            .strip()
                        )
                        m = re.search(r'\[.*\]', raw, re.DOTALL)
                        if m:
                            return json.loads(m.group(0))
                        return None
                    elif resp.status_code in (429, 403):
                        logger.warning(f"Enrich: key#{_ki+1} {model_name} → {resp.status_code}, rotating key")
                        break  # try next key
                    else:
                        logger.warning(f"Gemini {model_name} returned HTTP {resp.status_code}")
                except Exception as e:
                    logger.warning(f"Gemini enrich error ({model_name}): {e}")
                    continue
            api_key = get_next_key(failed_key=api_key)
    except ImportError:
        logger.warning("requests not installed — cannot call Gemini for LILO enrichment")
    return None


def _call_grok_enrich(prompt: str) -> Optional[list]:
    """
    Groq Cloud fallback for batch LILO enrichment (fast LLaMA/Mixtral inference).
    Uses the GROQ_API_KEY from .env via OpenAI-compatible endpoint.
    Returns a JSON list or None.
    """
    api_key = _get_settings().GROQ_API_KEY
    if not api_key:
        logger.warning("GROQ_API_KEY not set in .env — skipping Groq enrichment fallback")
        return None
    try:
        import requests
        # Groq Cloud is OpenAI-compatible; best free models for structured output
        models_to_try = ["llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768"]
        for model in models_to_try:
            try:
                resp = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.1,
                        "max_tokens": 4096,
                    },
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    timeout=60,
                )
                if resp.status_code == 200:
                    raw = (
                        resp.json()
                        .get("choices", [{}])[0]
                        .get("message", {})
                        .get("content", "")
                        .strip()
                    )
                    m = re.search(r'\[.*\]', raw, re.DOTALL)
                    if m:
                        logger.info(f"Groq enrichment succeeded with model: {model}")
                        return json.loads(m.group(0))
                elif resp.status_code == 429:
                    logger.warning(f"Groq rate limit on {model} — trying next model")
                    continue
                else:
                    logger.warning(f"Groq {model} returned HTTP {resp.status_code}")
            except Exception as e:
                logger.warning(f"Groq enrich error ({model}): {e}")
                continue
    except ImportError:
        logger.warning("requests not installed — cannot call Groq for LILO enrichment")
    return None


def enrich_lilos_with_llm(lilos: List[Dict]) -> List[Dict]:
    """
    LLM-Augmented LILO Enrichment — called ONCE per syllabus.

    Sends ALL lilos in a single batch prompt to Gemini (Grok as fallback).
    Each LILO is enriched with:
      - bloom_level:    LLM-verified Bloom level (overrides regex-guessed value)
      - key_concepts:   5-8 domain-specific terms implied by this LILO
      - enriched_text:  Expanded 1-2 sentence description covering all subtopics

    Results are in-memory cached by LILO text hash — re-uploading the same
    syllabus never triggers a second API call within the same server session.
    Falls back to the original un-enriched lilos if both APIs fail or are
    not configured.
    """
    if not lilos:
        return lilos

    cache_key = _hash_lilos(lilos)
    if cache_key in _ENRICHMENT_CACHE:
        logger.info("LILO enrichment: cache hit — skipping API call")
        cached = _ENRICHMENT_CACHE[cache_key]
        # Re-merge: start from current lilos (which carry correct metadata such
        # as topic_title and learning_content from the latest parser) and
        # overlay only the LLM-derived fields from the cache.  This prevents
        # stale cache entries (populated before parser improvements) from
        # silently returning dicts with missing/null metadata fields.
        result = [dict(l) for l in lilos]
        for i, cached_item in enumerate(cached):
            if i < len(result):
                for llm_field in ('bloom_level', 'key_concepts', 'enriched_text'):
                    val = cached_item.get(llm_field)
                    if val:
                        result[i][llm_field] = val
        return result

    BATCH_SIZE = 10
    enriched = [dict(l) for l in lilos]
    merged = 0
    total_batches = (len(lilos) + BATCH_SIZE - 1) // BATCH_SIZE
    logger.info(f"LILO enrichment: calling LLM for {len(lilos)} lilos in {total_batches} batch(es) of {BATCH_SIZE}...")

    for batch_start in range(0, len(lilos), BATCH_SIZE):
        batch = lilos[batch_start:batch_start + BATCH_SIZE]
        batch_end = batch_start + len(batch) - 1
        lilo_list_str = '\n'.join(
            f'{i}. "{l["text"]}"' for i, l in enumerate(batch)
        )
        batch_prompt = (
            "You are an expert curriculum designer. The following are Lesson Intended Learning Outcomes (LILOs) "
            "from a university course syllabus. Each LILO may belong to any academic discipline — "
            "engineering, medicine, business, computer science, arts, or any other field.\n\n"
            "For each LILO, perform a deep curriculum analysis regardless of subject matter:\n\n"
            f"LILOs (indices 0 through {len(batch) - 1}):\n{lilo_list_str}\n\n"
            f"Return a JSON array of EXACTLY {len(batch)} objects (indices 0 through {len(batch) - 1}), "
            "one per LILO in the same order. Each object has:\n"
            '  "index":         0-based integer matching the input list (0 to ' + str(len(batch) - 1) + ')\n'
            '  "bloom_level":   exactly one of [remember, understand, apply, analyze, evaluate, create]\n'
            '                   — verify carefully by the action verb, not just keywords.\n'
            '                   For framing phrases "develop/gain/build the ability/proficiency/\n'
            '                   skill/understanding to/in/of X", classify by the action (or the\n'
            '                   nature of the noun) in X, NOT by "develop". e.g. "Develop the\n'
            '                   ability to identify and extract frequent patterns" -> analyze;\n'
            '                   "Develop understanding of ethical issues" -> understand;\n'
            '                   "Develop a recommender system" -> create.\n'
            '  "key_concepts":  array of 5-8 specific domain terms, techniques, or topics this LILO implies\n'
            '                   — include both the explicit topic AND related subtopics a student must know\n'
            '  "enriched_text": a 1–2 sentence expansion of the LILO that names all implied subtopics,\n'
            '                   tools, frameworks, or skills a question testing this LILO might reference\n\n'
            "Examples of enriched_text depth:\n"
            '  Original: "Apply CRISP-DM to structure data mining projects"\n'
            '  Enriched: "Apply the CRISP-DM process model to plan and execute data mining projects, '
            'covering business understanding, data understanding, data preparation, modeling, '
            'evaluation, and deployment phases including real-world formulation of data problems."\n\n'
            '  Original: "Explain membrane transport mechanisms"\n'
            '  Enriched: "Explain passive and active transport across cell membranes including '
            'osmosis, diffusion, facilitated diffusion, sodium-potassium pump, endocytosis, '
            'and exocytosis with reference to concentration gradients and membrane proteins."\n\n'
            "Return ONLY a valid JSON array. No markdown fences, no extra text."
        )

        raw_result = _call_gemini_enrich(batch_prompt) or _call_grok_enrich(batch_prompt)
        if not raw_result:
            logger.warning(f"LILO enrichment: LLM unavailable for batch {batch_start}-{batch_end} — skipping")
            continue

        try:
            for item in raw_result:
                local_idx = item.get("index", -1)
                if not isinstance(local_idx, int) or not (0 <= local_idx < len(batch)):
                    continue
                global_idx = batch_start + local_idx
                llm_bloom = str(item.get("bloom_level", "")).lower()
                if llm_bloom in ("remember", "understand", "apply", "analyze", "evaluate", "create"):
                    enriched[global_idx]["bloom_level"] = llm_bloom
                kc = item.get("key_concepts")
                if isinstance(kc, list) and kc:
                    enriched[global_idx]["key_concepts"] = [str(k).lower().strip() for k in kc if k]
                et = item.get("enriched_text", "")
                if isinstance(et, str) and len(et.split()) >= 5:
                    enriched[global_idx]["enriched_text"] = et
                merged += 1
        except Exception as e:
            logger.warning(f"LILO enrichment merge error for batch {batch_start}-{batch_end}: {e}")
            continue

    _ENRICHMENT_CACHE[cache_key] = enriched
    logger.info(f"LILO enrichment complete: {merged}/{len(lilos)} lilos enriched and cached")
    return enriched


# ---------------------------------------------------------------------------
# CO advisory cognitive demand — SEMANTIC analysis (same approach as LILO
# enrichment: judge the complete statement, not the highest apparent verb).
# The result is ADVISORY guidance for the Activity/Quiz creation UI — it is
# never used in scoring or attainment computation.
# ---------------------------------------------------------------------------

_CO_BLOOM_CACHE: Dict[str, Dict] = {}

_BLOOM_SET = ("remember", "understand", "apply", "analyze", "evaluate", "create")


def analyze_co_bloom_llm(cos: List[Dict]) -> Optional[Dict[str, Dict]]:
    """Semantic Bloom analysis of Course Outcome statements.

    Input:  [{"id": "CO-1", "description": "..."}]
    Output: {"CO-1": {"level", "detected_actions", "confidence",
                      "explanation"}} or None when the LLM is unavailable.

    A CO like "Gain a deep understanding ... and formulate real-world problems
    as data mining genres" must NOT be classified CREATE merely because of the
    verb "formulate" — the model judges what the required performance actually
    is (structuring/classifying problems -> analyze). Hash-cached per CO set.
    """
    cos = [c for c in (cos or []) if c.get("id") and (c.get("description") or "").strip()]
    if not cos:
        return None
    cache_key = hashlib.md5(
        "|".join(f"{c['id']}:{c['description']}" for c in cos).encode("utf-8")
    ).hexdigest()
    if cache_key in _CO_BLOOM_CACHE:
        return _CO_BLOOM_CACHE[cache_key]

    co_list_str = "\n".join(f'{i}. [{c["id"]}] "{c["description"]}"' for i, c in enumerate(cos))
    prompt = (
        "You are an expert OBE curriculum designer. The following are Course Outcomes (COs) "
        "from a university syllabus. For EACH CO, determine its PRIMARY cognitive demand on "
        "Bloom's revised taxonomy by analyzing the COMPLETE statement semantically — the "
        "framing language, the operative actions, the required student output, and the "
        "context — never just the highest-sounding verb.\n\n"
        "Rules:\n"
        "- Framing phrases ('gain a deep understanding of', 'develop the ability to', "
        "'demonstrate proficiency in') are classified by the ACTION they frame, not the "
        "framing verb itself.\n"
        "- 'Formulate problems as categories/genres' is structuring/classifying -> analyze, "
        "NOT create. 'Formulate a novel model/solution' -> create.\n"
        "- A CO with several distinct required performances ('explain X and design Y') gets "
        "the level of its DOMINANT graded performance as 'level', with every detected action "
        "listed; confidence should be 'medium' or 'low' when the dominant one is debatable.\n"
        "- confidence: 'high' when one clear performance dominates; 'medium' when multiple "
        "actions but a clear dominant; 'low' when genuinely ambiguous.\n\n"
        f"COs (indices 0 through {len(cos) - 1}):\n{co_list_str}\n\n"
        f"Return a JSON array of EXACTLY {len(cos)} objects, one per CO in the same order:\n"
        '  "index":            0-based integer matching the input list\n'
        '  "level":            exactly one of [remember, understand, apply, analyze, evaluate, create]\n'
        '  "detected_actions": array of the distinct cognitive actions found, each as\n'
        '                      "verb (level)" e.g. ["formulate problems (analyze)", "understand concepts (understand)"]\n'
        '  "confidence":       exactly one of [high, medium, low]\n'
        '  "explanation":      ONE short sentence saying why this level was chosen\n\n'
        "Return ONLY a valid JSON array. No markdown fences, no extra text."
    )
    raw = _call_gemini_enrich(prompt) or _call_grok_enrich(prompt)
    if not raw:
        return None
    out: Dict[str, Dict] = {}
    try:
        for item in raw:
            idx = item.get("index", -1)
            if not isinstance(idx, int) or not (0 <= idx < len(cos)):
                continue
            level = str(item.get("level", "")).lower()
            if level not in _BLOOM_SET:
                continue
            conf = str(item.get("confidence", "")).lower()
            actions = item.get("detected_actions")
            out[cos[idx]["id"]] = {
                "level": level,
                "detected_actions": [str(a) for a in actions if a][:8] if isinstance(actions, list) else [],
                "confidence": conf if conf in ("high", "medium", "low") else "medium",
                "explanation": str(item.get("explanation", ""))[:400],
            }
    except Exception as e:
        logger.warning(f"CO bloom analysis merge error: {e}")
        return None
    if out:
        _CO_BLOOM_CACHE[cache_key] = out
        logger.info(f"CO bloom analysis: {len(out)}/{len(cos)} COs classified")
    return out or None


# ---------------------------------------------------------------------------
# lilo parsing
# ---------------------------------------------------------------------------

def parse_lilos(lilo_text: str, return_meta: bool = False):
    """
    Parse any Philippine CHED-format syllabus (or plain learning outcome list)
    into a list of labelled outcome dicts.

    Handles all common Philippine HEI naming conventions:
      ILO  - Intended Learning Outcomes   (most ISU/CHED syllabi)
      lilo  - Lesson Learning Outcomes
      CLO  - Course Learning Outcomes
      TLO  - Terminal Learning Outcomes
      ELO  - Enabling Learning Outcomes
      PLO  - Program Learning Outcomes
      CO   - Course Outcomes
      LO   - Learning Outcomes
      OBJ  - Objectives

    Detection order (first strategy that yields outcomes wins):
      Priority A — CHED "able to" blocks:
          "At the end of [period], the students should be able to:"
          Lines after the trigger that START WITH a Bloom's action verb are
          captured as ILOs. Domain sub-headers (Cognitive/Affective/
          Psychomotor) are silently skipped.
      Priority B — Section headers ("Course Outcomes:", "Intended Learning
          Outcomes", "Learning Objectives", etc.) followed by Bloom-verb lines.
          (Handled automatically — _ABLE_TO_SHORT also fires on these blocks.)
      Priority C — Coded prefixes (ILO 1:, CLO 2.3 |, CO 1 -, etc.).
      Priority D — Numbered/bulleted lines that begin with a Bloom's verb.
      Priority E — Any line in the document starting with a Bloom's verb.
      Priority F — Sentence-split fallback.

    Returns:
        [{"id": "LILO-1", "text": "...", "bloom_level": "understand",
          "source_type": "ilo"|"co"}, ...]

        ``source_type`` is:
          - ``"co"``  — Course Outcome (broad, course-level objective)
          - ``"ilo"`` — Intended/Lesson Learning Outcome (specific, weekly/lesson-level)

    With ``return_meta=True`` returns ``(outcomes, meta)`` where meta carries
    ``priority`` — the letter of the branch that produced the list. Callers use
    it to judge how much to trust the result: "E" means the document had no
    usable line structure and every pattern above it missed, which is a far
    weaker parse than the same count from "A" or "B".

    NOTE: every priority except A/B is line-oriented and anchors at the start
    of a line, so whatever produces ``lilo_text`` must preserve the document's
    line breaks. Text extracted one line per PAGE degrades straight to "E".
    """
    if not lilo_text or not lilo_text.strip():
        return ([], {"priority": None, "count": 0}) if return_meta else []

    lines = [l.strip() for l in lilo_text.splitlines() if l.strip()]
    outcomes: List[Dict] = []
    idx = 1

    # ── Compiled patterns ──────────────────────────────────────────────────

    # Trigger: "At/By the end of X, students should be able to:"
    _ABLE_TO = re.compile(
        r'(?:at\s+the\s+end\s+of|by\s+the\s+end\s+of|upon\s+completion\s+of'
        r'|after\s+completing|at\s+the\s+close\s+of)'
        r'.{0,100}'
        r'(?:should|will|must|are\s+expected\s+to)\s+be\s+able\s+to',
        re.IGNORECASE,
    )
    # Shorter form: "students/learners should be able to:"
    _ABLE_TO_SHORT = re.compile(
        r'(?:students?|learners?)\s+(?:should|will|must|are\s+expected\s+to)\s+be\s+able\s+to',
        re.IGNORECASE,
    )
    # Section headings that introduce a block of Course Outcomes (CO-level)
    # Matches variants like "COURSE LEARNING OUTCOMES (CLOs):"
    _CO_HEADER = re.compile(
        r'^(?:'
        r'course\s+(?:learning\s+)?outcomes?'
        r'|course\s+objectives?'
        # A bare code is only a heading when it is plural or punctuated. An
        # alignment matrix puts a lone "CLO" on its own line as a column label,
        # and reading that as a heading reopened the course-outcome block right
        # after the section title had closed it — the weekly lesson outcomes
        # under the matrix then came back as course outcomes.
        r'|(?:clo|co)s\s*:?'
        r'|(?:clo|co)\s*:'
        r')(?:\s*\([^)]*\))?\s*:?$',
        re.IGNORECASE,
    )
    # Section headings that introduce a block of ILOs/LILOs (lesson-level)
    # Matches variants like "LESSON INTENDED LEARNING OUTCOMES (LILOs)"
    # Lesson-level headers. NOTE: "Program Learning Outcomes"/"PLO" are NOT here —
    # they are program-level (like POs) and must never be captured as lessons.
    _OUTCOME_HEADER = re.compile(
        r'^(?:'
        r'(?:lesson\s+)?intended\s+learning\s+outcomes?'
        r'|lesson\s+(?:learning\s+)?outcomes?'
        r'|desired\s+learning\s+outcomes?'
        r'|learning\s+outcomes?'
        r'|terminal\s+learning\s+outcomes?'
        r'|enabling\s+learning\s+outcomes?'
        r'|lesson\s+objectives?'
        r'|learning\s+objectives?'
        # Same rule as _CO_HEADER: a lone "LO" or "ILO" is a table column label
        # far more often than it is a section heading.
        r'|(?:ilo|lilo|tlo|elo|dlo|lo)s\s*:?'
        r'|(?:ilo|lilo|tlo|elo|dlo|lo)\s*:'
        r')(?:\s*\([^)]*\))?\s*:?$',
        re.IGNORECASE,
    )
    # Program-level headers — end any lesson/course block so their lines are not
    # captured as learning outcomes (POs are extracted by a separate parser).
    _PO_HEADER = re.compile(
        r'^(?:program(?:me)?\s+(?:learning\s+)?outcomes?|plos?|graduate\s+attributes?|gas?)'
        r'(?:\s*\([^)]*\))?\s*:?$',
        re.IGNORECASE,
    )
    # Coded prefix: ILO 1:, CLO 2.3 |, CO 1 -, OBJ 3.  Group 1 = the code
    # (used to tell course-level CLO/CO from lesson-level ILO/LILO), group 2 = text.
    _CODED = re.compile(
        r'^(ILO|LILO|CLO|TLO|ELO|PLO|CO|LO|OBJ)\s*[\d]+(?:\.[\d]+)?\s*[|:\-\.\)]\s*(.+)',
        re.IGNORECASE,
    )
    # Domain sub-headers inside a CHED "able to" block — skip but stay in block
    _DOMAIN_SKIP = re.compile(
        r'^(?:cognitive|affective|psychomotor|knowledge|skills?|attitude|values?|domain)$',
        re.IGNORECASE,
    )
    # An ALL-CAPS section title: "ALIGNMENT MATRIX 2. CLO - GA - IO MAPPING",
    # "TARGET SUSTAINABLE DEVELOPMENT GOALS (SDGS)", "COURSE PLAN". Nothing used
    # to close an outcome block except the next header, so the Course Outcomes
    # block of an ISU syllabus ran on through the weekly LILO matrix, the SDG
    # paragraphs and the assessment table: 4 course outcomes came back as 15.
    _SECTION_BREAK = re.compile(r'^(?=.{4,90}$)[^a-z]+$')

    def _is_section_break(ln: str) -> bool:
        """A heading that ends the block above it — never a line INSIDE a table.

        Deliberately narrow. The code and level lines that sit between outcomes
        in an alignment matrix are also upper-case — `CLO1 - CLO2`, `D E E D E`,
        `PO1 PO2 PO3 PO4` — and closing on one of those would throw away every
        outcome after it. So a break needs two words, one of them four or more
        letters, and must not read as an outcome itself.
        """
        if not _SECTION_BREAK.match(ln):
            return False
        words = ln.split()
        if len(words) < 2:
            return False
        if not any(len(re.sub(r'[^A-Z]', '', w)) >= 4 for w in words):
            return False
        return not _starts_with_bloom_verb(ln)

    # Strip leading numbering / bullets from simple lists
    _BULLET_STRIP = re.compile(
        r'^(?:[a-z]\)|\d+[\:\.\)\-]\s*|[*\-•–]\s*)',
        re.IGNORECASE,
    )
    # Standalone PO code line inside COURSE PLAN tables: `PO1, PO2, PO6`
    _PO_CODE_LINE = re.compile(
        r'^(?:PO\s*\d+\s*(?:,\s*PO\s*\d+\s*)*)$',
        re.IGNORECASE,
    )
    # Standalone ISU Graduate Attribute code line: `GA1`, `IFK`, `Cm`, `IFK, Cp`, etc.
    # Also matches revised format codes GA1-GA10 used in newer ISU syllabi.
    _GA_CODE_LINE = re.compile(
        r'^(?:GA\d+|IFK?|CEL|Cm|Cp|LL)(?:\s*[,/]\s*(?:GA\d+|IFK?|CEL|Cm|Cp|LL))*$',
        re.IGNORECASE,
    )
    # Lines we know are NOT ILOs regardless of starting verb
    _NOT_ILO = re.compile(
        r'^(?:machine\s+problem|lecture|face\s+to|quiz|exam(?:ination)?'
        r'|use\s+of|rubric|recitation|brainstorming|peer\s+teach'
        r'|hands.on|asynchronous|synchronous|blended|online|face-to-face'
        r'|reference|textbook|module\s*\d|week\s*\d|unit\s*\d|chapter\s*\d'
        r'|total|hours?|laboratory|po\s*\d|cm\b|cp\b|cel\b|ifk\b|ll\b'
        r'|prerequisite|vision|mission|quality\s+policy|goals?\s+of'
        r'|graduate\s+attribute|program\s+outcome|institutional\s+outcome'
        r'|republic\s+of|state\s+university|college\s+of'
        # Grade / formula lines — must never become ILOs
        # Matches: "Grade = FA + ME + FE", "[TOTAL (Lec)] * 0.6 + ...",
        # "= FA + ME", "%  mastery", "75% mastery or 3/4 Proficient",
        # ">=75% of students", "Formative Assessment", "Summative"
        r'|grade\s*=|=\s*fa\s*\+|\[total\b'
        r'|>=?\s*\d+%|\d+%\s+mastery|\d+/\d+\s+proficient'
        r'|formative\s+assessment|summative\s+assessment|mastery\s+threshold'
        # PDCA / OBE implementation instruction lines (e.g., "PLAN - Align CLOs...",
        # "DO - Implement the course plan...", "CHECK / ASSESS - Compute...",
        # "ACT - Use the attainment summary...").  These start with an OBE phase
        # label (all-caps verb + dash) and must never become student ILOs.
        r'|(?:plan|do|act)\s*[-–]|check\s*/\s*assess'
        r'|minimum\s+attainment|acceptable\s+ai|ai\s+assistance'
        r'|minor\s+violations?|moderate\s+violations?|severe\s+violations?'
        r')',
        re.IGNORECASE,
    )
    _POST_BLOCK_SKIP = re.compile(
        r'^(?:'
        r'machine\s+problem|lecture(?:\s+and\s+discussion)?|face\s*to\s*face'
        r'|quizzes?|hands[\s\-]?on(?:\s+exercises?)?(?:\(laboratory\))?'
        r'|use\s+of\s+numerical\s+scores|rubrics?(?:\s+for\s+hands[\s\-]?on\s+exercises?)?'
        r'|brainstorming|reflective\s+discussion|peer\s+teaching|demonstration'
        r'|program\s+coding|watching\s+videos|solving\s+problem\s+sets|recitation'
        r'|prelim\s+examination.*|midterm\s+examination.*|final\s+examination.*'
        r'|face\s+to\s+Face|online|asynchronous|synchronous|blended'
        # PDCA / OBE implementation instructions (also filtered here so they
        # never become topic_title in _finalize_block)
        r'|(?:plan|do|act)\s*[-–]|check\s*/\s*assess'
        r'|minimum\s+attainment|acceptable\s+ai|minor\s+violations?'
        r'|moderate\s+violations?|severe\s+violations?'
        r'|alignment\s+matrix|clo\s*[-–]\s*ga|clo\s*[-–]\s*io'
        r')$',
        re.IGNORECASE,
    )

    # ── Helper: clean and add an outcome ──────────────────────────────────
    def _add(text: str, source_type: str = "ilo", **extra) -> Optional[Dict]:
        nonlocal idx
        # Strip trailing table-cell artifacts: "Apply  PO1, PO2" or "(Apply)"
        text = re.sub(
            r'\s+(?:Apply|Create|Analyze|Evaluate|Remember|Understand)'
            r'(?:\s*/\s*\w+)?\s+(?:PO|ILO|CLO|CO)[\d,\s]*$',
            '', text, flags=re.IGNORECASE,
        ).strip()
        # Reject grade/formula lines that accidentally pass Bloom-verb checks
        # e.g. "Grade = FA + ME + FE" or "[TOTAL (Lec)] * 0.6 + [TOTAL (Lab)] * 0.4"
        if re.search(r'grade\s*=|=\s*fa\s*\+|\[total\b|>=?\s*\d+%|\d+%\s+mastery', text, re.IGNORECASE):
            return None
        if len(text.split()) >= 4:
            entry = {
                "id": f"LILO-{idx}",
                "text": text,
                "bloom_level": classify_bloom(text, is_outcome=True),
                "source_type": source_type,
                **extra,
            }
            outcomes.append(entry)
            idx += 1
            return entry
        return None

    # ── Patterns used inside _finalize_block ──────────────────────────────
    _WEEK_RANGE_LINE = re.compile(
        r'^(?:Preparatory|Pre(?:liminary)?|Week\s*\d+|\d{1,2}(?:[–\-]\d{1,2})?)$',
        re.IGNORECASE,
    )

    def _finalize_block(block: Optional[Dict]) -> None:
        if not block or not block.get("outcomes"):
            return

        raw_post_lines = block.get("post_lines") or []
        content_lines: List[str] = []
        for raw_line in raw_post_lines:
            line = raw_line.strip()
            if not line:
                continue
            if _PO_CODE_LINE.match(line):
                continue
            if _GA_CODE_LINE.match(line):
                continue
            if _POST_BLOCK_SKIP.match(line):
                continue
            if re.fullmatch(r'\d+(?:\.\d+)?', line):
                continue
            if re.fullmatch(r'[A-Z][A-Z\d]*(?:\s*,\s*[A-Z][A-Z\d]*)*', line):
                continue
            content_lines.append(line)

        block_outcomes = block.get("outcomes", [])

        # ── Week-range distribution for multi-week Course Plan blocks ────────
        # If the post_lines contain week-range labels (e.g. "Preparatory",
        # "2-3", "4-5") and the count of those labels roughly equals the number
        # of LILOs in the block, distribute each LILO to its own week slot so
        # that build_content_blocks_from_lilos creates *separate* content blocks.
        week_entry_indices = [
            (i, line) for i, line in enumerate(content_lines)
            if _WEEK_RANGE_LINE.match(line)
        ]

        if week_entry_indices and len(block_outcomes) > 1:
            for outcome_idx, outcome in enumerate(block_outcomes):
                if outcome_idx < len(week_entry_indices):
                    week_pos, week_label = week_entry_indices[outcome_idx]
                    outcome["week"] = week_label
                    # Look for a topic title: first short non-week-range line
                    # that appears AFTER this week entry and BEFORE the next one
                    next_week_pos = (
                        week_entry_indices[outcome_idx + 1][0]
                        if outcome_idx + 1 < len(week_entry_indices)
                        else len(content_lines)
                    )
                    topic_candidates = [
                        line for line in content_lines[week_pos + 1 : next_week_pos]
                        if not _WEEK_RANGE_LINE.match(line)
                        and 2 <= len(line.split()) <= 7
                    ]
                    if topic_candidates:
                        outcome["topic_title"] = topic_candidates[0]
                else:
                    # More outcomes than week labels → reuse the last week
                    outcome["week"] = week_entry_indices[-1][1]

                if block.get("po_codes"):
                    outcome["po_codes"] = list(block["po_codes"])
                if block.get("ga_codes"):
                    outcome["ga_codes"] = list(block["ga_codes"])

            return  # Per-week assignment done; skip uniform assignment below

        # ── Uniform assignment (original behaviour, no week-range detected) ──
        topic_title = content_lines[0] if content_lines else None
        learning_lines = content_lines[1:] if len(content_lines) > 1 else []
        learning_content = " | ".join(learning_lines) if learning_lines else None

        for outcome in block_outcomes:
            if block.get("po_codes"):
                outcome["po_codes"] = list(block["po_codes"])
            if block.get("ga_codes"):
                outcome["ga_codes"] = list(block["ga_codes"])
            if block.get("week"):
                outcome["week"] = block["week"]
            if topic_title:
                outcome["topic_title"] = topic_title
            if learning_content:
                outcome["learning_content"] = learning_content

    # ── Outcome-code family detection + ILO course/lesson heuristic ───────
    # Distinguish course-level (CLO/CO) from lesson-level (ILO/LILO/TLO/ELO/LO)
    # by the code itself. "ILO" is genuinely ambiguous — course-level in many
    # CHED syllabi, lesson-level in ISU — so treat it as course-level ONLY when
    # it is the sole family present (no CLO/CO and no LILO to anchor the split).
    _codes_present = set()
    for _ln in lines:
        _m = _CODED.match(_ln)
        if _m:
            _codes_present.add(_m.group(1).upper().replace(' ', ''))
    _ilo_is_course = ('ILO' in _codes_present) and not (_codes_present & {'CLO', 'CO', 'LILO'})

    def _code_st(code: str) -> str:
        """Map an outcome code to a source type: 'co', 'ilo', or 'po' (skip)."""
        c = (code or '').upper().replace(' ', '')
        if c in ('CLO', 'CO'):
            return 'co'
        if c in ('PLO', 'PO', 'GA'):
            return 'po'
        if c == 'ILO':
            return 'co' if _ilo_is_course else 'ilo'
        return 'ilo'  # LILO, TLO, ELO, LO, OBJ

    # ── Priority A: CHED "able to" block detection ────────────────────────
    in_block = False
    current_section_type = "ilo"  # default; flips to "co" under CO headers
    pending_po_codes: List[str] = []
    pending_ga_codes: List[str] = []
    active_block: Optional[Dict] = None

    def _start_block(source_type: str, trigger_line: str) -> None:
        nonlocal in_block, current_section_type, active_block, pending_po_codes, pending_ga_codes
        _finalize_block(active_block)
        in_block = True
        current_section_type = source_type
        active_block = {
            "source_type": source_type,
            "trigger_line": trigger_line,
            "week": re.sub(r'\s+', ' ', trigger_line).strip() if source_type == "ilo" else None,
            "po_codes": list(pending_po_codes),
            "ga_codes": list(pending_ga_codes),
            "outcomes": [],
            "post_lines": [],
        }
        pending_po_codes = []
        pending_ga_codes = []

    # A header wraps across two lines whenever the PDF column is narrow enough:
    # "Lesson Intended Learning" / "Outcomes (LILOs)" is one heading, and neither
    # half matches on its own. Missing it is what let the course-outcome block
    # swallow the whole alignment matrix that followed it.
    _skip_next = False

    for _li, line in enumerate(lines):
        if _skip_next:
            _skip_next = False
            continue
        if _GA_CODE_LINE.match(line):
            # GA code signals the next block's graduate attribute — capture it
            # and prevent it from contaminating the current block's post_lines.
            pending_ga_codes = _extract_ga_codes(line)
            continue

        if _PO_CODE_LINE.match(line):
            if active_block and active_block.get("outcomes"):
                _finalize_block(active_block)
                active_block = None
                in_block = False
            pending_po_codes = _extract_po_codes(line)
            continue

        # Program-level header (Program Learning Outcomes / PLO / GA): end the
        # current block so its lines are not captured as learning outcomes.
        if _PO_HEADER.match(line):
            _finalize_block(active_block)
            active_block = None
            in_block = False
            continue

        is_trigger = bool(_ABLE_TO.search(line)) or bool(_ABLE_TO_SHORT.search(line))
        is_co_hdr  = bool(_CO_HEADER.match(line))
        is_ilo_hdr = bool(_OUTCOME_HEADER.match(line))

        # Nothing matched — try this line joined with the next one, in case the
        # heading was wrapped. Both regexes are anchored at both ends, so a join
        # only matches when the two halves really are one heading.
        if not (is_trigger or is_co_hdr or is_ilo_hdr) and _li + 1 < len(lines):
            joined = f"{line} {lines[_li + 1]}".strip()
            if _CO_HEADER.match(joined):
                is_co_hdr, line, _skip_next = True, joined, True
            elif _OUTCOME_HEADER.match(joined):
                is_ilo_hdr, line, _skip_next = True, joined, True

        if is_trigger:
            # Detect course-level vs lesson-level "able to" block:
            # "at the end of the COURSE / PROGRAM / SEMESTER" → CO
            # "at the end of WEEK X / LESSON X / TOPIC X"     → ILO
            _line_l = line.lower()
            if (re.search(r'\b(course|program(?:me)?|semester|term|subject)\b', _line_l)
                    and not re.search(r'\b(week|lesson|topic|unit|chapter|module)\b', _line_l)):
                _start_block("co", line)
            else:
                _start_block("ilo", line)
            continue

        if is_co_hdr:
            _start_block("co", line)
            continue

        if is_ilo_hdr:
            _start_block("ilo", line)
            continue

        # A new section title ends the block above it.
        if in_block and _is_section_break(line):
            _finalize_block(active_block)
            active_block = None
            in_block = False
            continue

        if in_block:
            if _DOMAIN_SKIP.match(line):
                continue  # skip "Cognitive" / "Affective" but stay in block
            # Coded outcome line under a header (e.g. "CLO 1: ...", "ILO 2: ...").
            # These don't start with a verb, so the bloom check below would DROP
            # them. Classify by the code itself (CLO→course, ILO/LILO→lesson).
            _mc = _CODED.match(line)
            if _mc:
                _st = _code_st(_mc.group(1))
                if _st == 'po':
                    continue  # program-level, not a learning outcome for this list
                _txt = _mc.group(2).strip()
                entry = _add(_txt, source_type=_st)
                if entry and active_block is not None:
                    active_block["outcomes"].append(entry)
                continue
            if _NOT_ILO.match(line):
                if active_block and active_block.get("outcomes"):
                    active_block["post_lines"].append(line)
                continue
            # A COURSE PLAN table lists its lesson outcomes as a numbered list
            # inside the cell — "1. Explain the course outcomes, ...". The
            # bloom check reads "1." as the first word, so the real outcomes
            # were dropped and only the table's prose rows survived.
            _cand = _BULLET_STRIP.sub('', line).strip()
            # `_cand[:1].isupper()` rejects a wrapped continuation. "This course
            # aims to develop students' ability to explain ... / implement and
            # test linear, hierarchical, ... structures" is one sentence broken
            # across two lines, and the second half was being kept as an outcome
            # of its own purely because "implement" opens it.
            if (_starts_with_bloom_verb(_cand) and 4 <= len(_cand.split()) <= 60
                    and _cand[:1].isupper()):
                entry = _add(_cand, source_type=current_section_type)
                if entry and active_block is not None:
                    active_block["outcomes"].append(entry)
                continue
            if active_block and active_block.get("outcomes"):
                active_block["post_lines"].append(line)

    _finalize_block(active_block)

    # ── Priority B: coded prefixes (ILO 1:, CLO 2.3 |, CO 1 -, …) ────────
    # Classify each by its code: CLO/CO → course, ILO/LILO/TLO/ELO/LO → lesson,
    # PLO → program (skipped). The ILO course/lesson call uses _ilo_is_course.
    if not outcomes:
        for line in lines:
            m = _CODED.match(line)
            if m:
                st = _code_st(m.group(1))
                if st == 'po':
                    continue
                clean = m.group(2).strip()
                clean = re.sub(
                    r'\s+(?:Apply|Create|Analyze|Evaluate|Remember|Understand)'
                    r'(?:\s*/\s*\w+)?\s+(?:PO|ILO|CLO)[\d,\s]*$',
                    '', clean, flags=re.IGNORECASE,
                ).strip()
                _add(clean, source_type=st)
    _after_b = len(outcomes)

    # ── Priority C: numbered / bulleted list with Bloom's verb ────────────
    if not outcomes:
        for line in lines:
            if _NOT_ILO.match(line):
                continue
            clean = _BULLET_STRIP.sub('', line).strip()
            if _starts_with_bloom_verb(clean) and 4 <= len(clean.split()) <= 50:
                _add(clean)
    _after_c = len(outcomes)

    # ── Priority D: any line starting with a Bloom's verb ─────────────────
    if not outcomes:
        for line in lines:
            if _NOT_ILO.match(line):
                continue
            if _starts_with_bloom_verb(line) and 5 <= len(line.split()) <= 50:
                _add(line)
    _after_d = len(outcomes)

    parse_priority = "A" if outcomes else None
    for _letter, _n in (("B", _after_b), ("C", _after_c), ("D", _after_d)):
        if parse_priority is None and _n:
            parse_priority = _letter

    # ── Priority E: sentence-split fallback ───────────────────────────────
    # Last resort, for a document with no usable line structure at all. It must
    # hold to the SAME bar as Priority D — an outcome starts with a Bloom verb
    # and is not boilerplate — because a bare sentence split returns the whole
    # document: an 11-page syllabus yielded 213 "outcomes" whose first entries
    # were the university vision statement and the MISSION paragraph, and whose
    # last was the signature block. Sentences are the unit here; the filters are
    # Priority D's.
    if not outcomes:
        for s in re.split(r'(?<=[.!?])\s+', lilo_text.strip()):
            s = s.strip()
            if not s or _NOT_ILO.match(s):
                continue
            clean = _BULLET_STRIP.sub('', s).strip()
            if _starts_with_bloom_verb(clean) and 5 <= len(clean.split()) <= 50:
                _add(clean)
        if outcomes:
            parse_priority = "E"

    # ── Course-outcome rows get course-outcome ids ────────────────────────
    # `_add` stamps LILO-{idx} on everything it finds, the Course Outcomes block
    # included, and the two groups were told apart by `source_type` alone. That
    # left the COs unaddressable: every consumer keys a course outcome by CO-{n}
    # — co_descriptions, lilo_co_mapping, co_po_mapping, a question's
    # target_co_id — so a CO called "LILO-3" matched none of them and screens
    # that group by CO came up empty on a fully configured class.
    #
    # Only the CO rows are renumbered. Lesson LILOs keep the id they already
    # had, so every stored mapping, selected target and saved tag still resolves.
    _co_n = 0
    for _o in outcomes:
        if _o.get("source_type") == "co":
            _co_n += 1
            _o["id"] = f"CO-{_co_n}"

    logger.info(
        "Parsed %d ILO/lilo statements from syllabus (priority %s, %d course outcome(s))",
        len(outcomes), parse_priority or "-", _co_n,
    )
    if return_meta:
        return outcomes, {"priority": parse_priority, "count": len(outcomes)}
    return outcomes


# ---------------------------------------------------------------------------
# Core chunk alignment (used for content coverage dashboards)
# ---------------------------------------------------------------------------

def filter_lilos_for_topic(
    lilos: List[Dict],
    topic: str,
    source_text: str = "",
    top_n: int = 1,
    threshold: float = 0.25,
) -> List[Dict]:
    """
    For quiz scope: keep only the LILOs that are semantically relevant to
    the given topic (and optionally the source lesson text).

    Strategy:
      1. Build a query from: topic + first 300 words of source_text
      2. Encode query + each lilo's enriched_text with the sentence transformer
      3. Keep LILOs with cosine similarity >= threshold
      4. If fewer than 3 survive (degenerate case), fall back to top_n=5 by score
      5. Return the survivors in their original order (preserves curriculum sequence)

    Returns the filtered list, or the original list if no topic is given.
    """
    if not topic or not lilos:
        return lilos

    content_blocks = build_content_blocks_from_lilos(lilos)
    matched_blocks = select_content_blocks_for_topic(content_blocks, topic, source_text, top_n=top_n, threshold=threshold)
    if matched_blocks:
        matched_ids = {block["id"] for block in matched_blocks}
        filtered = [lilo for lilo in lilos if lilo.get("content_block_id") in matched_ids]
        logger.info(
            f"filter_lilos_for_topic: matched {len(matched_blocks)} syllabus content block(s) and kept {len(filtered)}/{len(lilos)} LILOs "
            f"for topic '{topic[:60]}'"
        )
        return filtered

    import numpy as np

    query_parts = [topic.strip()]
    if source_text:
        words = source_text.split()
        query_parts.append(" ".join(words[:300]))
    query = " ".join(query_parts)

    model = _get_model()
    query_vec = model.encode([query], normalize_embeddings=True)[0]

    scored = []
    for lilo in lilos:
        lilo_text = (
            lilo.get("enriched_text")
            or lilo.get("text")
            or ""
        )
        lilo_vec = model.encode([lilo_text], normalize_embeddings=True)[0]
        sim = float(np.dot(query_vec, lilo_vec))
        scored.append((sim, lilo))

    # Keep those above threshold
    relevant = [(s, l) for s, l in scored if s >= threshold]

    if len(relevant) < 3:
        # Fallback: top-3 with a soft floor of 0.20 to avoid truly irrelevant LILOs
        scored_sorted = sorted(scored, key=lambda x: x[0], reverse=True)
        candidates = [x for x in scored_sorted if x[0] >= 0.20]
        relevant = candidates[:min(3, len(candidates))] if candidates else scored_sorted[:1]

    # Re-order to original curriculum sequence
    relevant_ids = {id(l) for _, l in relevant}
    filtered = [l for l in lilos if id(l) in relevant_ids]

    logger.info(
        f"filter_lilos_for_topic: kept {len(filtered)}/{len(lilos)} LILOs "
        f"for topic '{topic[:60]}' (threshold={threshold})"
    )
    return filtered


def semantic_filter_lilos(
    lilos: List[Dict],
    topic: str,
    source_text: str = "",
    min_sim: float = 0.30,
    min_count: int = 3,
) -> List[Dict]:
    """
    Per-LILO semantic scoring filter for quiz topic scoping.

    Scores each LILO individually against the quiz topic using sentence
    transformer embeddings, then keeps only those above ``min_sim``.
    A ``min_count`` floor ensures at least that many LILOs survive (by
    taking the top-N if the threshold would discard too many).

    This is the primary fix for single-block syllabi where
    ``select_content_blocks_for_topic`` always returns the one block and
    all 9+ LILOs survive unchanged.
    """
    if not topic or not lilos:
        return lilos

    import numpy as np

    query_parts = [topic.strip()]
    if source_text:
        words = source_text.split()
        query_parts.append(" ".join(words[:300]))
    query = " ".join(query_parts)

    model = _get_model()
    query_vec = model.encode([query], normalize_embeddings=True)[0]

    scored = []
    for lilo in lilos:
        lilo_text = lilo.get("enriched_text") or lilo.get("text") or ""
        lilo_vec = model.encode([lilo_text], normalize_embeddings=True)[0]
        sim = float(np.dot(query_vec, lilo_vec))
        scored.append((sim, lilo))
        logger.debug(
            f"  [semantic_filter] {lilo.get('id', '?'):10s} sim={sim:.4f}  '{lilo_text[:60]}'"
        )

    # Keep those above the minimum similarity floor
    relevant = [(s, l) for s, l in scored if s >= min_sim]

    # If not enough survive, fall back to top-min_count by score
    if len(relevant) < min_count:
        sorted_scored = sorted(scored, key=lambda x: x[0], reverse=True)
        relevant = sorted_scored[:min(min_count, len(sorted_scored))]

    # Preserve original curriculum order
    relevant_ids = {id(l) for _, l in relevant}
    filtered = [l for l in lilos if id(l) in relevant_ids]

    logger.info(
        f"semantic_filter_lilos: {len(lilos)} → {len(filtered)} LILOs retained "
        f"for topic '{topic[:60]}' (min_sim={min_sim})"
    )
    return filtered


# Week-1 orientation / housekeeping rows: VMGO, syllabus walkthrough, grading
# system, course requirements, academic-integrity & classroom policies, responsible
# AI/house rules, prerequisite review. These live mostly in the TOPIC cell, not the
# outcome text, so this pattern is checked against topic/learning-content too.
#
# EVERY ALTERNATIVE MUST BE A MULTI-WORD INSTITUTIONAL PHRASE. Never add a bare
# subject word here. A single `\bvision\b` shipped in the old (now deleted)
# _ORIENTATION_KW and silently deleted every "computer vision" LILO from the
# tagging pool, so questions generated for those outcomes were re-tagged onto
# unrelated ones and their attainment read as zero forever. The same trap is
# waiting in `orientation` (gradient orientation, orientation histogram / HOG),
# `mission` (mission-critical, mission planning) and `policy` (policy gradient,
# policy iteration) — all of which are ordinary course content. Real ISU week-1
# rows read "Orientation Quality Policy BSCS Program Outcomes Goals of the
# College ...", so the qualified phrases below still catch them; the bare word
# was never doing the work. Regression cases: _INSTITUTIONAL_KW_CASES.
_INSTITUTIONAL_KW = re.compile(
    r"\bvmgo\b"
    r"|\bvision[,\s]+(?:and\s+)?mission\b|\bmission[,\s]+(?:and\s+)?vision\b"
    r"|\binstitut(?:e|ion|ional)\s+(?:vision|mission|goals)\b"
    r"|\b(?:isu|university|college|school|campus)\s+(?:vision|mission|goals)\b"
    r"|\bgoals?\s+of\s+the\s+college\b|\bquality\s+policy\b"
    r"|\b(?:course|class|program|student|freshman|university|college|school|semester)\s+orientation\b"
    r"|\borientation\s+(?:week|day|program|programme|session|activit)\w*"
    r"|\bgrading\s+system|\bcourse\s+requirements|\bcourse\s+syllabus|\bsyllabus\s+overview"
    r"|\bacademic\s+integrity|\bacademic\s+honesty|\bclassroom\s+polic|\bhouse\s+rules"
    r"|\bresponsible[\s-]*ai\b|\bassessment\s+schedule"
    r"|\bprerequisite\s+review|\breview\s+of\s+prerequisite",
    re.IGNORECASE,
)

# Checked by test_institutional_filter.py. Left beside the pattern deliberately:
# the failure mode is a plausible-looking word quietly eating course content, and
# the only defence is a list of the content phrases that must survive.
_INSTITUTIONAL_KW_CASES = [
    # (text, should_be_flagged_as_institutional)
    ("computer vision",                  False),
    ("machine vision",                   False),
    ("vision transformer",               False),
    ("mission-critical systems",         False),
    ("mission planning",                 False),
    ("gradient orientation",             False),
    ("orientation histogram",            False),
    ("edge orientation and magnitude",   False),
    ("object orientation",               False),
    ("policy gradient methods",          False),
    ("University Vision and Mission",    True),
    ("VMGO Orientation",                 True),
    ("Institutional Vision",             True),
    ("Quality Policy",                   True),
    ("Course Orientation",               True),
    ("Grading System",                   True),
    ("Academic Integrity",               True),
    ("Course Requirements",              True),
    ("Orientation Quality Policy BSCS Program Outcomes Goals of the College "
     "Course Overview and Requirements",  True),
]

# Fields scanned by both institutional checks. The outcome text alone is not
# enough (the giveaway is usually in the topic cell) and the topic cell alone is
# not enough (stale enriched LILOs carry it in key_concepts), so both callers
# read the same full set.
_INSTITUTIONAL_FIELDS = ("topic_title", "learning_content", "text", "enriched_text")


def _institutional_blob(ilo: Dict) -> str:
    """Every field either institutional check looks at, as one string."""
    parts = [str(ilo.get(k) or "") for k in _INSTITUTIONAL_FIELDS]
    parts.append(" ".join(str(kc) for kc in (ilo.get("key_concepts") or [])))
    return " ".join(parts)


def _is_housekeeping_row(ilo: Dict) -> bool:
    """True when a syllabus-unmapped LILO is a non-content orientation/housekeeping
    row (VMGO, policies, grading walkthrough). Used to leave such rows without a
    (misleading) forced CO."""
    return bool(_INSTITUTIONAL_KW.search(_institutional_blob(ilo)))


def map_ilos_to_cos(ilos: List[Dict], cos: List[Dict]) -> List[Dict]:
    """
    Assign each ILO to its semantically closest Course Outcome (CO).
    Adds ``co_id`` and ``co_text`` fields to each ILO in-place.

    Orientation / housekeeping rows the syllabus left unmapped are flagged
    ``co_unassigned`` and given no CO (rather than a forced, misleading one).

    If there are no COs, ILOs are returned unmodified.
    """
    if not cos or not ilos:
        return ilos

    import numpy as np

    model = _get_model()

    co_texts  = [c.get("enriched_text") or c.get("text") or "" for c in cos]
    co_vecs   = model.encode(co_texts, normalize_embeddings=True)

    # Build a normalised ID lookup: "CLO-1" / "CLO1" → index in cos list
    def _norm_clo_id(raw: str) -> str:
        return re.sub(r"[\s\-]", "", raw.upper())

    co_id_index: Dict[str, int] = {
        _norm_clo_id(c["id"]): i for i, c in enumerate(cos)
    }

    def _canon(raw_id: str) -> str:
        # Normalise CLO-N / CLO1 / CO N → CO-N so it matches co_po_mapping keys
        _m = re.match(r"^(?:CLO|CO)[-\s]?(\d+)$", str(raw_id).strip(), re.IGNORECASE)
        return f"CO-{_m.group(1)}" if _m else str(raw_id)

    for ilo in ilos:
        # ── Step 1: prefer explicit CLO alignment from the course-plan table ─
        # DOCX ISU syllabi set ``aligned_clo`` directly on each ILO so we can
        # skip the expensive semantic match and use the authoritative mapping.
        # A row listing SEVERAL CLOs ("CLO2, CLO3") is an authoritative
        # multi-CO mapping: ALL of them go into co_ids; the primary co_id is
        # the semantically closest of the candidates.
        best_idx = None
        explicit_idxs: List[int] = []
        aligned_clo_raw = ilo.get("aligned_clo", "")
        if aligned_clo_raw:
            clo_codes = re.findall(r"CLO\s*-?\s*\d+|CO\s*-?\s*\d+", aligned_clo_raw.upper())
            clo_codes = [_norm_clo_id(c) for c in clo_codes]
            candidate_idxs = [
                co_id_index[code] for code in dict.fromkeys(clo_codes) if code in co_id_index
            ]
            explicit_idxs = candidate_idxs
            if len(candidate_idxs) == 1:
                best_idx = candidate_idxs[0]
            elif len(candidate_idxs) > 1:
                # Multiple CLOs listed for this week — pick semantically closest
                # among the candidates rather than blindly taking the first one.
                ilo_text = ilo.get("enriched_text") or ilo.get("text") or ""
                ilo_vec  = model.encode([ilo_text], normalize_embeddings=True)[0]
                sims = [float(np.dot(ilo_vec, co_vecs[i])) for i in candidate_idxs]
                best_idx = candidate_idxs[int(max(range(len(sims)), key=lambda i: sims[i]))]

        # ── Step 1b: orientation / housekeeping rows stay UNASSIGNED ─────────
        # Only reachable when the syllabus gave no explicit aligned_clo (best_idx
        # is None). Rather than force these onto a semantically nearest CO — a
        # misleading badge — leave co_id empty and flag them. Generation never
        # targets a LILO with no CO, so they stay out of quizzes/exams unless the
        # professor deliberately assigns one in the Outcomes Editor.
        if best_idx is None and not aligned_clo_raw and _is_housekeeping_row(ilo):
            ilo["co_id"] = None
            ilo["co_text"] = ""
            ilo["co_ids"] = []
            ilo["co_unassigned"] = True
            ilo["co_note"] = "orientation"
            logger.debug("map_ilos_to_cos: %s -> UNASSIGNED (orientation/housekeeping row)",
                         ilo.get("id"))
            continue

        # ── Step 2: fall back to semantic similarity ─────────────────────────
        semantic_sims = None
        if best_idx is None:
            ilo_text = ilo.get("enriched_text") or ilo.get("text") or ""
            ilo_vec  = model.encode([ilo_text], normalize_embeddings=True)[0]
            semantic_sims = [float(np.dot(ilo_vec, cv)) for cv in co_vecs]
            best_idx = int(max(range(len(semantic_sims)), key=lambda i: semantic_sims[i]))

        ilo["co_id"]   = _canon(cos[best_idx]["id"])
        ilo["co_text"] = cos[best_idx]["text"]
        # This row now HAS a CO (e.g. professor assigned one in the editor) —
        # drop any stale orientation flag from a previous unassigned pass.
        ilo.pop("co_unassigned", None)
        ilo.pop("co_note", None)

        # Per-ILO trace (which path decided the CO) — the previous version only
        # logged an aggregate count, making edit/realign debugging impossible.
        logger.debug(
            "map_ilos_to_cos: %s -> %s via %s%s",
            ilo.get("id"), ilo["co_id"],
            "explicit aligned_clo" if explicit_idxs else "semantic",
            ("" if semantic_sims is None
             else f" (sim={semantic_sims[best_idx]:.3f})"),
        )

        # ── co_ids (multi-CO) + co_suggestions ────────────────────────────────
        # Explicit syllabus mapping → authoritative co_ids (ALL listed CLOs).
        # Semantic-only → co_ids = [primary]; near-tied other COs are exposed as
        # SUGGESTIONS only (never auto-added to the official mapping).
        if explicit_idxs:
            ilo["co_ids"] = list(dict.fromkeys(_canon(cos[i]["id"]) for i in explicit_idxs))
        else:
            ilo["co_ids"] = [ilo["co_id"]]
            if semantic_sims is not None:
                best_sim = semantic_sims[best_idx]
                near = [
                    _canon(cos[i]["id"])
                    for i, s in enumerate(semantic_sims)
                    if i != best_idx and best_sim > 0 and s >= 0.92 * best_sim
                ]
                if near:
                    ilo["co_suggestions"] = near[:3]

        # ── Step 3: propagate only actual PO codes (not GA codes) ────────────
        # GA codes (GA1, GA2 …) are Graduate Attributes — a different axis from
        # Program Outcomes (PO1, PO2 …).  Only PO\d+ codes are valid po_codes.
        parent_pos = [
            p for p in (cos[best_idx].get("po_codes") or [])
            if re.match(r'^PO\s*\d+$', p, re.IGNORECASE)
        ]
        if parent_pos:
            existing = list(ilo.get("po_codes") or [])
            merged = list(dict.fromkeys(existing + parent_pos))
            ilo["po_codes"] = sorted(merged)

    logger.info(f"map_ilos_to_cos: linked {len(ilos)} ILOs → {len(cos)} COs")
    return ilos


def align_chunks_to_lilos(
    chunks: List[str],
    lilos: List[Dict],
    threshold: float = 0.45,
) -> List[Dict]:
    """
    Map each text chunk to its best-matching lilo using composite scoring.

        Uses the same composite formula as tag_questions_with_lilos:
            FinalScore = 0.45 * cosine + 0.20 * outcome_jaccard
                                 + 0.15 * content_overlap + 0.20 * bloom_match

    Args:
        chunks:    List of text strings (lesson content chunks)
        lilos:      Output of parse_lilos()
        threshold: Minimum composite score to assign an lilo (default 0.45).

    Returns list of dicts with chunk_index, best_lilo_id, similarity_score,
    confidence, and all_scores.
    """
    if not lilos:
        logger.warning("No lilos provided -- skipping alignment")
        return [
            {
                "chunk_index":    i,
                "chunk_preview":  chunk[:80],
                "best_lilo_id":    "LILO-NONE",
                "best_lilo_text":  None,
                "similarity_score": 0.0,
                "confidence":     0.0,
                "all_scores":     {},
            }
            for i, chunk in enumerate(chunks)
        ]

    import numpy as np

    # Step 8: guard against embedding model load failures — return LILO-NONE
    # for all chunks so the caller (assessment.py align phase) can degrade
    # gracefully instead of crashing the SSE stream.
    try:
        model = _get_model()
    except Exception as _model_err:
        logger.error(f"[align_chunks_to_lilos] embedding model unavailable: {_model_err}")
        return [
            {
                "chunk_index":      i,
                "chunk_preview":    chunk[:80],
                "best_lilo_id":      "LILO-NONE",
                "best_lilo_text":    None,
                "similarity_score": 0.0,
                "confidence":       0.0,
                "all_scores":       {},
            }
            for i, chunk in enumerate(chunks)
        ]
    # Use outcome text plus syllabus course-plan context for semantic matching
    lilo_texts = [_lilo_alignment_text(l) for l in lilos]

    chunk_embs = model.encode(chunks,     convert_to_numpy=True, show_progress_bar=False)
    lilo_embs   = model.encode(lilo_texts,  convert_to_numpy=True, show_progress_bar=False)

    chunk_norm = chunk_embs / np.maximum(np.linalg.norm(chunk_embs, axis=1, keepdims=True), 1e-10)
    lilo_norm   = lilo_embs   / np.maximum(np.linalg.norm(lilo_embs,   axis=1, keepdims=True), 1e-10)
    sim_matrix = chunk_norm @ lilo_norm.T   # (n_chunks, n_lilos)

    # Use LLM-verified key concepts if available, else extract with KeyBERT/stems
    lilo_keywords = [
        set(l.get("key_concepts", [])) | _extract_keywords(_combine_alignment_parts(l.get("enriched_text"), l.get("text")))
        for l in lilos
    ]
    lilo_content_keywords = [
        _expand_phrases(_extract_keywords(_lilo_content_text(l)))
        for l in lilos
    ]

    results = []
    for i, (chunk, cos_scores) in enumerate(zip(chunks, sim_matrix)):
        # Expanded to unigrams to match the LILO-side sets (see _expand_phrases)
        chunk_kw    = _expand_phrases(_extract_keywords(chunk))
        chunk_bloom = classify_bloom(chunk)

        composite_scores = [
            0.45 * float(cos_scores[j])
            + 0.20 * _jaccard_score(chunk_kw, lilo_keywords[j])
            + 0.15 * _jaccard_score(chunk_kw, lilo_content_keywords[j])
            + 0.2 * _bloom_match_score(chunk_bloom, lilos[j]["bloom_level"])
            for j in range(len(lilos))
        ]

        best_idx   = int(np.argmax(composite_scores))
        best_score = composite_scores[best_idx]
        confidence = _softmax_confidence(composite_scores)

        if best_score >= threshold:
            best_lilo = lilos[best_idx]
        else:
            best_lilo   = {"id": "LILO-NONE", "text": None}
            best_score = 0.0

        results.append({
            "chunk_index":      i,
            "chunk_preview":    chunk[:80].replace("\n", " "),
            "best_lilo_id":      best_lilo["id"],
            "best_lilo_text":    best_lilo.get("text"),
            "similarity_score": round(best_score, 4),
            "confidence":       round(confidence, 4),
            "all_scores": {
                lilos[j]["id"]: round(composite_scores[j], 4)
                for j in range(len(lilos))
            },
        })

    matched = sum(1 for r in results if r["best_lilo_id"] != "LILO-NONE")
    logger.info(f"Chunk alignment done. Matched: {matched}/{len(chunks)}")
    return results


# ---------------------------------------------------------------------------
# Tag questions -- full composite pipeline (Steps 1-11)
# ---------------------------------------------------------------------------

def _is_orientation_lilo(lilo: Dict) -> bool:
    """True for LILOs that describe institutional vision/mission/orientation
    rather than course content.  These should never tag a content question.

    Shares _INSTITUTIONAL_KW with _is_housekeeping_row: one concept, one pattern.
    They used to be two separate regexes and drifted apart — this one grew a bare
    `\\bvision\\b` that ate "computer vision"."""
    return bool(_INSTITUTIONAL_KW.search(_institutional_blob(lilo)))


def tag_questions_with_lilos(
    questions: List[Dict],
    chunks: List[str],            # kept for API compatibility; not used
    lilo_text: str,
    threshold: float = 0.30,
    close_score_delta: float = 0.15,
    top_k: int = 3,
    llm_max_calls: int = 10,
    enriched_lilos: Optional[List[Dict]] = None,
    topic: Optional[str] = None,
    cross_block_penalty: float = 0.5,
) -> List[Dict]:
    """
    Tag each generated question with its best-matching lilo.

    Implements the full recommended composite pipeline (Steps 1-11).

    Adds to each question dict:
      "lilo_tag": {
        "lilo_id":        "LILO-2",
        "lilo_text":      "Explain membrane transport",
        "score":         0.846,        # composite FinalScore
        "confidence":    0.84,         # softmax confidence (0-1)
        "bloom_level":   "understand", # Bloom level DETECTED from the question
        "lilo_bloom":     "understand", # Bloom level the lilo TARGETS
        "bloom_aligned": True,         # strict equality of the two above
        "bloom_gap":     0,            # signed distance: -1 = one level below target
        "format_ceiling":      "understand",  # highest level this question TYPE can carry
        "bloom_above_ceiling": True,   # detected level exceeds that ceiling — the
                                       # item reads higher than its format can measure.
                                       # Diagnostic only: it never changes a score, and
                                       # it deliberately coexists with bloom_aligned=True.
                                       # None (both fields) when the type is unrecognised.
            "score_breakdown": {
                    "cosine": 0.89,
                    "jaccard": 0.67,
                    "content": 0.71,
                    "bloom":   1.0,
                },
            "score_available": {       # did the channel have inputs to compare?
                    "cosine":  True,   # False means nothing was measured, so the
                    "jaccard": True,   # matching 0.0 above is absence, not a
                    "content": True,   # non-match. A display normaliser may drop
                    "bloom":   True,   # a weight only when this says False.
                },
            "semantic_best": {         # what the COMPOSITE alone picked, when that
                    "lilo_id": "LILO-22",   # is not the lilo_id above. None when
                    "lilo_text": "...",     # they agree. The source-ILO override
                    "score":  0.4471,       # makes lilo_id == source_ilo_id for every
                    "margin": 0.0183,       # generated question, so a tag-vs-source
                },                          # comparison is 0 by construction and this
                                            # is the only remaining record of drift.
                                            # Diagnostic: nothing scores or attains off it.
        "llm_validated": False,        # True if LLM was consulted
      }
    """
    if enriched_lilos is not None:
        lilos = enriched_lilos
    else:
        lilos = parse_lilos(lilo_text)
        lilos = enrich_lilos_with_llm(lilos)

    # ── Source-LILO invariant ──────────────────────────────────────────────
    # A LILO that some question was GENERATED FOR is never removed from the
    # candidate pool — not by the institutional filter, not by topic scoping.
    # source_ilo_id is provenance: the professor selected that outcome and the
    # generator wrote the question against it. No heuristic outranks that.
    #
    # This is the structural fix for the "computer vision" incident. The regex
    # was wrong, but the reason a wrong regex could corrupt attainment data is
    # that both filters ran BEFORE the source-ILO override below, so the
    # override's lookup silently resolved to None and every affected question
    # fell through to composite matching. Fixing only the regex would leave the
    # next bad pattern free to do the same thing.
    _source_ids = {
        q.get("source_ilo_id") for q in questions if q.get("source_ilo_id")
    }
    _pool_ids = {l.get("id") for l in lilos}
    _missing_source_ids = sorted(_source_ids - _pool_ids)
    if _missing_source_ids:
        # Never fabricate the outcome — a synthesised LILO would report
        # attainment for something no syllabus row backs. Warn, name the ids,
        # and let those questions fall through to normal composite matching.
        logger.warning(
            f"[tag_questions_with_lilos] {len(_missing_source_ids)} source_ilo_id(s) "
            f"absent from the supplied LILO pool: {_missing_source_ids} — those "
            f"questions fall back to composite matching (no LILO is fabricated)"
        )

    def _protected(lilo: Dict) -> bool:
        return lilo.get("id") in _source_ids

    # ── Defensive: drop institutional vision/mission/orientation LILOs ──────
    # These are syllabus boilerplate (Week 1 orientation), never appropriate
    # tags for content questions.  Stale enriched_lilos in the DB may still
    # contain them; the parser fix only affects new uploads.
    _orientation_lilos = [
        l for l in lilos if _is_orientation_lilo(l) and not _protected(l)
    ]
    _exempted = [l for l in lilos if _is_orientation_lilo(l) and _protected(l)]
    if _exempted:
        # Loud on purpose: this is the tripwire for the next bad pattern.
        logger.warning(
            f"[tag_questions_with_lilos] institutional filter matched "
            f"{len(_exempted)} LILO(s) that questions were generated for — "
            f"KEPT by the source-LILO invariant: {[l.get('id') for l in _exempted]}. "
            f"Check _INSTITUTIONAL_KW for a phrase that also reads as course content."
        )
    if _orientation_lilos:
        _drop = {id(l) for l in _orientation_lilos}
        lilos = [l for l in lilos if id(l) not in _drop]
        logger.info(
            f"[tag_questions_with_lilos] dropped {len(_orientation_lilos)} "
            f"orientation/vision-mission LILO(s) from the candidate pool"
        )

    # ── Topic-scoped LILO pool ─────────────────────────────────────────────
    # When the assessment carries a topic string, restrict the candidate
    # pool to LILOs whose content_block matches the topic.  This prevents
    # a Week-2 "Real-World Data Formulation · Graphs" LILO from winning over
    # a Week-7 "Mining Network Data" LILO just because both mention "graph".
    # Works for any subject — relies only on the content_block_id field that
    # build_content_blocks_from_lilos assigns to every LILO from any syllabus.
    if topic and lilos:
        try:
            _all_blocks = build_content_blocks_from_lilos(lilos)
            # Multi-topic exam: split on '·' and union matched blocks per subtopic
            _subtopics = [t.strip() for t in topic.split('·') if t.strip()] or [topic]
            _allowed_block_ids: set = set()
            for _sub in _subtopics:
                _matched = select_content_blocks_for_topic(
                    _all_blocks, _sub, "", top_n=2
                )
                _allowed_block_ids.update(b["id"] for b in _matched)
            if _allowed_block_ids:
                # Source LILOs survive scoping too (see the invariant above):
                # a question generated for a Week-11 outcome must still be able
                # to tag to it even if the topic string only resolved Week-12.
                _scoped = [
                    l for l in lilos
                    if l.get("content_block_id") in _allowed_block_ids
                    or _protected(l)
                ]
                _kept_by_source = [
                    l for l in _scoped
                    if l.get("content_block_id") not in _allowed_block_ids
                ]
                if _scoped:
                    logger.info(
                        f"[tag_questions_with_lilos] topic-scoped pool: "
                        f"{len(_scoped)}/{len(lilos)} LILOs across "
                        f"{len(_allowed_block_ids)} content block(s) for topic='{topic[:60]}'"
                        + (f" (+{len(_kept_by_source)} out-of-scope kept as question sources)"
                           if _kept_by_source else "")
                    )
                    lilos = _scoped
        except Exception as _scope_err:
            logger.warning(
                f"[tag_questions_with_lilos] topic-scoping failed ({_scope_err}); "
                "falling back to full LILO pool"
            )

    if not lilos:
        for q in questions:
            _qb = classify_bloom(q.get("question", ""))
            _qc, _qa = _format_ceiling_flags(q, _qb)
            q["lilo_tag"] = {
                "lilo_id":        "LILO-NONE",
                "lilo_text":      None,
                "score":         0.0,
                "confidence":    0.0,
                "bloom_level":   _qb,
                "lilo_bloom":     None,
                "bloom_aligned": False,
                "bloom_gap":     None,   # no LILO to measure against
                # still measurable without a LILO: the ceiling is a property of
                # the question's own format, not of what it was tagged to
                "format_ceiling":      _qc,
                "bloom_above_ceiling": _qa,
                "score_breakdown": {"cosine": 0.0, "jaccard": 0.0, "content": 0.0, "bloom": 0.5},
                # Nothing was measured on this path — the zeros are absence, not a
                # non-match, so no normaliser should read them as evidence.
                "score_available": {"cosine": False, "jaccard": False, "content": False, "bloom": False},
                "llm_validated": False,
            }
        return questions

    import numpy as np

    content_blocks = build_content_blocks_from_lilos(lilos)

    # Fast id → lilo lookup (used to carry co_id into each question tag)
    lilo_by_id: Dict[str, Dict] = {l["id"]: l for l in lilos}

    # -- Step 3: keyword sets (computed once per lilo) ----------------------
    # Expand multi-word key concepts (e.g. "data abstraction") into both the
    # original phrase AND its component unigrams so they can overlap with
    # question keywords regardless of whether KeyBERT or fallback tokenization
    # is used on the question side.
    def _expand_concepts(concepts: list) -> set:
        expanded: set = set()
        for kc in concepts:
            kc_lower = kc.lower().strip()
            expanded.add(kc_lower)
            expanded.update(kc_lower.split())
        return expanded

    lilo_keywords = [
        _expand_concepts(l.get("key_concepts", []))
        | _extract_keywords(_combine_alignment_parts(l.get("enriched_text"), l.get("text")))
        for l in lilos
    ]
    lilo_content_keywords = [
        _expand_phrases(_extract_keywords(_lilo_content_text(l)))
        for l in lilos
    ]

    # -- Step 4: embedding generation --------------------------------------
    # Step 8: guard against model load failure — return LILO-NONE tags so
    # tag phase never crashes the SSE stream (questions already generated).
    try:
        model = _get_model()
    except Exception as _model_err:
        logger.error(f"[tag_questions_with_lilos] embedding model unavailable: {_model_err}")
        _none_tag = {
            "lilo_id": "LILO-NONE", "lilo_text": None, "score": 0.0, "confidence": 0.0,
            "bloom_level": None, "lilo_bloom": None, "bloom_aligned": False, "bloom_gap": None,
            "format_ceiling": None, "bloom_above_ceiling": None,
            "score_breakdown": {"cosine": 0.0, "jaccard": 0.0, "content": 0.0, "bloom": 0.5},
            # Nothing was measured on this path — the zeros are absence, not a
            # non-match, so no normaliser should read them as evidence.
            "score_available": {"cosine": False, "jaccard": False, "content": False, "bloom": False},
            "llm_validated": False,
            "content_alignment": {"topic_block_aligned": False, "block_id": None,
                                  "topic_title": None, "learning_content": None,
                                  "week": None, "po_codes": [], "score": 0.0, "keyword_hits": []},
        }
        for q in questions:
            _qb = classify_bloom(q.get("question", ""))
            q["lilo_tag"] = dict(_none_tag)
            q["lilo_tag"]["bloom_level"] = _qb
            (q["lilo_tag"]["format_ceiling"],
             q["lilo_tag"]["bloom_above_ceiling"]) = _format_ceiling_flags(q, _qb)
        return questions
    lilo_texts = [_lilo_alignment_text(l) for l in lilos]
    q_texts   = [q.get("question", "") for q in questions]
    content_block_texts = [_content_block_alignment_text(block) for block in content_blocks]

    q_embs   = model.encode(q_texts,   convert_to_numpy=True, show_progress_bar=False)
    lilo_embs = model.encode(lilo_texts, convert_to_numpy=True, show_progress_bar=False)
    content_embs = model.encode(content_block_texts, convert_to_numpy=True, show_progress_bar=False) if content_block_texts else np.zeros((0, q_embs.shape[1]))

    # Cosine similarity: L2-normalise then dot product
    q_norm   = q_embs   / np.maximum(np.linalg.norm(q_embs,   axis=1, keepdims=True), 1e-10)
    lilo_norm = lilo_embs / np.maximum(np.linalg.norm(lilo_embs, axis=1, keepdims=True), 1e-10)
    cos_sim  = q_norm @ lilo_norm.T    # (n_questions, n_lilos)
    content_norm = content_embs / np.maximum(np.linalg.norm(content_embs, axis=1, keepdims=True), 1e-10) if len(content_block_texts) else np.zeros((0, 0))
    content_cos  = q_norm @ content_norm.T if len(content_block_texts) else np.zeros((len(questions), 0))
    content_block_keywords = [_expand_phrases(_extract_keywords(_content_block_alignment_text(block))) for block in content_blocks]

    llm_calls_used = 0

    # Build LILO id → content block id map for the cross-reference fallback
    lilo_to_block: Dict[str, str] = {}
    for block in content_blocks:
        for lilo_id in block.get("ilo_ids", []):
            lilo_to_block[lilo_id] = block["id"]

    for q_index, (q, cos_scores) in enumerate(zip(questions, cos_sim)):
        q_text  = q.get("question", "")
        # Step 2 — the cognitive level the QUESTION actually demands. Invariant:
        # this is assigned exactly once and never reassigned for the rest of the
        # iteration. It is not the LILO's level and must never be set from one.
        question_bloom = classify_bloom(q_text)
        # Step 2b — the highest level this question's FORMAT can carry, and
        # whether the level detected above exceeds it. Independent of any LILO:
        # a True/False item that reads as "analyze" is over its instrument's
        # ceiling no matter which outcome it is tagged to.
        q_ceiling, q_above_ceiling = _format_ceiling_flags(q, question_bloom)
        # Expand question keywords to include unigrams from any phrases
        _q_kw_raw = _extract_keywords(q_text)
        q_kw: set = set()
        for kw in _q_kw_raw:
            q_kw.add(kw)
            q_kw.update(kw.split())  # ensure unigrams are always present

        best_content_alignment = {
            "topic_block_aligned": False,
            "block_id": None,
            "topic_title": None,
            "learning_content": None,
            "week": None,
            "po_codes": [],
            "score": 0.0,
            "keyword_hits": [],
        }
        if content_blocks:
            content_candidates = []
            for block_idx, block in enumerate(content_blocks):
                keyword_hits = sorted(q_kw & content_block_keywords[block_idx])
                jaccard = _jaccard_score(q_kw, content_block_keywords[block_idx])
                cosine = float(content_cos[q_index][block_idx])
                score = 0.70 * cosine + 0.30 * jaccard
                content_candidates.append((score, keyword_hits, block))
            content_candidates.sort(key=lambda item: item[0], reverse=True)
            best_content_score, best_content_hits, best_content_block = content_candidates[0]
            # A question is topic-block aligned if:
            #   (a) its embedding+keyword score vs the best block exceeds 0.30, OR
            #   (b) its source_ilo_id or top-candidate LILO is explicitly
            #       assigned to a content block (curriculum-level ground truth).
            source_id_for_block = q.get("source_ilo_id") or ""
            _lilo_driven_block_id = lilo_to_block.get(source_id_for_block)
            _lilo_driven_aligned = (
                _lilo_driven_block_id is not None
                and _lilo_driven_block_id == best_content_block.get("id")
            )
            best_content_alignment = {
                "topic_block_aligned": best_content_score >= 0.30 or _lilo_driven_aligned,
                "block_id": best_content_block.get("id"),
                "topic_title": best_content_block.get("topic_title"),
                "learning_content": best_content_block.get("learning_content"),
                "week": best_content_block.get("week"),
                "po_codes": best_content_block.get("po_codes", []),
                "score": round(best_content_score, 4),
                "keyword_hits": best_content_hits[:8],
            }

        # -- Steps 5 & 6: three channels + composite score -----------------
        # Source-ILO shortcut: if question was generated targeting a specific ILO,
        # use that ILO as authoritative and skip full composite search.
        # We still compute the composite for the source ILO so score_breakdown
        # is populated for diagnostics / the alignment visualisation.
        source_ilo_id = q.get("source_ilo_id")
        _source_override_lilo = None
        if source_ilo_id:
            _source_override_lilo = next((l for l in lilos if l["id"] == source_ilo_id), None)

        candidates = []
        # Per-question best content block (computed above as best_content_alignment).
        # Penalise candidate LILOs whose own content block disagrees with the
        # question's best block — this catches lexical false-matches that
        # otherwise slip past topic scoping (e.g. "network" mentioned in two
        # different weeks).  Penalty only fires when the question's block
        # assignment is confident (score >= 0.30).
        _q_block_id = best_content_alignment.get("block_id")
        _q_block_confident = best_content_alignment.get("score", 0.0) >= 0.30
        for j, lilo in enumerate(lilos):
            cosine    = float(cos_scores[j])
            jaccard   = _jaccard_score(q_kw, lilo_keywords[j])
            content_m = _jaccard_score(q_kw, lilo_content_keywords[j])
            bloom_m   = _bloom_match_score(question_bloom, lilo["bloom_level"])
            composite = 0.45 * cosine + 0.20 * jaccard + 0.15 * content_m + 0.20 * bloom_m
            if (
                _q_block_confident
                and _q_block_id
                and lilo.get("content_block_id")
                and lilo.get("content_block_id") != _q_block_id
            ):
                composite *= cross_block_penalty
            candidates.append({
                "lilo_id":     lilo["id"],
                "lilo_text":   lilo["text"],
                "lilo_bloom":  lilo["bloom_level"],
                "cosine":     cosine,
                "jaccard":    jaccard,
                "content_match": content_m,
                "bloom_match": bloom_m,
                "composite":  composite,
                # Was the channel able to measure anything at all? A Jaccard of
                # 0 means two different things: "these share no vocabulary"
                # (a real, low match) and "one side had no keywords to compare"
                # (nothing was measured). The display normaliser has to tell
                # them apart — excusing a genuine zero inflates the score and,
                # at the threshold, ranks a worse question above a better one.
                # Availability is a property of the INPUTS, never of the value.
                "jaccard_available": bool(q_kw and lilo_keywords[j]),
                "content_available": bool(q_kw and lilo_content_keywords[j]),
                "bloom_available":   bool(question_bloom and lilo.get("bloom_level")),
            })

        # -- Step 7: threshold gate  /  Step 8: top-k selection -----------
        candidates.sort(key=lambda c: c["composite"], reverse=True)
        top_candidates = [c for c in candidates if c["composite"] >= threshold][:top_k]

        # If the question has a source_ilo_id, ensure it is always reachable
        # even if it fell below the threshold (composite scoring can still be low
        # when enrichment hasn't run, but the source is authoritative).
        if _source_override_lilo and not top_candidates:
            source_cand = next(
                (c for c in candidates if c["lilo_id"] == source_ilo_id), None
            )
            if source_cand:
                top_candidates = [source_cand]

        if not top_candidates:
            q["lilo_tag"] = {
                "lilo_id":        "LILO-NONE",
                "lilo_text":      None,
                "score":         0.0,
                "confidence":    round(_softmax_confidence([c["composite"] for c in candidates]), 4),
                "bloom_level":   question_bloom,
                "lilo_bloom":     None,
                "bloom_aligned": False,
                "bloom_gap":     None,
                "format_ceiling":      q_ceiling,
                "bloom_above_ceiling": q_above_ceiling,
                "score_breakdown": {"cosine": 0.0, "jaccard": 0.0, "content": 0.0, "bloom": 0.5},
                # Nothing was measured on this path — the zeros are absence, not a
                # non-match, so no normaliser should read them as evidence.
                "score_available": {"cosine": False, "jaccard": False, "content": False, "bloom": False},
                "content_alignment": best_content_alignment,
                "llm_validated": False,
            }
            continue

        best          = top_candidates[0]
        llm_validated = False
        # What the composite alone would have picked, captured BEFORE the source
        # override rewrites `best`. Without this the disagreement is unrecoverable:
        # the override makes the stored tag equal the source id for every generated
        # question, so a tag-vs-source comparison downstream is zero by
        # construction and reports "no drift" whether or not any exists.
        _semantic_best = best

        # -- Source-ILO override: generated questions know their target ILO --
        # Override the composite-chosen best with the generation source ILO
        # and skip the LLM tiebreak entirely — the source is authoritative.
        if _source_override_lilo:
            source_cand = next(
                (c for c in candidates if c["lilo_id"] == source_ilo_id), None
            )
            if source_cand:
                best = source_cand
                llm_validated = False  # source ILO is authoritative; no LLM needed
                # NOTE: question_bloom is deliberately NOT reassigned here. It used
                # to be overwritten with best["lilo_bloom"] ("trust the generation
                # target"), which made bloom_aligned compare a value to itself —
                # vacuously True for every source-ILO question — and silently
                # disabled the Bloom half of _false_align and the mismatch branch
                # of _bloom_note. The source ILO decides WHICH LILO wins; it does
                # not get to decide what the question actually asks for.
        else:
            # -- Step 9: close-score decision (only for non-ILO-driven questions) --
            if (
                len(top_candidates) >= 2
                and (top_candidates[0]["composite"] - top_candidates[1]["composite"]) < close_score_delta
                and llm_calls_used < llm_max_calls
            ):
                llm_choice = _llm_validate_alignment(q_text, top_candidates)
                llm_calls_used += 1
                if llm_choice:
                    matched = next((c for c in top_candidates if c["lilo_id"] == llm_choice), None)
                    if matched:
                        best          = matched
                        llm_validated = True

        # -- Step 11: final alignment output --------------------------------
        confidence = round(_softmax_confidence([c["composite"] for c in candidates]), 4)

        # Look up CO parent for the winning ILO (populated by map_ilos_to_cos)
        _tagged_lilo = lilo_by_id.get(best["lilo_id"], {})

        # -- Phase 3 Item 2: Confidence label --------------------------------
        _conf_label = (
            "High"    if confidence >= 0.70 else
            "Medium"  if confidence >= 0.50 else
            "Low"     if confidence >= 0.30 else
            "Very Low"
        )

        # -- Phase 3 Item 4: Domain concept validation -----------------------
        _key_concepts   = _tagged_lilo.get("key_concepts") or []
        _q_normalized   = set(_normalize_text(q_text).split())
        _concept_norm   = set()
        for _kc in _key_concepts:
            _concept_norm.update(_normalize_text(_kc).split())
        _concept_overlap = (
            round(_jaccard_score(_q_normalized, _concept_norm), 4)
            if _concept_norm else None
        )

        # -- Phase 3 Item 8: False alignment detection -----------------------
        _false_align = bool(
            best["composite"] < 0.30
            or (
                question_bloom != best["lilo_bloom"]
                and _concept_overlap is not None
                and _concept_overlap == 0.0
            )
        )

        # -- Phase 3 Item 3: Human-readable alignment rationale -------------
        _shared = sorted(
            kc for kc in _key_concepts
            if any(w in _q_normalized for w in _normalize_text(kc).split())
        )[:4]
        _bloom_note = (
            f"same cognitive level ({question_bloom})"
            if question_bloom == best["lilo_bloom"]
            else f"cognitive mismatch — question is {question_bloom}, ILO expects {best['lilo_bloom']}"
        )
        _co_note = f" → CO: {_tagged_lilo.get('co_id')}" if _tagged_lilo.get("co_id") else ""
        _alignment_rationale = (
            f"Matched to {best['lilo_id']}{_co_note} with "
            f"{_conf_label.lower()} confidence ({confidence:.0%}, composite={best['composite']:.3f}). "
            + (f"Shared concepts: {', '.join(_shared)}. " if _shared else "No key concept overlap. ")
            + f"Bloom: {_bloom_note}."
            + (" ⚠ Low-confidence — review recommended." if _false_align else "")
        )

        # -- Phase 3 Item 9: Secondary LILO (multi-ILO support) -------------
        _secondary_lilo = None
        if len(top_candidates) >= 2 and not _source_override_lilo:
            _sec = top_candidates[1]
            _diff = best["composite"] - _sec["composite"]
            if _diff < 0.10 or (best["composite"] > 0 and _diff / best["composite"] < 0.15):
                _sec_obj = lilo_by_id.get(_sec["lilo_id"], {})
                _secondary_lilo = {
                    "lilo_id":   _sec["lilo_id"],
                    "lilo_text": _sec["lilo_text"],
                    "co_id":     _sec_obj.get("co_id"),
                    "ga_codes":  _sec_obj.get("ga_codes") or [],
                    "po_codes":  _sec_obj.get("po_codes") or [],
                    "score":     round(_sec["composite"], 4),
                }

        # -- Generation drift, kept visible ---------------------------------
        # Populated only when the composite's own winner is NOT the outcome the
        # question was generated for. The source still wins the tag — provenance
        # is authoritative and attainment is computed from it — but a question
        # that reads as a different outcome than the one it was written for is a
        # signal about the GENERATION, and silently discarding it is how the
        # override turns from a correctness fix into a way to stop measuring.
        # None when they agree, and absent entirely on records saved before this
        # field existed, so "no drift" and "never measured" stay distinguishable.
        _semantic_drift = None
        if _semantic_best["lilo_id"] != best["lilo_id"]:
            _semantic_drift = {
                "lilo_id":   _semantic_best["lilo_id"],
                "lilo_text": _semantic_best["lilo_text"],
                "score":     round(_semantic_best["composite"], 4),
                "margin":    round(_semantic_best["composite"] - best["composite"], 4),
            }

        q["lilo_tag"] = {
            "lilo_id":        best["lilo_id"],
            "lilo_text":      best["lilo_text"],
            "co_id":         _tagged_lilo.get("co_id"),
            "co_text":       _tagged_lilo.get("co_text"),
            "ga_codes":      _tagged_lilo.get("ga_codes") or [],
            "po_codes":      _tagged_lilo.get("po_codes") or [],
            "score":         round(best["composite"], 4),
            "confidence":    confidence,
            "confidence_label": _conf_label,
            "bloom_level":   question_bloom,              # what the question demands
            "lilo_bloom":     best["lilo_bloom"],         # what the LILO targets
            "bloom_aligned": question_bloom == best["lilo_bloom"],
            "bloom_gap":     _bloom_gap(question_bloom, best["lilo_bloom"]),
            "format_ceiling":      q_ceiling,        # highest level this FORMAT can carry
            "bloom_above_ceiling": q_above_ceiling,  # detected level exceeds that ceiling
            "concept_overlap_score": _concept_overlap,
            "false_alignment_flag":  _false_align,
            "alignment_rationale":   _alignment_rationale,
            "secondary_lilo":        _secondary_lilo,
            # The composite's own pick when it disagrees with the tag above.
            # Diagnostic only — nothing scores or attains off this.
            "semantic_best":         _semantic_drift,
            "score_breakdown": {
                "cosine":  round(best["cosine"],  4),
                "jaccard": round(best["jaccard"], 4),
                "content": round(best["content_match"], 4),
                "bloom":   best["bloom_match"],
            },
            # Which channels actually had data. Read this, not the value, when
            # deciding whether a channel's weight belongs in a normaliser's
            # denominator. A zero with available=True is a measured non-match
            # and must count against the score.
            "score_available": {
                "cosine":  True,          # the embedding always runs on this path
                "jaccard": best.get("jaccard_available", True),
                "content": best.get("content_available", True),
                "bloom":   best.get("bloom_available", True),
            },
            "content_alignment": best_content_alignment,
            "llm_validated": llm_validated,
        }

    matched       = sum(1 for q in questions if q["lilo_tag"]["lilo_id"] != "LILO-NONE")
    bloom_aligned = sum(1 for q in questions if q["lilo_tag"].get("bloom_aligned"))
    logger.info(
        f"Question tagging done. "
        f"LILO-matched: {matched}/{len(questions)}, "
        f"Bloom-aligned: {bloom_aligned}/{len(questions)}, "
        f"LLM-calls: {llm_calls_used}"
    )
    return questions


# ---------------------------------------------------------------------------
# Evaluation: confusion matrix + per-lilo metrics (requires scikit-learn)
# ---------------------------------------------------------------------------

def measure_alignment_accuracy(
    predictions: List[Dict],
    ground_truth: List[Dict],
) -> Dict:
    """
    Compare predicted lilo tags against manually labelled ground truth.

    Metrics:
      - Overall accuracy
      - Cohen's Kappa  (> 0.80 almost perfect  |  0.61-0.80 substantial)
      - Confusion matrix
      - Per-lilo Precision / Recall / F1

    Args:
        predictions:  List of dicts containing "best_lilo_id" or
                      nested "lilo_tag.lilo_id" (output of this module)
        ground_truth: Same structure with human-assigned lilo labels
    """
    if len(predictions) != len(ground_truth):
        raise ValueError("predictions and ground_truth must have the same length")

    pred_ids = [
        p.get("best_lilo_id") or (p.get("lilo_tag") or {}).get("lilo_id") or "LILO-NONE"
        for p in predictions
    ]
    gt_ids = [
        g.get("best_lilo_id") or (g.get("lilo_tag") or {}).get("lilo_id") or "LILO-NONE"
        for g in ground_truth
    ]

    total   = len(pred_ids)
    correct = sum(p == g for p, g in zip(pred_ids, gt_ids))

    try:
        from sklearn.metrics import (
            confusion_matrix, classification_report, cohen_kappa_score,
        )
        labels            = sorted(set(gt_ids))
        cm                = confusion_matrix(gt_ids, pred_ids, labels=labels)
        report            = classification_report(
                                gt_ids, pred_ids, labels=labels,
                                output_dict=True, zero_division=0,
                            )
        kappa             = cohen_kappa_score(gt_ids, pred_ids)
        sklearn_available = True
    except ImportError:
        logger.warning("scikit-learn not installed -- skipping confusion matrix and kappa.")
        labels            = sorted(set(gt_ids))
        cm                = None
        report            = {}
        kappa             = None
        sklearn_available = False

    per_lilo: Dict[str, Dict] = {}
    for pred_id, gt_id in zip(pred_ids, gt_ids):
        if gt_id not in per_lilo:
            per_lilo[gt_id] = {"correct": 0, "total": 0}
        per_lilo[gt_id]["total"] += 1
        if pred_id == gt_id:
            per_lilo[gt_id]["correct"] += 1

    none_count = pred_ids.count("LILO-NONE")
    return {
        "total":             total,
        "correct":           correct,
        "accuracy":          round(correct / total, 4) if total else 0.0,
        "cohen_kappa":       round(kappa, 4) if kappa is not None else None,
        "confusion_matrix":  cm.tolist() if cm is not None else None,
        "labels":            labels,
        "per_lilo_report":    report,
        "per_lilo":           per_lilo,
        "none_rate":         round(none_count / total, 4) if total else 0.0,
        "sklearn_available": sklearn_available,
    }
