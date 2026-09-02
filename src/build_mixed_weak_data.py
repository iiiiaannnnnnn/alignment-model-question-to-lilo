"""Build provenance-labelled weak training pairs from real question text.

Inputs are existing PRIME generated assessments and the OpenStax structural
benchmark. Neither source is independent per-question human ground truth.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRIME_DIRS = [
    Path(r"C:\Users\Jericho Mico\Desktop\Thesis\quiz_exam_withalignement"),
    Path(r"C:\Users\Jericho Mico\Desktop\Thesis\Q_Testing_Models\Syllabus and FIle\jsons"),
]
WORKSPACE_OPENSTAX_ROWS = Path(
    r"C:\Users\Jericho Mico\Desktop\Thesis\Q_Testing_Models\Syllabus and FIle\test\results\exp5_rows_chemistry2e.csv"
)
WORKSPACE_OPENSTAX_MAP = Path(
    r"C:\Users\Jericho Mico\Desktop\Thesis\Q_Testing_Models\Syllabus and FIle\test\datasets\smartstem\chemistry2e_subchapter_to_learning_goal.json"
)


def tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def similarity(a: str, b: str) -> float:
    left, right = tokens(a), tokens(b)
    return len(left & right) / max(len(left | right), 1)


def split_for(group: str) -> str:
    return "validation" if int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 5 == 0 else "train"


def pair(question_id: str, question: str, candidate: dict, label: int, source: str,
         pair_type: str, group_id: str) -> dict:
    return {
        "question_id": question_id,
        "question": question.strip(),
        "lilo_id": str(candidate["id"]),
        "lilo": candidate["text"].strip(),
        "label": label,
        "topic": candidate.get("topic") or candidate.get("week") or candidate.get("topic_title") or "",
        "source": source,
        "pair_type": pair_type,
        "group_id": group_id,
        "label_quality": "weak",
    }


def prime_rows(directories: list[Path], seed: int) -> tuple[list[dict], list[dict], dict]:
    rng = random.Random(seed)
    train, validation = [], []
    files = []
    for directory in directories:
        files.extend(directory.glob("alignment_*.json"))
    files = sorted(set(files))
    questions = skipped = 0
    for path in files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        candidates = {
            str(item.get("id")): {
                "id": item.get("id"),
                "text": item.get("text") or item.get("enriched_text") or "",
                "topic": item.get("topic_title") or item.get("learning_content") or item.get("week") or "",
            }
            for item in data.get("enrichedLlos", [])
            if item.get("id") and (item.get("text") or item.get("enriched_text"))
        }
        group = path.stem
        destination = validation if split_for(group) == "validation" else train
        for index, question in enumerate(data.get("questions", []), 1):
            target_id = str(question.get("source_ilo_id") or "")
            target = candidates.get(target_id)
            text = str(question.get("question") or "").strip()
            if not target or not text:
                skipped += 1
                continue
            alternatives = [item for key, item in candidates.items() if key != target_id]
            if not alternatives:
                skipped += 1
                continue
            qid = f"prime-{path.stem}-{index}"
            destination.append(pair(qid, text, target, 1, "prime_generation_target_weak",
                                    "generation_target", group))
            ranked = sorted(alternatives, key=lambda item: similarity(text, item["text"]), reverse=True)
            destination.append(pair(qid, text, ranked[0], 0, "prime_generation_target_weak",
                                    "lexical_hard_negative", group))
            remaining = ranked[1:]
            for negative in rng.sample(remaining, k=min(2, len(remaining))):
                destination.append(pair(qid, text, negative, 0, "prime_generation_target_weak",
                                        "assessment_pool_negative", group))
            questions += 1
    return train, validation, {"files": len(files), "questions": questions, "skipped": skipped}


def openstax_rows(rows_path: Path, map_path: Path, seed: int) -> tuple[list[dict], list[dict], dict]:
    import csv

    rng = random.Random(seed)
    mapping = json.loads(map_path.read_text(encoding="utf-8"))
    objectives = []
    by_subchapter = {}
    for heading, texts in mapping.items():
        subchapter = heading.split()[0]
        chapter = subchapter.split(".")[0]
        by_subchapter[subchapter] = []
        for index, text in enumerate(texts, 1):
            item = {"id": f"OPENSTAX-{subchapter}-{index}", "text": text, "topic": heading,
                    "subchapter": subchapter, "chapter": chapter}
            objectives.append(item)
            by_subchapter[subchapter].append(item)

    train, validation = [], []
    questions = 0
    with rows_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            question = row["question"].strip()
            subchapter = row["gold_subchapter"]
            positives = by_subchapter.get(subchapter, [])
            if not question or not positives:
                continue
            chapter = subchapter.split(".")[0]
            same_chapter = [item for item in objectives if item["chapter"] == chapter and item["subchapter"] != subchapter]
            other_chapter = [item for item in objectives if item["chapter"] != chapter]
            group = f"openstax-chapter-{chapter}"
            destination = validation if split_for(group) == "validation" else train
            for positive in positives:
                destination.append(pair(row["qid"], question, positive, 1,
                                        "openstax_subchapter_structure_weak", "subchapter_objective", group))
                if same_chapter:
                    hard = max(same_chapter, key=lambda item: similarity(question, item["text"]))
                    destination.append(pair(row["qid"], question, hard, 0,
                                            "openstax_subchapter_structure_weak", "sibling_subchapter_negative", group))
                if other_chapter:
                    destination.append(pair(row["qid"], question, rng.choice(other_chapter), 0,
                                            "openstax_subchapter_structure_weak", "other_chapter_negative", group))
            questions += 1
    return train, validation, {"questions": questions, "objectives": len(objectives)}


def write(path: Path, rows: list[dict], seed: int) -> None:
    random.Random(seed).shuffle(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prime-dir", action="append", type=Path)
    parser.add_argument("--openstax-rows", type=Path, default=WORKSPACE_OPENSTAX_ROWS)
    parser.add_argument("--openstax-map", type=Path, default=WORKSPACE_OPENSTAX_MAP)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "mixed_weak")
    parser.add_argument("--seed", type=int, default=20260829)
    args = parser.parse_args()

    prime_train, prime_val, prime_stats = prime_rows(args.prime_dir or DEFAULT_PRIME_DIRS, args.seed)
    open_train, open_val, open_stats = openstax_rows(args.openstax_rows, args.openstax_map, args.seed)
    write(args.output_dir / "prime_generated_train.jsonl", prime_train, args.seed)
    write(args.output_dir / "prime_generated_validation.jsonl", prime_val, args.seed)
    write(args.output_dir / "openstax_structural_train.jsonl", open_train, args.seed)
    write(args.output_dir / "openstax_structural_validation.jsonl", open_val, args.seed)
    write(args.output_dir / "mixed_train.jsonl", prime_train + open_train, args.seed)
    write(args.output_dir / "mixed_validation.jsonl", prime_val + open_val, args.seed)
    manifest = {
        "prime": prime_stats,
        "openstax": open_stats,
        "rows": {
            "prime_train": len(prime_train), "prime_validation": len(prime_val),
            "openstax_train": len(open_train), "openstax_validation": len(open_val),
            "mixed_train": len(prime_train) + len(open_train),
            "mixed_validation": len(prime_val) + len(open_val),
        },
        "warnings": [
            "PRIME source_ilo_id is generation provenance, not independent human annotation.",
            "OpenStax labels are subchapter objective groups, not exact per-question LILO annotations.",
            "CHEM 31A human labels are excluded from all training output.",
        ],
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
