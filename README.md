# Alignment model: question to LILO

Research workspace for the outcome-alignment component of PRIME, an outcomes-based
education system built for Isabela State University. The task is narrow: given an
assessment question, decide which **LILO** (Lesson Intended Learning Outcome)
it actually measures, so that attainment can be computed per outcome rather than
per test.

This repository holds the code, the datasets, the evaluation harness and every
result. It is a record of what was tried and what the numbers say, including the
results that did not go the way the project wanted them to.

---

## The two systems being compared

**The composite matcher** is what PRIME ships. It scores a question against each
candidate outcome on four channels and takes a weighted sum:

| channel | weight | what it measures |
|---|---|---|
| cosine | 0.45 | sentence-embedding similarity, question vs. enriched outcome |
| bloom | 0.20 | cognitive-level proximity, question verb vs. outcome verb |
| jaccard | 0.20 | keyword overlap, question vs. outcome key concepts |
| content | 0.15 | keyword overlap, question vs. that week's learning content |

**The reranker** is the candidate replacement: `cross-encoder/ms-marco-MiniLM-L6-v2`
(6 layers, ~22.7M parameters) fine-tuned to rerank the top-10 candidates the
composite matcher proposes. Seven training runs are recorded here, v2 through v5.

---

## Findings

### 1. The shipping matcher agrees with authoring provenance 74% of the time

Measured on 197 questions across seven real assessment exports, with the
`source_ilo_id` provenance field removed so the four channels have to find the
outcome on their own:

| export | lectures | n | pool | top-1 agreement | chance | lift |
|---|---|---|---|---|---|---|
| 01 Quiz - Real-World Data Formulation | 1 | 10 | 4 | 90.00% | 25.00% | 3.6x |
| 02 Quiz - Data Formulation and Item Sets | 1-2 | 20 | 8 | 80.00% | 12.50% | 6.4x |
| 03 Quiz - Mining Item Sets | 2 | 10 | 4 | 50.00% | 25.00% | 2.0x |
| 04 Quiz - Text and Image Mining | 4-5 | 20 | 7 | 75.00% | 14.29% | 5.2x |
| 05 Quiz - Text and Image Mining (40 items) | 4-5 | 40 | 7 | 65.00% | 14.29% | 4.6x |
| 06 Quiz - Big Data and Behavior Mining | 8-9 | 35 | 5 | 91.43% | 20.00% | 4.6x |
| 07 Final Exam - Full Course | 1-9 | 62 | 31 | 69.35% | 3.23% | 21.5x |
| **corpus** | | **197** | | **74.11%** | | |

The export names come from each file's own `topic` field, which records the
lectures it draws on; all seven are from the same Data Mining course, which is
why the outcome pools overlap. `jsons/EXPORTS.md` maps each one back to its
original filename and to the `group_id` the datasets and results still use.

The lift column is the honest way to read this. A 91% agreement against a
five-outcome pool is a weaker result than a 69% agreement against a
thirty-one-outcome pool, and only the lift makes that visible.

**This is agreement with provenance, not correctness.** The label is "the outcome
this question was generated for", which is weak supervision. Nothing here
establishes that either the matcher or the original generation was educationally
right.

### 2. Two of the four channels do almost nothing

| channel | weighted contribution | share of composite |
|---|---|---|
| cosine | 0.2460 | 61.34% |
| bloom | 0.1464 | 36.52% |
| jaccard | 0.0115 | 2.87% |
| content | 0.0081 | 2.01% |

Jaccard and content together carry under 5% of the score while holding 35% of the
weight. They score near zero because they are raw keyword overlaps between a short
question and a short outcome statement: median jaccard 0.0566, median content
0.0465, and they return a hard zero on 12 and 14 of the 197 questions
respectively. Bloom does the opposite — median 0.75, minimum 0.25, so it almost
never discriminates downward and mostly acts as a constant offset.

In practice the ranking is close to cosine similarity with a bloom tiebreak.

### 3. Tagged is not the same as covered

Every question receives an outcome, so a per-question completion count always
reads 100%. That is not the same as every selected outcome receiving a question.

Across the corpus, if the composite matcher alone decided the tagging, **4 of 66
selected outcomes would receive no question at all** — absorbed by a semantic
neighbour while the question count still looked complete. An outcome with no
question reports zero attainment permanently, and nothing in the old summary said
so. This is what the source-outcome invariant in the shipping system prevents.

### 4. A quarter of generated questions drift off their own target

50 of 197 questions (25.38%) score highest against an outcome other than the one
they were generated for. But the margin distribution matters: **38% of those
drifts have a margin below 0.02**, which for a composite whose mean is 0.4010 is a
near-tie — the choice between the two outcomes is close to arbitrary. Median drift
margin 0.0283, max 0.3273. Only the large-margin drifts are evidence of a real
authoring problem.

### 5. Fine-tuning helped less than the validation numbers claimed

Two benchmarks, and they disagree sharply.

**The grouped 4-candidate validation** (192 held-out IT questions, 4 candidates
each) is the setup the models were trained and selected on:

| model | top-1 | MRR |
|---|---|---|
| v4a_it | **0.8802** | 0.9375 |
| v4b_it_from_v3 | 0.8698 | 0.9297 |
| pretrained (no fine-tuning) | 0.8281 | 0.9036 |
| v3_cs | 0.8281 | 0.9058 |
| v5_tload_pilot | 0.8281 | 0.9045 |

**The full 120-outcome pool**, same 192 questions, realistic candidate set:

| model | top-1 | MRR |
|---|---|---|
| v3_cs | 0.6719 | 0.7565 |
| v5_tload_pilot | 0.6719 | 0.7566 |
| **pretrained (no fine-tuning)** | **0.6510** | 0.7496 |
| v4b_it_from_v3 | 0.6302 | 0.7306 |
| v4a_it | 0.6198 | 0.7241 |

**v4a_it wins the benchmark it was selected on and loses to the untrained baseline
on the realistic one** — 0.6198 against 0.6510. Training against a 4-candidate
objective taught it to discriminate among near-neighbours, not to rank against 120
outcomes.

v5_tload_pilot is the sharpest case of the same failure. It posts the best
validation MRR in the whole project — 0.9667 on 60 validation pairs — and lands
0.5127 on the real-export benchmark, among the worst of the seven. Sixty pairs
cannot select a model.

### 6. Cross-domain transfer separates the models more than in-domain accuracy does

Held-out Stanford CHEM 31A, 103 human-labelled questions, 75-outcome pool. No
model was trained on chemistry:

| model | top-1 | MRR |
|---|---|---|
| v3_cs | **0.3495** | 0.4840 |
| v4b_it_from_v3 | 0.3010 | 0.4427 |
| v4a_it | 0.1942 | 0.3309 |
| pretrained | 0.1650 | 0.2744 |
| v5_tload_pilot | 0.1650 | 0.2911 |

v3_cs more than doubles the untrained baseline on a domain it never saw. v4a_it
barely moves it. The difference in their training data is that v3_cs learned from
305 real generated questions and v4a_it learned from 768 synthetic
template-instantiated ones. **The synthetic data taught a template, the real data
taught a task** — and CHEM is the only benchmark here uncontaminated enough to
show it. See finding 7.

Note that v4b (v3_cs weights, further trained on IT data) *loses* chemistry
ability relative to v3_cs: 0.3010 down from 0.3495. Continued fine-tuning on the
narrower domain partially undid the transfer.

### 7. Benchmark contamination reaches three models, not one

The project caught contamination in `v4a_it_extended_v1` and rebuilt it as
`extended_clean_v2`, which is good practice and the manifests say so. **The audit
in this repository found the same leak in two further models, undocumented.**

Normalising question text (`[^a-z0-9]+` collapsed to spaces, lowercased) and
intersecting each training file against the 197 benchmark questions:

| training file | rows | unique questions | benchmark questions leaked |
|---|---|---|---|
| `data/mixed_weak/mixed_train.jsonl` | 12839 | 1653 | **135** |
| `data/mixed_weak/prime_generated_train.jsonl` | 1205 | 305 | **135** |
| `v4a_it_extended/train.jsonl` | 7637 | 1145 | **135** |
| `data/it_cs_v4/pairs/train.jsonl` | 3072 | 768 | 0 |
| `data/prime_tload_revised/pairs/train.jsonl` | 288 | 72 | 0 |
| `data/synthetic/synthetic_weak_train.jsonl` | 9000 | 300 | 0 |
| `v4a_it_extended_clean/train.jsonl` | 7097 | 1010 | 0 |

Matching row counts against each `training_manifest.json` identifies which model
consumed which file: v2 records `training_rows: 12839`, v3_cs records `1205`,
extended_v1 records `7637`. So **v2, v3_cs and v4a_it_extended_v1 were all trained
on 135 of the 197 benchmark questions** — 68% of the benchmark. And
v4b_it_from_v3 starts from v3_cs's weights, so it inherits the leak transitively.

That reframes the headline table. On the union-31 real-export benchmark:

| model | top-1 | status |
|---|---|---|
| v4a_it_extended_v1 | 0.9086 | contaminated (documented in the manifests) |
| v3_cs | 0.8629 | **contaminated (found by this audit)** |
| v4b_it_from_v3 | 0.8528 | **contaminated via v3_cs weights (found by this audit)** |
| v4a_it_extended_clean_v2 | **0.7817** | clean — the best trustworthy result |
| v4a_it | 0.6244 | clean |
| v5_tload_pilot | 0.5127 | clean |
| pretrained | 0.4569 | clean baseline |

Removing the leak cost extended_v1 to clean_v2 **12.7 points** (0.9086 to 0.7817),
which is a direct measurement of how much of that score was memorisation. The same
correction should be assumed to apply to v3_cs and v4b, whose remaining advantage
over clean_v2 is not evidence of anything.

**The honest headline is 0.7817**, from `v4a_it_extended_clean_v2`, against a
0.4569 untrained baseline.

Two things survive the contamination check intact and can still be quoted:

- The **held-out IT validation is clean** — zero benchmark-question overlap in
  every training file — so the 120-outcome table in finding 5, including v4a_it
  losing to the baseline, stands.
- The **CHEM benchmark is clean for every model**, since chemistry appears in no
  training file. Finding 6 stands, and it is where v3_cs's advantage is real.

### 8. What that means for the shipping system

No reranker in this repository is a safe promotion. The best clean one improves
top-1 agreement on real exports from 0.4569 to 0.7817, but it is trained against
weak provenance labels, it is selected on a benchmark of 197 questions from seven
exports authored by one system, and the one model that transfers across domains
does so with a score no reader should trust. The composite matcher's 74.11%
agreement — also against weak labels — remains the shipping behaviour.

The measurement that would settle this does not exist yet: professor-reviewed
question-to-outcome decisions, made by people who did not see the model's answer.

---

## What is not in this repository, and why

**Trained weights.** Each `models/<name>/` keeps its `config.json`, tokenizer and
`training_manifest.json` — enough to audit and reproduce a run — but not the
86.7 MB `model.safetensors` that goes with it, nor the frozen release zip. Seven
checkpoints would be roughly 600 MB of binary in a repository whose point is the
method and the numbers.

**Stanford CHEM 31A question text and its human labels.** From the SmartSTEM
repository, which carries no licence; `DATA_SOURCES.md` records the constraint as
*"Keep this for internal academic evaluation; do not redistribute publicly without
permission."* The eight files carrying question text or course-staff labels are
excluded. The CHEM *metrics* in `results/` are published in full — they contain
only question ids, gold outcome ids and ranked outcome ids, no text — so finding 6
is fully checkable.

OpenStax Chemistry 2e content is CC BY 4.0 and attributed in `DATA_SOURCES.md`.
No student records, identifiers or personal data appear anywhere in this
repository.

---

## Layout

```
src/                     training, evaluation and dataset-construction scripts
  train_reranker.py            fine-tunes the cross-encoder
  evaluate_reranker.py         full-pool evaluation
  evaluate_grouped_reranker.py grouped n-candidate evaluation
  build_*_data.py              dataset construction, one per source
  validate_workspace.py        integrity checks

baseline/app/utils/      snapshot of the deployed alignment module
candidate/app/utils/     the module under modification

data/
  it_cs_v4/              IT/CS pairs, held out by outcome family
  mixed_weak/            PRIME-generated + OpenStax weak-supervision pairs
  synthetic/             deprecated template experiment, kept for reproducibility
  prime_tload_revised/   the t-load pilot set
v4a_it_extended/         extended IT set (contaminated — kept as the record)
v4a_it_extended_clean/   the same set with benchmark questions removed

jsons/                   the 7 real assessment exports, EXPORTS.md, the evaluation report
models/<name>/           config, tokenizer and training manifest per run
results/                 every metric reported above
```

`jsons/ALIGNMENT_MODEL_EVALUATION.txt` is the long-form evaluation report and the
primary source for findings 1-4. `WORKSPACE.md` is the operating manual: Colab
setup, and the exact commands for each training and evaluation run.

---

## Reproducing the contamination audit

```python
import glob, json, re

norm = lambda s: re.sub(r'[^a-z0-9]+', ' ', (s or '').lower()).strip()

def train_questions(path):
    with open(path, encoding='utf-8') as fh:
        return {norm(json.loads(l)['question']) for l in fh if l.strip()}

bench = set()
for path in glob.glob('jsons/export_*.json'):
    with open(path, encoding='utf-8') as fh:
        for q in json.load(fh)['questions']:
            bench.add(norm(q['question']))

print(len(bench))                                                    # 197
print(len(bench & train_questions('data/mixed_weak/prime_generated_train.jsonl')))  # 135
```

Then read `training_rows` in each `models/*/training_manifest.json` to see which
model consumed which file.

---

## Status

Research code. It does not run inside the PRIME application and changes nothing in
it. Everything here is one thesis project's evaluation of one component: the
sample is small, the labels are weak, and the findings are reported with those
limits attached rather than around them.
