# Data layout

- `evaluation/chem31a_human.jsonl`: 103 real university exam questions with one or more course-staff LILO labels. Evaluation only.
- `synthetic/synthetic_weak_train.jsonl`: deprecated first experiment; 9,000 easy template pairs.
- `mixed_weak/prime_generated_*.jsonl`: unique, real generated PRIME question text paired with generation targets and hard negatives.
- `mixed_weak/openstax_structural_*.jsonl`: real OpenStax questions with subchapter-level objective groups and hard negatives.
- `mixed_weak/mixed_*.jsonl`: both weak sources combined, split by assessment/chapter so a question group cannot cross train/validation.
- `prime_reviewed/`: put de-identified PRIME professor-review exports here. This is the training data that can eventually support ISU-CS claims.

Prefer `mixed_weak` over the template corpus. The PRIME labels are selected generation targets, not professor judgments. The OpenStax labels mean correct published subchapter, not exact individual objective. Neither may be used as final accuracy evidence.

CHEM 31A is deliberately excluded from training and remains in `evaluation/`.
