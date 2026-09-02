# PRIME alignment-model research

This folder is a training workspace. It does not change the PRIME application.

## What is included

- `baseline/app/utils/alignment_model.py`: exact snapshot of today's deployed module.
- `candidate/app/utils/alignment_model.py`: initially the same snapshot, so it has the same public API and can later be promoted safely.
- `data/evaluation/chem31a_human.jsonl`: 103 human-labelled Stanford CHEM 31A questions. This is evaluation data, never training input.
- `data/mixed_weak/`: diverse weak-supervision pairs from unique PRIME-generated questions and real OpenStax questions. Labels retain their provenance and limitations.
- `data/synthetic/synthetic_weak_train.jsonl`: deprecated easy-template experiment retained only for reproducibility.
- `src/train_reranker.py`: fine-tunes a six-layer cross-encoder only after enough reviewed data exists.

## Important data rule

Do not report accuracy on a dataset used in training. In particular, do not train on `data/evaluation/chem31a_human.jsonl` or use it to choose epochs or thresholds.

The included synthetic data is templated from LILO statements. It is not human-labelled and must not be used to claim educational alignment accuracy. Replace or supplement it with de-identified professor-reviewed PRIME question-to-LILO decisions.

## Google Colab quick start

1. Zip this `alignment_model` folder and upload it to Google Drive, then extract it in `MyDrive`.
2. In Colab, enable a GPU: `Runtime` -> `Change runtime type` -> `T4 GPU`.
3. Run:

```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/alignment_model
!pip install -q -r requirements-colab.txt
```

Do not install `requirements-prime-baseline.txt` unless you also upload the PRIME backend dependencies required by the copied baseline module. The included training and evaluation commands run independently of FastAPI and API keys.

4. Confirm the copied modules are identical and inspect the datasets:

```bash
!python -m src.validate_workspace
```

If you uploaded an older copy that does not yet contain `data/mixed_weak`, copy the updated local `alignment_model` folder to Drive again. The mixed JSONL files are already built; Colab does not need access to the Thesis workspace.

5. Score the untrained pretrained reranker on the human-labelled benchmark. This is a comparison run, not training:

```bash
!python -m src.evaluate_reranker \
  --data data/evaluation/chem31a_human.jsonl \
  --model cross-encoder/ms-marco-MiniLM-L6-v2 \
  --output results/pretrained_chem31a.json
```

6. Train the improved weak-data experiment with held-out validation:

```bash
!python -m src.train_reranker \
  --train data/mixed_weak/mixed_train.jsonl \
  --validation data/mixed_weak/mixed_validation.jsonl \
  --output models/prime-lilo-reranker-v2 \
  --epochs 4
```

7. Evaluate the saved model on the separate human-labelled file:

```bash
!python -m src.evaluate_reranker \
  --data data/evaluation/chem31a_human.jsonl \
  --model models/prime-lilo-reranker-v2 \
  --output results/trained_chem31a_v2.json
```

Compare this result with both `results/pretrained_chem31a.json` and PRIME's all-question baseline of `31/103 = 30.10%`. The often quoted `31.31%` excludes four PRIME no-match questions, while this reranker evaluator always scores all 103.

## Add your own PRIME review data

Create one JSON object per line in `data/prime_reviewed/reviewed_train.jsonl`:

```json
{"question_id":"q-001","question":"Explain the role of a fact table in a data warehouse.","lilo_id":"LILO-12","lilo":"Explain fact and dimension tables in a star schema.","label":1,"topic":"Data warehousing","source":"professor_review"}
{"question_id":"q-001","question":"Explain the role of a fact table in a data warehouse.","lilo_id":"LILO-22","lilo":"Apply association-rule metrics to transactions.","label":0,"topic":"Data mining","source":"professor_review"}
```

Use one positive row for every professor-approved LILO. Add hard negative rows from similar but rejected LILOs. Do not include student names, answers, IDs, or grades.

Train with the reviewed data once it is large enough:

```bash
!python -m src.train_reranker \
  --train data/prime_reviewed/reviewed_train.jsonl \
  --output models/prime-lilo-reranker-isu-cs \
  --epochs 3
```

## Model plan

The production module stays responsible for candidate scoping, Bloom diagnostics, no-match handling, generation provenance, and review safeguards. A trained reranker should only break ties among the current top candidates for non-generated questions.

```text
Existing candidate scope and retrieval
-> top 10 LILOs
-> trained cross-encoder reranker
-> existing review/no-match behavior
```

Start with `cross-encoder/ms-marco-MiniLM-L6-v2`: six transformer layers, about 22.7M parameters. Do not build a neural network from scratch or use a larger model until this one is shown to be inadequate on frozen professor-reviewed ISU-CS data.

## Before production replacement

Do not copy a trained model into PRIME merely because training finishes. Promote it only after it improves a held-out, human-reviewed ISU-CS test set and preserves the existing module contract. `candidate/app/utils/alignment_model.py` must retain all current public functions and the `tag_questions_with_lilos()` signature.

See `candidate/README.md` for the safe integration point.

## IT/CS V4 weak-data run

`data/it_cs_v4` contains the training-ready transformation of the 960-question
IT/CS synthetic corpus. It has 120 versioned LILOs across 12 domains, holds out
complete LILO/question families for validation, and creates one positive plus
three negatives per question. These remain weak labels.

Upload the updated `src` and `data/it_cs_v4` folders to the existing Google
Drive `alignment_model` folder. Keep the existing `models` and `results`
folders; a full folder replacement is unnecessary.

Train one clean-base candidate and one continued V3 candidate:

```bash
!python -m src.validate_workspace

!python -m src.train_reranker \
  --train data/it_cs_v4/pairs/train.jsonl \
  --validation data/it_cs_v4/pairs/validation.jsonl \
  --output models/prime-lilo-reranker-v4a-it \
  --base-model cross-encoder/ms-marco-MiniLM-L6-v2 \
  --epochs 6 --batch-size 16 --learning-rate 1e-5 --patience 2

!python -m src.train_reranker \
  --train data/it_cs_v4/pairs/train.jsonl \
  --validation data/it_cs_v4/pairs/validation.jsonl \
  --output models/prime-lilo-reranker-v4b-it-from-v3 \
  --base-model models/prime-lilo-reranker-v3-cs \
  --epochs 6 --batch-size 16 --learning-rate 5e-6 --patience 2
```

Evaluate both on grouped IT validation. Use MRR, then Recall@1, to select the
winner. Do not use CHEM to choose between them.

```bash
!python -m src.evaluate_grouped_reranker \
  --data data/it_cs_v4/pairs/validation.jsonl \
  --model models/prime-lilo-reranker-v4a-it \
  --output results/v4a_it_validation.json

!python -m src.evaluate_grouped_reranker \
  --data data/it_cs_v4/pairs/validation.jsonl \
  --model models/prime-lilo-reranker-v4b-it-from-v3 \
  --output results/v4b_it_validation.json
```
