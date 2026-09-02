"""Evaluate a cross-encoder on grouped question-LILO pair rows."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from src.train_reranker import load_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    rows = load_rows(args.data)
    if any("question_id" not in row or "lilo_id" not in row for row in rows):
        raise ValueError("Grouped evaluation rows need question_id and lilo_id")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    scores = []
    with torch.no_grad():
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start:start + args.batch_size]
            encoded = tokenizer(
                [row["question"] for row in batch], [row["lilo"] for row in batch],
                padding=True, truncation=True, max_length=args.max_length, return_tensors="pt",
            )
            logits = model(**{key: value.to(device) for key, value in encoded.items()}).logits.squeeze(-1)
            scores.extend(logits.cpu().tolist())

    groups = defaultdict(list)
    for row, score in zip(rows, scores):
        groups[row["question_id"]].append((score, row))
    details = []
    reciprocal_ranks = []
    hits = {1: 0, 3: 0, 5: 0}
    for question_id, candidates in groups.items():
        ranked = sorted(candidates, key=lambda item: item[0], reverse=True)
        rank = next(index for index, (_, row) in enumerate(ranked, 1) if row["label"] == 1)
        reciprocal_ranks.append(1.0 / rank)
        for k in hits:
            hits[k] += rank <= k
        details.append({
            "question_id": question_id,
            "positive_rank": rank,
            "ranked": [{"lilo_id": row["lilo_id"], "label": row["label"], "score": score}
                       for score, row in ranked],
        })
    n = len(groups)
    result = {
        "questions": n,
        "pairs": len(rows),
        "mrr": round(sum(reciprocal_ranks) / n, 4),
        "recall_at_1": round(hits[1] / n, 4),
        "recall_at_3": round(hits[3] / n, 4),
        "recall_at_5": round(hits[5] / n, 4),
        "model": args.model,
        "details": details,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "details"}, indent=2))


if __name__ == "__main__":
    main()
