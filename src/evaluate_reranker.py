"""Rank every LILO for a held-out human-labelled question set."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.build_synthetic_data import DEFAULT_SOURCE, load_lilos


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    rows = [json.loads(line) for line in args.data.read_text(encoding="utf-8").splitlines() if line.strip()]
    lilos = load_lilos(DEFAULT_SOURCE)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    top1 = top3 = top5 = 0
    details = []
    with torch.no_grad():
        for row in rows:
            scores = []
            for start in range(0, len(lilos), args.batch_size):
                candidates = lilos[start:start + args.batch_size]
                inputs = tokenizer([row["question"]] * len(candidates), [c["lilo"] for c in candidates],
                                   padding=True, truncation=True, max_length=256, return_tensors="pt")
                logits = model(**{key: value.to(device) for key, value in inputs.items()}).logits.squeeze(-1)
                scores.extend(logits.cpu().tolist())
            ranked = [lilos[i]["id"] for i in sorted(range(len(lilos)), key=lambda i: scores[i], reverse=True)]
            gold = set(row["gold_lilo_ids"])
            top1 += ranked[0] in gold
            top3 += bool(gold & set(ranked[:3]))
            top5 += bool(gold & set(ranked[:5]))
            details.append({"question_id": row["question_id"], "gold_lilo_ids": sorted(gold), "ranked_lilo_ids": ranked[:5]})
    n = len(rows)
    result = {"questions": n, "recall_at_1": round(top1 / n, 4), "recall_at_3": round(top3 / n, 4),
              "recall_at_5": round(top5 / n, 4), "model": args.model, "details": details}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "details"}, indent=2))


if __name__ == "__main__":
    main()
