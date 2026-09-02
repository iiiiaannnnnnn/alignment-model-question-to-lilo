"""Fine-tune a compact cross-encoder on labelled question-LILO pairs."""
from __future__ import annotations

import argparse
import copy
import json
import random
from collections import defaultdict
from pathlib import Path


def load_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or any({"question", "lilo", "label"} - row.keys() for row in rows):
        raise ValueError("Training JSONL needs question, lilo, and label fields")
    if any(row["label"] not in (0, 1) for row in rows):
        raise ValueError("Training labels must be 0 or 1")
    if {row["label"] for row in rows} != {0, 1}:
        raise ValueError("Training JSONL must contain both positive and negative pairs")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-model", default="cross-encoder/ms-marco-MiniLM-L6-v2")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260830)
    args = parser.parse_args()

    rows = load_rows(args.train)
    validation_rows = load_rows(args.validation) if args.validation else []
    if validation_rows and any("question_id" not in row for row in validation_rows):
        raise ValueError("Validation rows need question_id for grouped ranking metrics")

    import torch
    from torch.optim import AdamW
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(args.base_model, num_labels=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    class Pairs(Dataset):
        def __init__(self, examples):
            self.examples = examples
        def __len__(self):
            return len(self.examples)
        def __getitem__(self, index):
            row = self.examples[index]
            encoded = tokenizer(row["question"], row["lilo"], truncation=True, max_length=args.max_length)
            encoded["labels"] = float(row["label"])
            return encoded, str(row.get("question_id", index))

    def collate(batch):
        encoded, question_ids = zip(*batch)
        labels = torch.tensor([item.pop("labels") for item in encoded], dtype=torch.float32)
        return tokenizer.pad(encoded, padding=True, return_tensors="pt"), labels, question_ids

    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(Pairs(rows), batch_size=args.batch_size, shuffle=True,
                        collate_fn=collate, generator=generator)
    validation_loader = DataLoader(Pairs(validation_rows), batch_size=args.batch_size,
                                   shuffle=False, collate_fn=collate) if validation_rows else None
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    positives = sum(row["label"] for row in rows)
    weight = torch.tensor([(len(rows) - positives) / max(positives, 1)], device=device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=weight)

    best_mrr = -1.0
    best_recall_at_1 = 0.0
    best_loss = float("inf")
    best_epoch = None
    best_state = None
    stale_epochs = 0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for encoded, labels, _ in loader:
            encoded = {key: value.to(device) for key, value in encoded.items()}
            logits = model(**encoded).logits.squeeze(-1)
            loss = loss_fn(logits, labels.to(device))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        train_loss = total_loss / len(loader)
        if validation_loader:
            model.eval()
            validation_loss = 0.0
            grouped = defaultdict(list)
            with torch.no_grad():
                for encoded, labels, question_ids in validation_loader:
                    encoded = {key: value.to(device) for key, value in encoded.items()}
                    logits = model(**encoded).logits.squeeze(-1)
                    validation_loss += loss_fn(logits, labels.to(device)).item()
                    for question_id, score, label in zip(question_ids, logits.cpu().tolist(), labels.tolist()):
                        grouped[question_id].append((score, int(label)))
            validation_loss /= len(validation_loader)
            reciprocal_ranks = []
            for candidates in grouped.values():
                ranked = sorted(candidates, key=lambda item: item[0], reverse=True)
                rank = next((index for index, (_, label) in enumerate(ranked, 1) if label), None)
                reciprocal_ranks.append(0.0 if rank is None else 1.0 / rank)
            mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
            recall_at_1 = sum(value == 1.0 for value in reciprocal_ranks) / len(reciprocal_ranks)
            print(f"epoch {epoch + 1}/{args.epochs}: train_loss={train_loss:.4f} "
                  f"validation_loss={validation_loss:.4f} validation_mrr={mrr:.4f} "
                  f"validation_recall_at_1={recall_at_1:.4f}")
            if mrr > best_mrr or (mrr == best_mrr and validation_loss < best_loss):
                best_mrr = mrr
                best_recall_at_1 = recall_at_1
                best_loss = validation_loss
                best_epoch = epoch + 1
                best_state = copy.deepcopy(model.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= args.patience:
                    print("Early stopping: grouped validation MRR did not improve")
                    break
        else:
            print(f"epoch {epoch + 1}/{args.epochs}: loss={train_loss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    (args.output / "training_manifest.json").write_text(json.dumps({
        "base_model": args.base_model, "training_rows": len(rows),
        "validation_rows": len(validation_rows), "requested_epochs": args.epochs,
        "best_epoch": best_epoch, "seed": args.seed,
        "best_validation_mrr": None if best_mrr < 0 else best_mrr,
        "best_validation_recall_at_1": best_recall_at_1,
        "best_validation_loss": None if best_loss == float("inf") else best_loss,
        "warning": "Weak-supervision rows cannot establish alignment accuracy."
    }, indent=2), encoding="utf-8")
    print(f"Saved model to {args.output}")


if __name__ == "__main__":
    main()
