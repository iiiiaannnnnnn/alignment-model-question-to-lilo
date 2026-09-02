"""Build reproducible weak-supervision pairs from a LILO list.

This creates template-generated data, not human-labelled ground truth. It is
kept separate from the human evaluation set on purpose.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "data" / "external" / "chem31a_learning_goal_list.txt"
DEFAULT_OUTPUT = ROOT / "data" / "synthetic" / "synthetic_weak_train.jsonl"


def load_lilos(path: Path) -> list[dict]:
    topic = None
    items = []
    for raw in path.read_text(encoding="utf-8").splitlines()[1:]:
        line = raw.strip()
        if not line:
            continue
        header = re.match(r"^(\d+)\.(.+)$", line)
        if header:
            topic = header.group(1)
            continue
        if topic:
            items.append({"id": f"LILO-{len(items) + 1}", "topic": topic, "lilo": line})
    if not items:
        raise ValueError(f"No LILOs found in {path}")
    return items


def question_variants(lilo: str) -> list[str]:
    clean = lilo.rstrip(".")
    return [
        f"Which learning outcome is assessed when a student must {clean.lower()}?",
        f"Create an assessment question that asks students to {clean.lower()}.",
        f"A student is required to {clean.lower()}. What should the question assess?",
        f"Select the course outcome best aligned with this task: {clean}",
    ]


def build(source: Path, output: Path, copies: int, seed: int) -> int:
    rng = random.Random(seed)
    lilos = load_lilos(source)
    by_topic = {topic: [l for l in lilos if l["topic"] == topic] for topic in {l["topic"] for l in lilos}}
    output.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with output.open("w", encoding="utf-8") as f:
        for target in lilos:
            same_topic = [l for l in by_topic[target["topic"]] if l["id"] != target["id"]]
            other_topic = [l for l in lilos if l["topic"] != target["topic"]]
            for copy_index in range(copies):
                question = question_variants(target["lilo"])[copy_index % 4]
                qid = f"synthetic-{target['id']}-{copy_index:02d}"
                rows = [(target, 1, "template_target")]
                if same_topic:
                    rows.append((rng.choice(same_topic), 0, "same_topic_hard_negative"))
                rows.extend((negative, 0, "other_topic_negative") for negative in rng.sample(other_topic, k=2))
                for candidate, label, pair_type in rows:
                    f.write(json.dumps({
                        "question_id": qid,
                        "question": question,
                        "lilo_id": candidate["id"],
                        "lilo": candidate["lilo"],
                        "label": label,
                        "topic": candidate["topic"],
                        "source": "synthetic_template_weak_supervision",
                        "pair_type": pair_type,
                    }, ensure_ascii=True) + "\n")
                    total += 1
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--copies", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260829)
    args = parser.parse_args()
    count = build(args.source, args.output, args.copies, args.seed)
    print(f"Wrote {count} weak-supervision rows to {args.output}")


if __name__ == "__main__":
    main()
