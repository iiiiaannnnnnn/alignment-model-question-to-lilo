"""Turn the audited Test 7 CSV into standalone evaluation JSONL."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from src.build_synthetic_data import DEFAULT_SOURCE, load_lilos


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROWS = ROOT / "data" / "external" / "exp7_rows_chem31a.csv"
DEFAULT_OUTPUT = ROOT / "data" / "evaluation" / "chem31a_human.jsonl"


def main() -> None:
    lilos = load_lilos(DEFAULT_SOURCE)
    key_to_id = {f"{l['topic']}.{chr(65 + i)}": l["id"] for l in lilos for i in []}
    # Goal letters reset inside each topic.
    key_to_id = {}
    seen_by_topic: dict[str, int] = {}
    for l in lilos:
        i = seen_by_topic.get(l["topic"], 0)
        key_to_id[f"{l['topic']}.{chr(65 + i)}"] = l["id"]
        seen_by_topic[l["topic"]] = i + 1

    seen = set()
    DEFAULT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with DEFAULT_ROWS.open(encoding="utf-8", newline="") as src, DEFAULT_OUTPUT.open("w", encoding="utf-8") as out:
        for row in csv.DictReader(src):
            if row["condition"] != "A_whole_syllabus" or row["qid"] in seen:
                continue
            seen.add(row["qid"])
            gold_keys = row["gold_keys"].split("|")
            out.write(json.dumps({
                "question_id": row["qid"],
                "question": row["question"],
                "gold_lilo_ids": [key_to_id[key] for key in gold_keys],
                "gold_keys": gold_keys,
                "source": "Stanford CHEM 31A 2021 course-staff labels via SmartSTEM",
            }, ensure_ascii=True) + "\n")
    print(f"Wrote {len(seen)} human-labelled evaluation rows to {DEFAULT_OUTPUT}")


if __name__ == "__main__":
    main()
