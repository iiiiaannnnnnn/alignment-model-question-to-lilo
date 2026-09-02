"""Small checks that fail when the baseline, candidate, or data changes unexpectedly."""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "baseline" / "app" / "utils" / "alignment_model.py"
CANDIDATE = ROOT / "candidate" / "app" / "utils" / "alignment_model.py"


def lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    baseline_hash = hashlib.sha256(BASELINE.read_bytes()).hexdigest()
    candidate_hash = hashlib.sha256(CANDIDATE.read_bytes()).hexdigest()
    assert baseline_hash == candidate_hash, "Candidate is no longer identical to the baseline"
    human = lines(ROOT / "data" / "evaluation" / "chem31a_human.jsonl")
    synthetic = lines(ROOT / "data" / "synthetic" / "synthetic_weak_train.jsonl")
    mixed_train = lines(ROOT / "data" / "mixed_weak" / "mixed_train.jsonl")
    mixed_validation = lines(ROOT / "data" / "mixed_weak" / "mixed_validation.jsonl")
    assert len(human) == 103, f"Expected 103 human-labelled rows, got {len(human)}"
    assert len(synthetic) == 9000, f"Expected 9000 synthetic rows, got {len(synthetic)}"
    assert {row["label"] for row in synthetic} == {0, 1}
    assert all(row["source"] == "synthetic_template_weak_supervision" for row in synthetic)
    assert all(row["source"].startswith("Stanford CHEM 31A") for row in human)
    assert not ({row["group_id"] for row in mixed_train} & {row["group_id"] for row in mixed_validation})
    assert all(row["label_quality"] == "weak" for row in mixed_train + mixed_validation)
    assert not any(row["source"].startswith("Stanford CHEM 31A") for row in mixed_train + mixed_validation)
    it_root = ROOT / "data" / "it_cs_v4"
    if it_root.exists():
        it_train = lines(it_root / "pairs" / "train.jsonl")
        it_validation = lines(it_root / "pairs" / "validation.jsonl")
        assert {row["label"] for row in it_train} == {0, 1}
        assert {row["label"] for row in it_validation} == {0, 1}
        assert not ({row["question_id"] for row in it_train} &
                    {row["question_id"] for row in it_validation})
        assert not ({row["question_family_id"] for row in it_train} &
                    {row["question_family_id"] for row in it_validation})
        for rows in (it_train, it_validation):
            groups = defaultdict(list)
            for row in rows:
                groups[row["question_id"]].append(row)
            assert all(len(group) == 4 and sum(row["label"] for row in group) == 1
                       for group in groups.values())
    print("Workspace valid")
    print(f"baseline/candidate sha256: {baseline_hash}")
    print(f"human evaluation rows: {len(human)}")
    print(f"synthetic weak-supervision rows: {len(synthetic)}")
    print(f"mixed weak train/validation rows: {len(mixed_train)}/{len(mixed_validation)}")
    if it_root.exists():
        print(f"IT/CS V4 train/validation pairs: {len(it_train)}/{len(it_validation)}")


if __name__ == "__main__":
    main()
