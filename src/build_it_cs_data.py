"""Build leakage-safe weak IT/CS question-LILO training pairs."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = Path(
    r"C:\Users\Jericho Mico\Downloads\prime_it_v3_IT_CS_question_LILO_synthetic.jsonl"
)
TOPIC_CODES = {
    "Programming": "PROG",
    "Data Structures": "DSTR",
    "Algorithms": "ALG",
    "Databases": "DB",
    "Networking": "NET",
    "Cybersecurity": "SEC",
    "Artificial Intelligence": "AI",
    "Data Science": "DATA",
    "Web Development": "WEB",
    "Software Engineering": "SE",
    "Cloud & DevOps": "CLOUD",
    "Emerging Technologies": "EMERGE",
}
CATALOG_VERSION = "PRIME-TLOAD-1.0"


def tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def similarity(left: str, right: str) -> float:
    a, b = tokens(left), tokens(right)
    return len(a & b) / max(len(a | b), 1)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def stable_id(topic: str, original_id: str) -> str:
    number = int(re.search(r"\d+", original_id).group())
    topic_start = ((number - 1) // 10) * 10
    return f"{TOPIC_CODES[topic]}-{number - topic_start:02d}"


def bloom_level(lilo: str) -> str:
    verb = lilo.split(maxsplit=1)[0].lower()
    return {
        "explain": "understand", "interpret": "understand",
        "apply": "apply", "use": "apply", "implement": "apply",
        "write": "apply", "traverse": "apply", "troubleshoot": "apply",
        "analyze": "analyze", "select": "analyze", "compare": "analyze",
        "evaluate": "evaluate", "design": "create", "communicate": "create",
    }.get(verb, "unspecified")


def build(source: Path, output: Path, seed: int) -> dict:
    raw = read_jsonl(source)
    if not raw or {row.get("label") for row in raw} != {1}:
        raise ValueError("Source must contain the positive-only synthetic question catalogue")

    objective_rows: dict[str, dict] = {}
    by_topic: dict[str, list[str]] = defaultdict(list)
    for row in raw:
        topic = row["topic"]
        if topic not in TOPIC_CODES:
            raise ValueError(f"Unknown topic: {topic}")
        lilo_id = stable_id(topic, row["lilo_id"])
        candidate = {
            "lilo_id": lilo_id,
            "lilo": row["lilo"].strip(),
            "domain": topic,
            "subdomain": "unspecified",
            "lilo_family_id": f"{TOPIC_CODES[topic]}-FAMILY",
            "bloom_level": bloom_level(row["lilo"]),
            "catalog_version": CATALOG_VERSION,
            "objective_source": "synthetic_authored",
            "review_status": "unreviewed",
        }
        previous = objective_rows.setdefault(lilo_id, candidate)
        if previous != candidate:
            raise ValueError(f"Conflicting catalogue entry for {lilo_id}")
        if lilo_id not in by_topic[topic]:
            by_topic[topic].append(lilo_id)

    # Hold out two complete objective/question families per domain. Their texts
    # never appear in training, so validation measures transfer to unseen LILOs.
    validation_ids = set()
    for topic, ids in by_topic.items():
        ranked = sorted(ids, key=lambda value: hashlib.sha256(f"{seed}:{value}".encode()).hexdigest())
        validation_ids.update(ranked[:2])

    questions = []
    pairs = {"train": [], "validation": []}
    rng = random.Random(seed)
    catalog = list(objective_rows.values())
    for row in raw:
        target_id = stable_id(row["topic"], row["lilo_id"])
        target = objective_rows[target_id]
        split = "validation" if target_id in validation_ids else "train"
        family_id = f"{target_id}-GENERATED-V3"
        question = {
            "question_id": row["question_id"].replace(row["lilo_id"], target_id),
            "question": row["question"].strip(),
            "gold_lilo_ids": [target_id],
            "question_family_id": family_id,
            "lilo_family_id": target["lilo_family_id"],
            "domain": row["topic"],
            "subdomain": "unspecified",
            "question_type": "short_answer",
            "bloom_level": target["bloom_level"],
            "difficulty": "unspecified",
            "generator_prompt_version": "synthetic_it_cs_human_style_v3",
            "catalog_version": CATALOG_VERSION,
            "label_source": "synthetic_generation_target",
            "label_quality": "weak",
            "review_status": "unreviewed",
            "split": split,
        }
        questions.append(question)

        candidate_catalog = catalog if split == "validation" else [
            item for item in catalog if item["lilo_id"] not in validation_ids
        ]
        same_domain = [item for item in candidate_catalog
                       if item["domain"] == target["domain"] and item["lilo_id"] != target_id]
        other_domain = [item for item in candidate_catalog if item["domain"] != target["domain"]]
        hard_same = max(same_domain, key=lambda item: similarity(question["question"], item["lilo"]))
        hard_cross = max(other_domain, key=lambda item: similarity(question["question"], item["lilo"]))
        remaining = [item for item in other_domain if item["lilo_id"] != hard_cross["lilo_id"]]
        random_cross = rng.choice(remaining)
        candidates = [
            (target, 1, "generation_target"),
            (hard_same, 0, "same_domain_hard_negative"),
            (hard_cross, 0, "cross_domain_hard_negative"),
            (random_cross, 0, "cross_domain_random_negative"),
        ]
        for candidate, label, pair_type in candidates:
            pairs[split].append({
                "question_id": question["question_id"],
                "question": question["question"],
                "lilo_id": candidate["lilo_id"],
                "lilo": candidate["lilo"],
                "label": label,
                "pair_type": pair_type,
                "question_family_id": family_id,
                "lilo_family_id": target["lilo_family_id"],
                "domain": question["domain"],
                "candidate_domain": candidate["domain"],
                "source": "synthetic_it_cs_human_style_v3",
                "label_source": "synthetic_generation_target",
                "label_quality": "weak",
                "review_status": "unreviewed",
                "catalog_version": CATALOG_VERSION,
            })

    train_questions = [row for row in questions if row["split"] == "train"]
    validation_questions = [row for row in questions if row["split"] == "validation"]
    train_families = {row["question_family_id"] for row in train_questions}
    validation_families = {row["question_family_id"] for row in validation_questions}
    assert not train_families & validation_families
    assert {row["lilo_id"] for row in pairs["train"]} & validation_ids == set()
    for split in pairs:
        grouped = defaultdict(list)
        for row in pairs[split]:
            grouped[row["question_id"]].append(row)
        assert all(len(rows) == 4 and sum(row["label"] for row in rows) == 1 for rows in grouped.values())

    write_jsonl(output / "catalog" / "lilos.jsonl", catalog)
    write_jsonl(output / "questions" / "train.jsonl", train_questions)
    write_jsonl(output / "questions" / "validation.jsonl", validation_questions)
    write_jsonl(output / "pairs" / "train.jsonl", pairs["train"])
    write_jsonl(output / "pairs" / "validation.jsonl", pairs["validation"])
    manifest = {
        "catalog_version": CATALOG_VERSION,
        "source": str(source),
        "seed": seed,
        "lilos": len(catalog),
        "domains": len(by_topic),
        "train_questions": len(train_questions),
        "validation_questions": len(validation_questions),
        "train_pairs": len(pairs["train"]),
        "validation_pairs": len(pairs["validation"]),
        "validation_lilo_ids": sorted(validation_ids),
        "split_rule": "Two complete LILO/question families held out per domain by seeded SHA-256 order.",
        "warning": "All labels are synthetic weak supervision and cannot establish educational alignment accuracy.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "it_cs_v4")
    parser.add_argument("--seed", type=int, default=20260830)
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.output, args.seed), indent=2))


if __name__ == "__main__":
    main()
