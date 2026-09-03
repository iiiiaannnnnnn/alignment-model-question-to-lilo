# The seven assessment exports

Each file is one real assessment exported from PRIME, with its questions, the
outcomes that were selectable at the time, and the alignment the system saved.
Together they are the 197-question benchmark used throughout this repository.

All seven come from the same Data Mining course, so the pools overlap: exports
04 and 05 draw on the same seven outcomes, and export 07 is the full-course
final whose 31-outcome pool contains most of the others.

## Names

The files were renamed on 2026-09-03. The originals carried whatever was typed
into the title box during testing. The names below come from each export's own
`topic` field -- the lectures it actually draws on -- and its question count.

| file | lectures | questions | outcome pool | original filename stem | title inside the file |
|---|---|---|---|---|---|
| `export_01_quiz_real-world-data-formulation_10q.json` | Lecture 1 | 10 | 4 | `alignment_quiz1111_2026-08-25` | 'quiz1111' |
| `export_02_quiz_data-formulation-and-item-sets_20q.json` | Lecture 1-2 | 20 | 8 | `alignment_test_quizzz_2026-08-25` | 'test quizzz' |
| `export_03_quiz_mining-item-sets_10q.json` | Lecture 2 | 10 | 4 | `alignment_quizzzzzzzzz_2026-08-25` | 'quizzzzzzzzz' |
| `export_04_quiz_text-and-image-mining_20q.json` | Lecture 4-5 | 20 | 7 | `alignment_quiz_tssstt_2026-08-25` | 'quiz-tssstt' |
| `export_05_quiz_text-and-image-mining_40q.json` | Lecture 4-5 | 40 | 7 | `alignment_testtt1uizzz_2026-08-26` | 'testtt1uizzz' |
| `export_06_quiz_big-data-and-behavior-mining_35q.json` | Lecture 8-9 | 35 | 5 | `alignment_478588_2026-08-26` | '478588' |
| `export_07_final-exam_full-course_62q.json` | Lecture 1-9 | 62 | 31 | `alignment_final_exam_2026-08-25` | 'Final Exam' |

## What was NOT renamed, and why

The `group_id` and `question_id` fields inside the training sets
(`data/mixed_weak/`, `v4a_it_extended/`) and inside every results CSV still
read `alignment_quizzzzzzzzz_2026-08-25`, and so on. Those are **identifiers,
not file paths**. The contamination audit and the grouped evaluation join on
them, and the published results were produced against them. Rewriting them
would sever the link between a metric and the data that produced it, which is
the one thing this repository exists to preserve.

The `assessment.title` field inside each JSON is likewise untouched: it records
what was actually typed, and that is part of the raw export.

Use the table above to map an identifier back to a file.
