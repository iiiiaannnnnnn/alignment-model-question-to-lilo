# Dataset decisions

## Included for weak training

### Existing PRIME assessment exports

- 432 unique generated questions have `source_ilo_id` and a matching LILO in the assessment pool.
- The source ID is the target selected before generation. It is useful weak supervision, but is not an independent human judgment.
- Saved `alignment.lilo_id` values are model outputs and are never used as target labels.
- Files under `quiz_exam_withalignement` and `Q_Testing_Models/Syllabus and FIle/jsons` are transformed into standalone JSONL; student records are not involved.

### OpenStax Chemistry 2e

- 1,711 real textbook questions and 306 objectives across 113 subchapters.
- CC BY 4.0 underlying textbook: https://openstax.org/details/books/chemistry-2e
- Label resolution is structural: every objective in the question's published subchapter is accepted. It is not exact per-question expert labeling.

## Evaluation only

### Stanford CHEM 31A 2021

- 103 joined real exam questions, 75 goals, mean 1.806 course-staff labels per question.
- Never included in generated training files.
- SmartSTEM repository has no explicit LICENSE. Keep this for internal academic evaluation; do not redistribute publicly without permission.
- Paper: Zur et al., EDM 2023, ERIC ED630880: https://eric.ed.gov/?id=ED630880

## Investigated but not included

- ASSISTments: useful expert skill tags, but problem bodies require contacting ASSISTments and written agreement to its Terms of Use. Do not scrape or redistribute it. https://sites.google.com/site/assistmentsdata/datasets/assistments-problems
- EdNet: 13,169 problems and expert tags, but the public content table does not include question text. It cannot directly train a text-to-LILO reranker. CC BY-NC 4.0. https://github.com/riiid/ednet
- Figshare Exam Question Datasets: 2,522 human Bloom labels, CC BY 4.0. Useful for Bloom classification, not question-to-LILO relevance. DOI: https://doi.org/10.6084/m9.figshare.22597957
- Draft internal “manual” labels: marked AI-suggested/review or draft; not professor-certified. Excluded.

## Final evidence still required

Collect de-identified, blind professor labels for ISU-CS questions. Keep an untouched test set separated by assessment. A model trained on weak or AI-generated targets cannot establish production accuracy by itself.
