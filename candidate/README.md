# Candidate module

This file begins as an exact copy of the production baseline. It is deliberately not edited while training is still experimental.

After a trained reranker wins on held-out professor-reviewed ISU-CS data, add the reranker only inside `tag_questions_with_lilos()` after the existing composite scorer has produced its candidate ranking and before the final non-source winner is selected.

Rules for the replacement:

- Keep all current public functions and their signatures.
- Keep `source_ilo_id` authoritative for generated questions.
- Keep `LILO-NONE`, candidate filtering, Bloom fields, diagnostics, and review behavior.
- If the trained model is missing or errors, use the current composite winner.
- Never use the chemistry synthetic data as evidence that the deployed ISU-CS model is accurate.

Run `python -m src.validate_workspace` before editing. It verifies that baseline and candidate are initially identical.
