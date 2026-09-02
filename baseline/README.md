# Baseline snapshot

`app/utils/alignment_model.py` is copied byte-for-byte from the live PRIME backend on 2026-08-29. It is the model implementation that every candidate must beat.

The baseline imports PRIME application modules, so it is retained for source/API comparison and future integration. The standalone Colab scripts use only exported data files and a pretrained reranker.
