# Reviewer guide

## The 90-second path

1. Read the architecture and tradeoffs in `SOLUTION_REPORT.md`.
2. Inspect `src/glance_retrieval/query.py` for bound color-garment parsing.
3. Inspect `src/glance_retrieval/retriever.py` for the multi-crop soft-min reranker.
4. Run `make test` to verify parser, scoring, data-audit, and end-to-end ranking invariants without downloading model weights.
5. With a prepared index, run `make benchmark INDEX=... JUDGMENTS=...` to compare all ablations on identical relevance judgments.

## Claims that are implemented

- Two independent embedding spaces with explicit late fusion.
- Five fashion views per image and one global context vector.
- Attribute bindings such as `red tie` and `white shirt` remain separate.
- A missing conjunct is penalized by a soft minimum rather than hidden by averaging.
- NumPy exact search for the assignment scale and FAISS HNSW shortlist support for larger collections.
- Machine-readable dataset audit, ranking metrics, ablations, and a self-contained HTML result gallery.

## Claims intentionally not made

No retrieval metric is fabricated. The assignment did not provide an image corpus or relevance judgments, so this repository provides the exact benchmark protocol and tooling needed to produce defensible numbers after the candidate supplies licensed images and annotations.

## Most important extension

If labeled query-image pairs become available, train a lightweight cross-encoder or fusion calibrator on hard negatives that swap color-garment bindings and environments. Keep the current system as the high-recall first stage and the explainable baseline.

