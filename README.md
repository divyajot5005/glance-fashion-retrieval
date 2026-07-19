# Glance Multimodal Fashion & Context Retrieval

A modular image search system that retrieves fashion scenes from natural-language queries. It is deliberately stronger than a one-vector, vanilla-CLIP baseline:

- `FashionCLIP` supplies domain-specific garment and attribute semantics.
- `SigLIP 2` supplies broader scene, activity, and environment semantics.
- Five deterministic image views preserve local evidence without requiring an object detector.
- A transparent query planner extracts bound attribute phrases such as `red tie` and `white shirt`.
- Late fusion uses a soft minimum over atomic phrases, so one excellent match cannot hide a missing conjunct.
- Optional manifest metadata gives a small, auditable boost; it never replaces visual similarity.

Start with [`REVIEWER_GUIDE.md`](REVIEWER_GUIDE.md) for the shortest review path. [`DATA_CARD.md`](DATA_CARD.md) and [`MODEL_CARD.md`](MODEL_CARD.md) state the collection contract, intended use, evaluation protocol, and limitations.

## Architecture

```text
Indexer:   image -> [full, upper, lower, center, wide-center]
                    | FashionCLIP -> multi-vector fashion store
                    + full image -> SigLIP 2 -> context store

Retriever: query -> high-confidence decomposition
                    | fashion atoms -> multi-crop conjunctive score
                    | full/context text -> global fashion + context scores
                    + weighted late fusion -> top-k images + score breakdown
```

The system uses exact NumPy search for the assignment-sized collection. Set `index.backend: faiss` to build HNSW indices for large collections; exact multi-view reranking still runs only on the shortlist.

## Quick start

Python 3.10+ is required. The first real indexing run downloads roughly 2.1 GB of model weights.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,faiss]"

python scripts/build_manifest.py /path/to/images data/manifest.jsonl --limit 1000
glance-retrieval audit --manifest data/manifest.jsonl --output artifacts/data-audit.json
glance-retrieval index --manifest data/manifest.jsonl --output artifacts/index

glance-retrieval search --index artifacts/index \
  --query "A red tie and a white shirt in a formal setting" -k 5 \
  --html artifacts/results.html
```

Use the provided Fashionpedia adapter when annotations are available:

```bash
python scripts/prepare_fashionpedia.py \
  --images /path/to/fashionpedia/images \
  --annotations /path/to/instances_attributes_val2020.json \
  --output data/fashionpedia.jsonl --limit 1000
```

Fashionpedia is rich in garment categories and attributes but comparatively weak on office/home/park coverage. For a faithful evaluation set, combine 700-800 Fashionpedia images with 200-300 licensed contextual fashion photographs covering office, urban street, park, and home scenes. `build_manifest.py` accepts any directory tree, and `source_folder` is retained for auditing.

## Evaluation

Replace the placeholder relevant IDs in `examples/evaluation_queries.jsonl` after annotating at least 3-5 relevant images per query, then run:

```bash
glance-retrieval evaluate --index artifacts/index \
  --judgments examples/evaluation_queries.jsonl
```

Reported metrics are Recall@1/5/10, MRR, and nDCG@10. Compare these ablations on the identical judgments:

1. `fashion_global`: vanilla one-vector FashionCLIP;
2. `context_only`: scene encoder alone;
3. `dual_global`: two global embeddings with query-aware late fusion;
4. `multi_crop_mean`: local evidence but permissive averaging;
5. `proposed`: local evidence with conjunctive soft-min reranking.

This isolates whether improvements come from domain adaptation, scene understanding, or compositional reranking.

Run all five variants in one command:

```bash
glance-retrieval benchmark --index artifacts/index \
  --judgments examples/evaluation_queries.jsonl \
  --output artifacts/benchmark.json
```

The repository intentionally ships no invented metric. The source assignment contains neither the actual image corpus nor relevance judgments, so defensible numbers can only be generated after those inputs are supplied and frozen.

## Repository layout

```text
src/glance_retrieval/
  indexer.py       # Part A: image ingestion and feature extraction
  storage.py       # NumPy store and optional FAISS HNSW
  query.py         # query decomposition with bound attributes
  retriever.py     # Part B: shortlist, late fusion, reranking
  evaluation.py    # ranking metrics
scripts/           # generic and Fashionpedia manifest builders
configs/           # model and retrieval settings
tests/             # parser and scoring invariants
.github/workflows/ # lightweight CI without model downloads
```

## Design tradeoffs

- **Why two encoders?** FashionCLIP is trained for fashion concepts, while SigLIP 2 is better suited to broad semantic scene matching. Late fusion keeps their embedding spaces separate and interpretable.
- **Why deterministic crops?** They add local evidence at very low engineering cost. A learned clothing detector or Fashionpedia masks should outperform them but increases latency and failure modes.
- **Why a rule-based planner?** The supplied evaluation vocabulary is compact, and deterministic parsing is testable and has no API dependency. A constrained LLM parser is a future option, but it must return a validated schema and fall back safely.
- **Why HNSW only for the global shortlist?** Storing every crop in ANN multiplies memory. Global ANN plus exact crop reranking balances recall, cost, and simplicity.

## Scaling to one million images

Use two FAISS HNSW or IVF-PQ indices over normalized global vectors, store crop vectors as float16 or product-quantized blocks, retrieve a 500-2,000 item union shortlist, and rerank it exactly. Batch GPU inference, content-hash images for incremental indexing, version model/config metadata, and use blue/green index swaps. The query path is stateless and horizontally scalable.

## Limitations and responsible use

Deterministic crops assume roughly upright people, color words remain sensitive to illumination, FashionCLIP's product-image training distribution differs from lifestyle scenes, and style terms are culturally subjective. Report subgroup performance where labels permit, never infer protected traits, retain licensed source metadata, and provide a deletion path for indexed images.

## Sources

- [FashionCLIP model card](https://huggingface.co/patrickjohncyh/fashion-clip)
- [SigLIP 2 model card](https://huggingface.co/google/siglip2-base-patch16-224)
- [SigLIP paper](https://arxiv.org/abs/2303.15343)
- [Fashionpedia project](https://fashionpedia.github.io/home/index.html)
- [FAISS documentation](https://faiss.ai/)
