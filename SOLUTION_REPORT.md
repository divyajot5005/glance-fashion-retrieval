# Solution Report: Compositional Fashion & Context Retrieval

## Executive summary

The proposed system is a zero-shot, dual-encoder retrieval pipeline with fashion-specific local evidence and general scene understanding. It combines FashionCLIP for garment semantics, SigLIP 2 for environment/activity semantics, deterministic multi-crop image embeddings, and a query planner that preserves attribute bindings such as `red tie` and `white shirt`. A soft-minimum reranker rewards candidates that satisfy every atomic phrase rather than averaging away a missing attribute.

This is deliberately more than a vanilla CLIP application while remaining feasible for an internship assignment: there is no training requirement, no custom vector database, and every score is inspectable. Exact NumPy search handles 500-1,000 images; optional FAISS HNSW supplies a clear path to one million images.

The submission is structured so every important claim is falsifiable. The CLI can run five retrieval variants on the same frozen judgments, the dataset audit detects missing/corrupt/duplicate images, result galleries expose component scores, and the test suite checks the exact compositional behavior the assignment calls out. No metric is invented in the absence of images and relevance labels.

## Requirement coverage

| Assignment requirement | Implementation |
|---|---|
| 500-1,000 varied images | Generic recursive manifest builder plus Fashionpedia annotation adapter; recommended mix is 700-800 Fashionpedia and 200-300 licensed contextual images. |
| Feature extraction | FashionCLIP over five views plus SigLIP 2 over the full image. |
| Vector storage | Memory-mapped NumPy arrays by default; optional FAISS HNSW global indices. |
| Natural-language top-k | CLI search command returns ranked JSON and a per-component score breakdown. |
| Multi-attribute queries | Bound attribute atoms, multi-crop max, conjunctive soft-min reranking, and fashion/context late fusion. |
| Better than vanilla CLIP | Domain-adapted fashion encoder, separate context encoder, local views, and explicit compositional scoring. |
| Modular code | Separate indexer, storage, query, retriever, evaluation, scripts, configuration, and tests. |
| Scalability | ANN shortlist followed by exact multi-vector reranking; float16/PQ and sharding plan. |
| Zero-shot | All default inference uses public pretrained encoders; no closed label set or task-specific training is required. |

## Why this is reviewer-ready

- **Clear ML hypothesis:** domain adaptation improves garments; a scene encoder improves context; local views improve small-item recall; soft-min improves conjunctive precision.
- **Executable evidence:** each hypothesis maps to a named ablation.
- **Inspectable decisions:** results expose global-fashion, atomic-fashion, context, and metadata scores.
- **Data discipline:** the audit checks size, integrity, decodability, metadata, and duplicates.
- **Reproducibility:** versioned config, deterministic crops, seeded builders, cards, CI, Docker, and tests.
- **Honest boundary:** implemented behavior is separated from future work; no results are fabricated.

## Approaches considered

1. **Single CLIP vector per image.** Simplest and fastest baseline, but weak on fine-grained fashion, attribute binding, and small garments.
2. **Caption then text retrieval.** Highly interpretable and cheap to search, but captions can omit visually present details and turn recall errors into permanent index errors.
3. **Detector/segmenter plus attribute classifiers.** Strong localization and binding, but requires detector training or brittle open-vocabulary detection, more GPU time, and a larger engineering surface.
4. **Fine-tuned cross-encoder reranker.** Highest potential precision, but needs judged query-image pairs and is expensive over large candidate sets.
5. **Chosen hybrid.** Two pretrained encoders, deterministic local views, structured decomposition, and late fusion. It captures most benefits of localization without training or detector infrastructure.

## Chosen architecture

### Part A: Indexer

For each image, create five deterministic views: full, upper, lower, center, and wide-center. FashionCLIP embeds all views; SigLIP 2 embeds the full image. Embeddings are L2-normalized and stored separately because the two models have different spaces and roles. Each record retains an ID, path, and optional source/label metadata. Index metadata records model names, crop views, and count for reproducibility.

The default store is three simple files: `fashion.npy` with shape `N x V x Df`, `context.npy` with shape `N x Dc`, and `records.jsonl`. This is transparent and sufficient for 1,000 images. With `backend: faiss`, the indexer also creates HNSW indices over the full-view fashion vectors and context vectors.

### Part B: Retriever

The planner extracts only high-confidence visual terms. It keeps colors bound to nearby garments, extracts scene and style concepts, and falls back to the original text whenever parsing is uncertain. The two encoders independently embed their relevant query strings.

Candidate generation uses a weighted sum of global fashion and context cosine similarities. For shortlisted candidates, each atomic fashion phrase is compared with every stored crop, and the best crop supplies that atom's evidence. A temperature-controlled soft minimum combines atomic scores, penalizing any missing conjunct. The final score is:

`S = wf(0.35 S_fashion_global + 0.65 S_atomic) + wc S_context + 0.05 S_metadata`

where `wf` and `wc` depend on whether the query is fashion-heavy, context-heavy, or mixed. The metadata term is capped and optional, so filenames or labels never dominate visual retrieval.

The retriever exposes five variants from the same code path: `fashion_global`, `context_only`, `dual_global`, `multi_crop_mean`, and `proposed`. This minimizes experimental confounds because indexing, candidate pool, judgments, and metric code remain fixed.

## Why compositional reranking helps

For `red tie and white shirt`, a vanilla embedding may rank an image with a red shirt and white tie highly because all four concepts are present. This system creates the bound atoms `red tie` and `white shirt`. Each atom must find supporting evidence in at least one crop, and the soft minimum makes the lower atomic score decisive. The same mechanism handles `blue shirt` plus `park` by combining a local fashion requirement with a global scene requirement.

## Dataset plan

Fashionpedia provides fine-grained categories, attributes, and segmentation-oriented imagery, but it does not guarantee balanced office, home, park, and urban contexts. Build a 1,000-image evaluation collection from:

- 700-800 Fashionpedia validation/test images, stratified by garment category and dominant color;
- 200-300 licensed lifestyle images, balanced across office, urban street, park, and home;
- deduplication by perceptual hash, source/license metadata, and no filename-derived retrieval features;
- a held-out query judgment file with at least 3-5 relevant IDs per prompt.

The repository contains both a generic directory manifest builder and a Fashionpedia COCO-annotation adapter. The images themselves are not redistributed.

## Evaluation protocol

Report Recall@1/5/10, MRR, and nDCG@10. Use the five supplied prompts plus at least 20 paraphrases and 20 hard negatives, including color swaps (`red shirt + blue pants` vs `blue shirt + red pants`) and context swaps (office vs home). Freeze the image set and judgments before comparing:

1. `fashion_global`: vanilla single-vector FashionCLIP;
2. `context_only`: scene encoder alone;
3. `dual_global`: query-aware global late fusion;
4. `multi_crop_mean`: local evidence with mean aggregation;
5. `proposed`: local evidence with soft-min aggregation.

Also measure indexing images/second, p50/p95 query latency, peak memory, and performance by query type. No empirical values are claimed here because the source assignment supplied neither the image corpus nor relevance judgments; inventing metrics would be misleading.

### Decision table

| Comparison | Question answered | Success criterion |
|---|---|---|
| Fashion global vs context only | Which encoder owns which query types? | Fashion wins attribute queries; context wins place-heavy queries. |
| Fashion global vs dual global | Does a separate scene branch add value? | Higher nDCG@10 on mixed and contextual prompts without material attribute regression. |
| Dual global vs multi-crop mean | Do local views recover small garments? | Higher Recall@5 on tie, shirt, and outerwear attributes. |
| Multi-crop mean vs proposed | Does conjunctive aggregation reduce swaps? | Better nDCG@10 on color-garment swap hard negatives. |

Run `glance-retrieval benchmark` once to produce all five metric blocks. Report bootstrap confidence intervals if the judgment set grows beyond the five seed prompts; with a tiny set, show per-query rankings rather than overstating aggregate significance.

## Failure analysis protocol

For every failed query, assign exactly one primary cause before changing the system: dataset coverage, query decomposition, candidate-generation miss, fashion-attribute miss, context miss, crop/localization miss, or fusion/ranking error. Save the top-10 HTML gallery and component scores. This prevents random weight tuning and makes each proposed repair testable.

Recommended hard-negative suites include color-garment swaps, visually similar garment categories, formal-vs-casual context swaps, indoor-vs-outdoor scenes, small accessories, occlusion, low light, and multiple people. A regression gate should require that a precision fix for one slice does not materially damage the others.

## Supplied-query behavior

- **Bright yellow raincoat:** atomic `yellow raincoat`; fashion-heavy, with local crop evidence.
- **Business attire in a modern office:** style terms route to FashionCLIP; `office` routes to SigLIP 2.
- **Blue shirt on a park bench:** bound `blue shirt` plus `park`; mixed late fusion.
- **Casual weekend city walk:** style inference plus `city`; broad zero-shot semantics remain in the raw query fallback.
- **Red tie and white shirt in a formal setting:** two bound atoms plus formal style; soft-min explicitly penalizes a missing or swapped item.

## Scaling to one million images

Use separate FAISS HNSW or IVF-PQ indices for global fashion and context vectors. Retrieve the union of 500-2,000 candidates, then load only their quantized crop vectors for exact reranking. Store global vectors in float16 or product-quantized form, shard by stable image ID, batch GPU indexing, content-hash images for incremental updates, and version every artifact by model/config/data hash. Perform blue/green index swaps so readers never see a partial build. The query service is stateless and horizontally scalable.

## Extending locations, cities, places, and weather

Keep visual evidence separate from external facts. Add a place branch using a geolocation/place-recognition encoder or image metadata, and represent city/place candidates in a dedicated index. Add weather as a multi-label visual classifier for rain, snow, sun, fog, and temperature proxies. Parse location and weather into their own query fields and fuse calibrated branch scores. When verified EXIF or source metadata exists, use it as a filter or capped boost; never infer a precise city solely from ambiguous visual cues.

## Improving precision

The highest-value next step is a two-stage learned reranker trained on hard negatives that swap color-garment bindings and environments. Other improvements are Fashionpedia-mask garment crops, open-vocabulary detection for accessories, calibrated weights learned on held-out judgments, query expansion with controlled fashion synonyms, color constancy, duplicate suppression, and relevance feedback. A constrained LLM can replace the rule-based planner only if its JSON schema is validated and the raw-query fallback remains.

## Limitations and safeguards

Deterministic crops assume roughly upright people. Color is lighting-sensitive. FashionCLIP was trained largely on product imagery, so lifestyle scenes are a domain shift. Style terms are subjective and culturally dependent. Track performance across available subgroups, avoid protected-trait inference, preserve source/license provenance, and support deletion/re-indexing. The system should explain component scores but should not present similarity as a factual claim about a person.

## Reproducibility and runbook

Install with `pip install -e ".[dev,faiss]"`, build a JSONL manifest, run `glance-retrieval index`, and query with `glance-retrieval search`. The first run downloads public model weights. Tests cover bound-attribute parsing, context weighting, raw-query fallback, and the missing-conjunct penalty. Configuration controls models, batch size, crop views, backend, shortlist size, and score weights.

The repository also contains a 90-second reviewer guide, model card, data card, Dockerfile, Makefile, lightweight GitHub Actions workflow, configuration validation, dataset audit, ablation benchmark, and self-contained HTML result gallery. The tests avoid downloading model weights by injecting deterministic fake encoders, so core ML logic is cheap to verify in CI.

The packaged codebase accompanies this report. A public GitHub URL was not supplied and is therefore marked **TBD after upload** rather than fabricated.

## Submission checklist

- Replace placeholder relevance IDs with frozen annotations and run the data audit.
- Run the five-way benchmark and add the generated metric table plus 2-3 representative result galleries.
- Upload the repository, replace the TBD GitHub URL, and verify a clean-machine install.
- Keep the PDF and code at the same commit; tag the submission commit.
- Explain one success, one failure, one ablation, and why soft-min beats mean for conjunctive queries.

## References

- FashionCLIP model card: https://huggingface.co/patrickjohncyh/fashion-clip
- SigLIP 2 model card: https://huggingface.co/google/siglip2-base-patch16-224
- Zhai et al., *Sigmoid Loss for Language Image Pre-Training*: https://arxiv.org/abs/2303.15343
- Jia et al., *Fashionpedia*: https://arxiv.org/abs/2004.12276
- FAISS documentation: https://faiss.ai/
