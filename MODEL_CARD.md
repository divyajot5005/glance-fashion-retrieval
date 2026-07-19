# Model card

## System

This retrieval system combines `patrickjohncyh/fashion-clip` for fashion semantics with `google/siglip2-base-patch16-224` for scene and activity semantics. It adds deterministic local views, high-confidence query decomposition, and late-fusion reranking. No task-specific training is required by default.

## Intended use

Natural-language search over a licensed fashion-image collection, with queries that mix garment attributes, clothing type, style, activity, and environment.

## Out of scope

Identity recognition, protected-trait inference, factual claims about people, precise city geolocation from visual appearance alone, or automated high-stakes decisions.

## Scoring

The proposed score combines global fashion similarity, conjunctive atomic fashion similarity, context similarity, and an optional capped metadata bonus. Results expose every component for debugging.

## Limitations

- FashionCLIP's product-image training distribution differs from lifestyle scenes.
- Deterministic crops assume people are reasonably centered and upright.
- Colors are illumination-sensitive.
- Style terms are culturally subjective.
- Rule-based decomposition is intentionally conservative and falls back to the raw query for unknown concepts.

## Evaluation

Use Recall@1/5/10, MRR, nDCG@10, latency, memory, and per-query-type slices. Always compare the proposed variant against `fashion_global`, `context_only`, `dual_global`, and `multi_crop_mean` using the same frozen judgments.

