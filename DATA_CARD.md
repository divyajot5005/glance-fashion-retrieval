# Data card

## Intended collection

The target collection contains 500-1,000 fashion photographs with deliberate variation across environment, clothing type, garment color, and style. A recommended 1,000-image composition is 700-800 Fashionpedia images plus 200-300 separately licensed contextual fashion photographs covering offices, urban streets, parks, and homes.

## Required record schema

Each JSONL row contains a stable `image_id`, a local `path`, and optional `metadata`. Retrieval never derives semantic features from filenames. Metadata is retained for provenance, auditing, and a capped optional score boost.

## Quality gates

Run `glance-retrieval audit --manifest ... --expected-min 500`. A passing collection has:

- at least the expected number of records;
- unique image IDs;
- no missing or unreadable images;
- no exact dHash duplicate groups;
- documented source and license coverage;
- a frozen relevance-judgment file separate from model development.

## Evaluation judgments

Annotate at least 3-5 relevant image IDs per query. Include the five supplied prompts, paraphrases, attribute swaps, environment swaps, and genuinely irrelevant hard negatives. Freeze judgments before tuning fusion weights.

## Known coverage risks

Fashionpedia is strong for localized garments and fine-grained attributes but is not designed to balance office, home, park, and city contexts. Lifestyle augmentation is therefore a requirement, not optional polish. Report per-query-type metrics so a strong aggregate score cannot conceal context or compositional failures.

## Privacy, licensing, and deletion

Keep source URL/license fields where available, do not infer protected traits, and maintain a deletion path that removes both the source record and derived vectors. The repository does not redistribute images.

