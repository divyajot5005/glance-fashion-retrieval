#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Fashionpedia COCO annotations to JSONL")
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    data = json.loads(args.annotations.read_text(encoding="utf-8"))
    categories = {item["id"]: item.get("name", str(item["id"])) for item in data["categories"]}
    attributes = {item["id"]: item.get("name", str(item["id"])) for item in data.get("attributes", [])}
    by_image: dict[int, dict[str, set[str]]] = defaultdict(
        lambda: {"garments": set(), "attributes": set()}
    )
    for annotation in data["annotations"]:
        bucket = by_image[annotation["image_id"]]
        bucket["garments"].add(categories.get(annotation["category_id"], "unknown"))
        for attribute_id in annotation.get("attribute_ids", []):
            if attribute_id in attributes:
                bucket["attributes"].add(attributes[attribute_id])

    images = list(data["images"])
    random.Random(args.seed).shuffle(images)
    images = images[: args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for item in images:
            labels = by_image[item["id"]]
            record = {
                "image_id": str(item["id"]),
                "path": str((args.images / item["file_name"]).resolve()),
                "metadata": {
                    "garments": sorted(labels["garments"]),
                    "attributes": sorted(labels["attributes"]),
                    "source": "fashionpedia",
                },
            }
            handle.write(json.dumps(record) + "\n")
    print(f"Wrote {len(images)} Fashionpedia records to {args.output}")


if __name__ == "__main__":
    main()

