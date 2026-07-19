#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a retrieval JSONL manifest from images")
    parser.add_argument("image_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    paths = sorted(path for path in args.image_root.rglob("*") if path.suffix.lower() in EXTENSIONS)
    random.Random(args.seed).shuffle(paths)
    paths = paths[: args.limit]
    if not paths:
        raise SystemExit(f"No images found below {args.image_root}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for path in paths:
            record = {
                "image_id": path.stem,
                "path": str(path.resolve()),
                "metadata": {"source_folder": path.parent.name},
            }
            handle.write(json.dumps(record) + "\n")
    print(f"Wrote {len(paths)} images to {args.output}")


if __name__ == "__main__":
    main()

