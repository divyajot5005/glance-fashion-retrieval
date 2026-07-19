from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from .audit import audit_json
from .config import validate_config
from .evaluation import benchmark_variants, evaluate
from .gallery import write_gallery
from .indexer import build_index
from .retriever import Retriever, VARIANTS


def load_config(path: Path) -> dict:
    return validate_config(yaml.safe_load(path.read_text(encoding="utf-8")))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Fashion and context retrieval")
    root.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    commands = root.add_subparsers(dest="command", required=True)
    index = commands.add_parser("index", help="Embed images from a JSONL manifest")
    index.add_argument("--manifest", type=Path, required=True)
    index.add_argument("--output", type=Path, required=True)
    search = commands.add_parser("search", help="Retrieve images for a natural-language query")
    search.add_argument("--index", type=Path, required=True)
    search.add_argument("--query", required=True)
    search.add_argument("-k", type=int, default=5)
    search.add_argument("--variant", choices=VARIANTS, default="proposed")
    search.add_argument("--html", type=Path, help="Write a self-contained visual result gallery")
    score = commands.add_parser("evaluate", help="Compute retrieval metrics from judgments")
    score.add_argument("--index", type=Path, required=True)
    score.add_argument("--judgments", type=Path, required=True)
    score.add_argument("--variant", choices=VARIANTS, default="proposed")
    benchmark = commands.add_parser("benchmark", help="Compare all retrieval ablations")
    benchmark.add_argument("--index", type=Path, required=True)
    benchmark.add_argument("--judgments", type=Path, required=True)
    benchmark.add_argument("--output", type=Path, help="Optional JSON report path")
    audit = commands.add_parser("audit", help="Validate image integrity and dataset coverage")
    audit.add_argument("--manifest", type=Path, required=True)
    audit.add_argument("--expected-min", type=int, default=500)
    audit.add_argument("--output", type=Path, help="Optional JSON report path")
    return root


def main() -> None:
    args = parser().parse_args()
    config = load_config(args.config)
    if args.command == "index":
        build_index(config, args.manifest, args.output)
    elif args.command == "search":
        engine = Retriever(config, args.index)
        results = engine.search(args.query, args.k, args.variant)
        print(engine.as_json(results))
        if args.html:
            write_gallery(args.query, results, args.html, args.variant)
    elif args.command == "evaluate":
        engine = Retriever(config, args.index)
        print(json.dumps(evaluate(engine, args.judgments, variant=args.variant), indent=2))
    elif args.command == "benchmark":
        engine = Retriever(config, args.index)
        report = json.dumps(benchmark_variants(engine, args.judgments), indent=2)
        print(report)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(report + "\n", encoding="utf-8")
    elif args.command == "audit":
        report = audit_json(args.manifest, args.expected_min)
        print(report)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
