PYTHON ?= python3
CONFIG ?= configs/default.yaml

.PHONY: install test audit index search benchmark

install:
	$(PYTHON) -m pip install -e ".[dev,faiss]"

test:
	PYTHONPATH=src $(PYTHON) -m pytest -q

audit:
	glance-retrieval --config $(CONFIG) audit --manifest $(MANIFEST) --output artifacts/data-audit.json

index:
	glance-retrieval --config $(CONFIG) index --manifest $(MANIFEST) --output $(INDEX)

search:
	glance-retrieval --config $(CONFIG) search --index $(INDEX) --query "$(QUERY)" -k 10 --html artifacts/results.html

benchmark:
	glance-retrieval --config $(CONFIG) benchmark --index $(INDEX) --judgments $(JUDGMENTS) --output artifacts/benchmark.json

