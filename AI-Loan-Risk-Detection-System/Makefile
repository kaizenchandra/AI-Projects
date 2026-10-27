PYTHON ?= .venv/bin/python
.PHONY: install test lint train serve audit container
install:
	python3.11 -m venv .venv
	$(PYTHON) -m pip install --require-hashes -r requirements-dev.lock
	$(PYTHON) -m pip install --no-deps -e .
test:
	$(PYTHON) -m pytest -q
lint:
	.venv/bin/ruff check src tests scripts
	.venv/bin/ruff format --check src tests scripts
	.venv/bin/mypy src/loan_risk
train:
	.venv/bin/loan-risk generate --output data/synthetic.csv
	.venv/bin/loan-risk train --data data/synthetic.csv --config configs/training.json --output artifacts/verified
serve:
	.venv/bin/uvicorn loan_risk.api:create_app --factory --host 127.0.0.1 --port 8000 --workers 1 --limit-concurrency 32 --timeout-keep-alive 5 --no-access-log --log-config src/loan_risk/logging.json
audit:
	.venv/bin/pip-audit --strict -r requirements.lock
container:
	docker build -t loan-risk:1.0.0 .
