.PHONY: setup up down test test-unit test-integration test-load ingest embed evaluate clean fmt lint

setup:
	/opt/homebrew/opt/python@3.11/bin/python3.11 -m venv .venv
	.venv/bin/pip install --upgrade pip -q
	.venv/bin/pip install -r requirements.txt
	.venv/bin/python -m spacy download en_core_web_lg
	cp -n .env.example .env || true
	@echo "\n✓ Run: source .venv/bin/activate"

up:
	docker compose up -d --build

down:
	docker compose down -v

logs:
	docker compose logs -f api worker

test-unit:
	PYTHONPATH=. .venv/bin/pytest tests/unit/ -v --cov=src --cov-report=term-missing

test-integration:
	PYTHONPATH=. .venv/bin/pytest tests/integration/ -v -s

test-load:
	.venv/bin/locust -f tests/load/locustfile.py --host=http://localhost:8000 --users=50 --spawn-rate=5 --run-time=60s --headless

test: test-unit test-integration

ingest:
	PYTHONPATH=. .venv/bin/python scripts/ingest_cfpb.py --sample 50000

embed:
	PYTHONPATH=. .venv/bin/python scripts/build_vector_store.py

golden:
	PYTHONPATH=. .venv/bin/python scripts/generate_golden_set.py --n 500

evaluate:
	PYTHONPATH=. .venv/bin/python scripts/evaluate_pipeline.py

rag-init:
	PYTHONPATH=. .venv/bin/python -c "from src.rag.knowledge_base import RegulatoryKnowledgeBase; RegulatoryKnowledgeBase().initialize(); print('RAG corpus ready')"

migrate:
	.venv/bin/alembic upgrade head

fmt:
	.venv/bin/black src/ tests/ scripts/ prompts/
	.venv/bin/isort src/ tests/ scripts/ prompts/

lint:
	.venv/bin/ruff check src/ tests/
	.venv/bin/mypy src/ --ignore-missing-imports

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	rm -rf .pytest_cache htmlcov .coverage

api-dev:
	PYTHONPATH=. .venv/bin/uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

worker-dev:
	PYTHONPATH=. .venv/bin/celery -A src.tasks.celery_app worker --loglevel=debug --concurrency=2
