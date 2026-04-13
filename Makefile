# Common commands for CMP ETL Pipeline

install-dev:
	pip install -r requirements.txt -r requirements-dev.txt

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=plugins --cov-report=term-missing

lint:
	ruff check dags/ plugins/ tests/

up:
	docker compose up -d

down:
	docker compose down

reset:
	docker compose down -v

build:
	docker compose build --no-cache

logs:
	docker compose logs airflow --tail=50 -f
