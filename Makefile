.PHONY: install init lint fmt type test check up up-obs down eval-fast eval m1-check test-live

install:
	uv sync

init:
	uv run gem-init-env

lint:
	uv run ruff check .
	uv run ruff format --check .

fmt:
	uv run ruff check --fix .
	uv run ruff format .

type:
	uv run mypy

test:
	uv run pytest

check: lint type test

up:
	docker compose up -d --build --wait

up-obs:
	docker compose -f docker-compose.yml -f docker-compose.langfuse.yml up -d --build --wait

down:
	docker compose -f docker-compose.yml -f docker-compose.langfuse.yml down

eval-fast eval:
	@echo "PENDING: the eval runner lands in M6 (R10); nothing to run yet." && exit 1

m1-check:
	uv run gem-m1-check

test-live:
	RUN_INTEGRATION=1 uv run pytest -m integration -s
