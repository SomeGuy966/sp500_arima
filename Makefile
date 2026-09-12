.PHONY: help install lint format typecheck test check diagnose fit backtest report clean

PY ?= poetry run

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Install the package with dev + fetch extras (Poetry)
	poetry install --all-extras

lint: ## Ruff lint + format check
	$(PY) ruff check src tests
	$(PY) ruff format --check src tests

format: ## Auto-format with Ruff
	$(PY) ruff format src tests
	$(PY) ruff check --fix src tests

typecheck: ## mypy --strict
	$(PY) mypy

test: ## pytest with coverage
	$(PY) pytest --cov

check: lint typecheck test ## Everything CI runs

diagnose: ## Stationarity + autocorrelation tests on recent data
	$(PY) sp500-arima diagnose

fit: ## Single 2024 train/test split with forecast + residual plots
	$(PY) sp500-arima fit --start 2024-01-01 --end 2025-01-01

backtest: ## Full 1975-2024 walk-forward backtest (~2 min on 8 cores)
	$(PY) sp500-arima backtest --out reports

report: ## Rebuild tables + figures from the saved backtest CSVs
	$(PY) sp500-arima report --out reports

clean: ## Remove caches and build artefacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage coverage.xml htmlcov dist build
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
