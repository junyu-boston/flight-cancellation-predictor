.PHONY: help setup install clean train evaluate predict test lint format prepare-data

# Variables
PYTHON := uv run python
PIP := uv pip
DATA_DIR := data
MODEL_DIR := models
LOG_DIR := logs

help: ## Show this help message
	@echo "Usage: make [target]"
	@echo ""
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

setup: ## Set up the development environment
	uv venv
	$(PIP) install -r requirements.txt
	mkdir -p $(DATA_DIR)/{raw,processed,features,external}
	mkdir -p $(MODEL_DIR)/{artifacts,registry,checkpoints}
	mkdir -p $(LOG_DIR)
	@echo "✅ Environment setup complete!"

install: ## Install dependencies
	$(PIP) install -r requirements.txt
	@echo "✅ Dependencies installed!"

clean: ## Clean generated files and directories
	rm -rf $(DATA_DIR)/processed/*
	rm -rf $(MODEL_DIR)/artifacts/*
	rm -rf $(LOG_DIR)/*
	rm -rf __pycache__ .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "✅ Cleaned generated files!"

prepare-data: ## Prepare and process data for training
	$(PYTHON) -m src.cli.main prepare-data \
		--data-config configs/data_config.yaml \
		--output $(DATA_DIR)/processed
	@echo "✅ Data prepared!"

train: prepare-data ## Train the model
	$(PYTHON) -m src.cli.main train \
		--train-data $(DATA_DIR)/processed/train.parquet \
		--test-data $(DATA_DIR)/processed/test.parquet \
		--model-config configs/model_config.yaml
	@echo "✅ Model trained!"

train-quick: ## Quick training with sample data
	$(PYTHON) scripts/train.py
	@echo "✅ Quick training complete!"

evaluate: ## Evaluate the latest model
	@latest_model=$$(ls -t $(MODEL_DIR)/artifacts | head -1); \
	if [ -z "$$latest_model" ]; then \
		echo "❌ No model found. Run 'make train' first."; \
		exit 1; \
	fi; \
	$(PYTHON) -m src.cli.main evaluate \
		--model-path $(MODEL_DIR)/artifacts/$$latest_model \
		--test-data $(DATA_DIR)/processed/test.parquet
	@echo "✅ Evaluation complete!"

predict: ## Make predictions with the latest model
	@latest_model=$$(ls -t $(MODEL_DIR)/artifacts | head -1); \
	if [ -z "$$latest_model" ]; then \
		echo "❌ No model found. Run 'make train' first."; \
		exit 1; \
	fi; \
	$(PYTHON) -m src.cli.main predict \
		--model-path $(MODEL_DIR)/artifacts/$$latest_model \
		--input-data $(DATA_DIR)/processed/test.parquet \
		--output predictions.csv \
		--threshold 0.02
	@echo "✅ Predictions saved to predictions.csv!"

model-info: ## Show information about the latest model
	@latest_model=$$(ls -t $(MODEL_DIR)/artifacts | head -1); \
	if [ -z "$$latest_model" ]; then \
		echo "❌ No model found. Run 'make train' first."; \
		exit 1; \
	fi; \
	$(PYTHON) -m src.cli.main model-info \
		--model-path $(MODEL_DIR)/artifacts/$$latest_model

test: ## Run tests
	$(PYTHON) -m pytest tests/ -v
	@echo "✅ Tests complete!"

test-coverage: ## Run tests with coverage
	$(PYTHON) -m pytest tests/ --cov=src --cov-report=html --cov-report=term
	@echo "✅ Coverage report generated in htmlcov/!"

lint: ## Run code linting
	$(PYTHON) -m pylint src/
	$(PYTHON) -m flake8 src/
	@echo "✅ Linting complete!"

format: ## Format code with black
	$(PYTHON) -m black src/ tests/ scripts/
	$(PYTHON) -m isort src/ tests/ scripts/
	@echo "✅ Code formatted!"

notebook: ## Run Jupyter notebook
	$(PYTHON) -m jupyter notebook notebooks/

docker-build: ## Build Docker image
	docker build -t flight-predictor .
	@echo "✅ Docker image built!"

docker-run: ## Run Docker container
	docker run -p 8000:8000 flight-predictor

all: clean setup prepare-data train evaluate ## Run complete pipeline

.DEFAULT_GOAL := help