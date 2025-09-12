.PHONY: venv install install-dev clean lint test install-uv

# Define variables
VENV_DIR := .venv

# Default target
all: help

# Create virtual environment using uv
venv: install-uv
	@echo "Creating virtual environment using uv..."
	@uv venv $(VENV_DIR)

# Install dependencies from pyproject.toml
install: venv
	@echo "Installing dependencies from pyproject.toml..."
	@uv pip install -e .

# Install development dependencies (black and hatch)
install-dev: install
	@echo "Installing development dependencies (black and hatch)..."
	@uv pip install black hatch

# Clean up virtual environment
clean:
	@echo "Cleaning up..."
	@rm -rf $(VENV_DIR)
	@rm -rf *.egg-info
	@rm -rf dist
	@rm -rf build
	@find . -type d -name __pycache__ -exec rm -rf {} +
	@find . -type f -name "*.pyc" -delete

# Run linting with black
lint: install-dev
	@echo "Running black..."
	@uv run black --diff --check -t py311 .

# Run tests with hatch
test: install-dev
	@echo "Running tests with hatch..."
	@uv run hatch run cov

# Install uv and uvx
install-uv:
	@echo "Installing uv and uvx..."
	@curl -LsSf https://astral.sh/uv/install.sh | sh

# Help target
help:
	@echo "Available targets:"
	@echo "  venv        - Create virtual environment using uv"
	@echo "  install     - Install dependencies from pyproject.toml"
	@echo "  install-dev - Install development dependencies (black and hatch)"
	@echo "  install-uv  - Install uv and uvx"
	@echo "  lint        - Run linting with black"
	@echo "  test        - Run tests with hatch"
	@echo "  clean       - Clean up virtual environment and build artifacts"
	@echo "  all         - Create venv and install dependencies (default)"
	@echo "  help        - Show this help message"