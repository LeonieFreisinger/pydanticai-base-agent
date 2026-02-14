# SQL Agent Showcase - Makefile
# Quick commands for setup and running the demo

.PHONY: setup run run-level1 run-level2 run-level3 run-level4 clean db test lint

# Default target
help:
	@echo "SQL Agent Showcase - Available commands:"
	@echo ""
	@echo "  make setup       - Install dependencies and initialize database"
	@echo "  make run         - Run the Chainlit app (uses DEMO_LEVEL env var)"
	@echo "  make run-level1  - Run with DEMO_LEVEL=1 (structured output only)"
	@echo "  make run-level2  - Run with DEMO_LEVEL=2 (schema exploration)"
	@echo "  make run-level3  - Run with DEMO_LEVEL=3 (full query execution)"
	@echo "  make run-level4  - Run with DEMO_LEVEL=4 (full + Langfuse)"
	@echo "  make db          - Reinitialize database with seed data"
	@echo "  make test        - Run tests"
	@echo "  make lint        - Run linter"
	@echo "  make clean       - Remove generated files"

# Setup: install deps and create database
setup:
	@echo "📦 Installing dependencies..."
	uv sync
	@echo ""
	@echo "🗄️ Initializing database..."
	uv run python database/init_db.py
	@echo ""
	@echo "✅ Setup complete! Run 'make run' to start the app."
	@echo "   Don't forget to set OPENAI_API_KEY in .env"

# Run with current DEMO_LEVEL (default: 4)
run:
	@echo "🚀 Starting SQL Agent Showcase..."
	unset VIRTUAL_ENV && uv run chainlit run app.py

# Run with specific demo levels
run-level1:
	@echo "🚀 Starting with DEMO_LEVEL=1 (Structured Output Only)..."
	unset VIRTUAL_ENV && DEMO_LEVEL=1 uv run chainlit run app.py

run-level2:
	@echo "🚀 Starting with DEMO_LEVEL=2 (Schema Exploration)..."
	unset VIRTUAL_ENV && DEMO_LEVEL=2 uv run chainlit run app.py

run-level3:
	@echo "🚀 Starting with DEMO_LEVEL=3 (Full Query Execution)..."
	unset VIRTUAL_ENV && DEMO_LEVEL=3 uv run chainlit run app.py

run-level4:
	@echo "🚀 Starting with DEMO_LEVEL=4 (Full + Langfuse)..."
	unset VIRTUAL_ENV && DEMO_LEVEL=4 uv run chainlit run app.py

# Reinitialize database
db:
	@echo "🗄️ Reinitializing database..."
	uv run python database/init_db.py

# Run tests
test:
	@echo "🧪 Running tests..."
	ENABLE_LANGFUSE=false uv run pytest -v

# Lint
lint:
	@echo "🔍 Running linter..."
	uv run ruff check .
	uv run ruff format --check .

# Format code
format:
	@echo "✨ Formatting code..."
	uv run ruff format .

# Clean generated files
clean:
	@echo "🧹 Cleaning up..."
	rm -f showcase.db
	rm -rf __pycache__ .pytest_cache .ruff_cache
	rm -rf agent/__pycache__ database/__pycache__
	find . -name "*.pyc" -delete
	@echo "✅ Clean complete"
