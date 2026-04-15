#!/usr/bin/env bash
set -euo pipefail

uv pip install .
ollama pull gemma4:e2b
uv run python -m src.cli bundles/research.mthds \
  --input "What are the latest trends in AI?"
