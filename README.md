# MTHDS + CrewAI + Ollama

Run [MTHDS](https://github.com/mthds-ai/mthds) bundles as multi-agent [CrewAI](https://www.crewai.com/) crews with local [Ollama](https://ollama.ai/) inference. No API keys required.

## What this does

MTHDS is a declarative language for defining typed AI pipelines in `.mthds` files (TOML-based). This project provides a Python runtime that:

1. **Parses** `.mthds` bundles — extracting concepts, pipes, model references, and execution order
2. **Maps** MTHDS constructs to CrewAI primitives — each `PipeLLM` becomes an Agent + Task, `PipeSequence` becomes sequential execution
3. **Runs** the crew using Ollama for model inference — model names and temperatures come straight from the bundle

### Mapping reference

| MTHDS Construct | CrewAI Equivalent | Notes |
|---|---|---|
| `PipeLLM` | `Agent` + `Task` | `system_prompt` → backstory |
| `PipeSequence` | `Process.sequential` | `steps` → task order |
| `PipeBatch` | Iterated sub-agent | `branch_pipe_code` → the actual LLM agent |
| `model` (inline) | `LLM(model=...)` | Temperature preserved |
| `Concept` | `expected_output` | Type hint for the task |
| Working Memory | Task context | Results pass between steps |

## Quick start

### 1. Install Ollama

```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

### 2. Pull a model

```bash
ollama pull gemma4:e2b
```

### 3. Install Python dependencies

```bash
uv pip install .
```

### 4. Run a bundle

**Quick start script:**

```bash
./run.sh
```

**CLI:**

```bash
# Research pipeline
uv run python -m src.cli bundles/research.mthds \
  --input "What are the latest developments in quantum computing?"

# Content creation
uv run python -m src.cli bundles/content.mthds \
  --input "Write an article about sustainable energy for a general audience"

# Code review
uv run python -m src.cli bundles/code_review.mthds \
  --input "def login(user, pw): return db.query(f'SELECT * FROM users WHERE name={user} AND pass={pw}')"
```

**Dry run** (inspect the crew without executing):

```bash
uv run python -m src.cli bundles/research.mthds --dry-run
```

**As a library:**

```python
from src.runtime import load_bundle, build_crew

bundle = load_bundle("bundles/research.mthds")
crew = build_crew(bundle)
result = crew.kickoff(inputs={
    "question": "What causes aurora borealis?"
})
print(result)
```

## Example bundles

### `bundles/research.mthds` — Research pipeline

Three agents in sequence:
- **Researcher** — gathers 3 source summaries (gemma4:e2b, temp=0.3)
- **Fact-checker** — verifies claims with confidence scores (gemma4:e2b, temp=0.1)
- **Writer** — synthesizes a structured research brief (gemma4:e2b, temp=0.5)

### `bundles/content.mthds` — Content creation

Three agents in sequence:
- **Writer** — produces a first draft from a content brief (gemma4:e2b, temp=0.7)
- **Editor** — gives structured editorial feedback with scores (gemma4:e2b, temp=0.2)
- **Copy editor** — polishes the draft using feedback (gemma4:e2b, temp=0.4)

### `bundles/code_review.mthds` — Code review

Three agents in sequence:
- **Security scanner** — OWASP-focused vulnerability audit (gemma4:e2b, temp=0.1)
- **Quality checker** — readability, SOLID, testability review (gemma4:e2b, temp=0.2)
- **Synthesizer** — combines findings into a prioritized action plan (gemma4:e2b, temp=0.3)

## Using different models

Change the `model` field in any `.mthds` bundle to swap models:

```toml
# Use Phi-3 instead of Llama
model = { model = "ollama/phi3", temperature = 0.3 }

# Use Gemma 2
model = { model = "ollama/gemma2", temperature = 0.5 }

# Bare handle (defaults to temp=0.7)
model = "ollama/mistral"
```

The runtime passes the model string to CrewAI's `LLM` class, which routes through Ollama.

## Project structure

```
mthds-crewai-ollama/
├── bundles/
│   ├── research.mthds       # Research pipeline bundle
│   ├── content.mthds        # Content creation bundle
│   └── code_review.mthds    # Code review bundle
├── src/
│   ├── __init__.py
│   ├── runtime.py            # Core: parser + CrewAI builder
│   └── cli.py                # CLI entry point
├── example.py                # Programmatic usage example
├── run.sh                    # Quick start script
├── pyproject.toml
└── README.md
```

## Requirements

- Python >= 3.10
- Ollama running locally (default: `http://localhost:11434`)
- At least one Ollama model pulled

## License

Apache-2.0
