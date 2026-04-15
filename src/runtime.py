"""
MTHDS → CrewAI + Ollama Runtime
================================

Parses .mthds TOML bundles and builds executable CrewAI crews
that use Ollama for local model inference.

Key mappings:
    .mthds Construct    →  CrewAI Equivalent
    ─────────────────────────────────────────
    PipeLLM             →  Agent + Task
    PipeSequence        →  Process.sequential
    PipeBatch           →  Iterated sub-agent
    model (inline)      →  Ollama(model=...)
    system_prompt       →  Agent.backstory
    Concept             →  Task.expected_output
    Working Memory      →  Task context passing
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # type: ignore[no-redef]

from crewai import Agent, Crew, LLM, Process, Task


# ──────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────


@dataclass
class ParsedPipe:
    """A single pipe extracted from a .mthds bundle."""

    code: str
    pipe_type: str
    description: str
    prompt: str = ""
    system_prompt: str = ""
    output: str = "Text"
    inputs: dict[str, str] = field(default_factory=dict)
    model_handle: str = "llama3.1"
    temperature: float = 0.7
    # Controller-specific
    steps: list[dict[str, str]] = field(default_factory=list)
    branch_pipe_code: str = ""
    input_list_name: str = ""
    input_item_name: str = ""


@dataclass
class ParsedBundle:
    """A fully parsed .mthds bundle."""

    domain: str
    description: str
    main_pipe: str
    concepts: dict[str, dict[str, Any]]
    pipes: dict[str, ParsedPipe]
    bundle_system_prompt: str = ""


# ──────────────────────────────────────────────────────────────
# TOML parsing helpers
# ──────────────────────────────────────────────────────────────


def _extract_model_handle(model_field: Any) -> str:
    """Extract the Ollama model name from a model reference.

    Supports:
      - Inline settings table: { model = "ollama/llama3.1", temperature = 0.3 }
      - Bare handle string:    "ollama/llama3.1"
      - Preset / alias:        "$writing-factual" → falls back to default
    """
    if isinstance(model_field, dict):
        raw = model_field.get("model", "llama3.1")
    elif isinstance(model_field, str):
        raw = model_field
    else:
        return "llama3.1"

    # Strip provider prefix (ollama/)
    raw = raw.removeprefix("ollama/")

    # If it starts with $ @ ~ it's a preset/alias/waterfall — use default
    if raw and raw[0] in ("$", "@", "~"):
        return "llama3.1"

    return raw


def _extract_temperature(model_field: Any) -> float:
    """Pull temperature from inline model settings, default 0.7."""
    if isinstance(model_field, dict):
        return float(model_field.get("temperature", 0.7))
    return 0.7


# ──────────────────────────────────────────────────────────────
# Bundle loader
# ──────────────────────────────────────────────────────────────


def load_bundle(path: str | Path) -> ParsedBundle:
    """Load and parse a .mthds TOML bundle from disk."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Bundle not found: {path}")

    with open(path, "rb") as f:
        raw = tomllib.load(f)

    bundle_system_prompt = raw.get("system_prompt", "")

    # Parse concepts
    concepts: dict[str, dict[str, Any]] = {}
    for key, value in raw.get("concept", {}).items():
        if isinstance(value, str):
            # Shorthand: ConceptName = "description"
            concepts[key] = {"description": value}
        elif isinstance(value, dict):
            concepts[key] = value

    # Parse pipes
    pipes: dict[str, ParsedPipe] = {}
    for code, pipe_raw in raw.get("pipe", {}).items():
        model_field = pipe_raw.get("model")
        pipe_system_prompt = pipe_raw.get("system_prompt", "")
        # Inherit bundle-level system_prompt if the pipe doesn't define its own
        if not pipe_system_prompt and bundle_system_prompt:
            pipe_system_prompt = bundle_system_prompt

        parsed = ParsedPipe(
            code=code,
            pipe_type=pipe_raw.get("type", ""),
            description=pipe_raw.get("description", ""),
            prompt=pipe_raw.get("prompt", ""),
            system_prompt=pipe_system_prompt,
            output=pipe_raw.get("output", "Text"),
            inputs=pipe_raw.get("inputs", {}),
            model_handle=_extract_model_handle(model_field) if model_field else "llama3.1",
            temperature=_extract_temperature(model_field) if model_field else 0.7,
            steps=pipe_raw.get("steps", []),
            branch_pipe_code=pipe_raw.get("branch_pipe_code", ""),
            input_list_name=pipe_raw.get("input_list_name", ""),
            input_item_name=pipe_raw.get("input_item_name", ""),
        )
        pipes[code] = parsed

    return ParsedBundle(
        domain=raw.get("domain", "unknown"),
        description=raw.get("description", ""),
        main_pipe=raw.get("main_pipe", ""),
        concepts=concepts,
        pipes=pipes,
        bundle_system_prompt=bundle_system_prompt,
    )


# ──────────────────────────────────────────────────────────────
# CrewAI builders
# ──────────────────────────────────────────────────────────────

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"


def pipe_to_agent(
    pipe: ParsedPipe,
    base_url: str = DEFAULT_OLLAMA_BASE_URL,
) -> Agent:
    """Convert a PipeLLM pipe into a CrewAI Agent.

    - pipe.system_prompt  → agent.backstory
    - pipe.description    → agent.goal
    - pipe.model_handle   → Ollama model name
    - pipe.temperature    → Ollama temperature
    """
    return Agent(
        role=pipe.code.replace("_", " ").title(),
        goal=pipe.description,
        backstory=pipe.system_prompt or "You are a helpful AI assistant.",
        llm=LLM(
            model=f"ollama/{pipe.model_handle}",
            temperature=pipe.temperature,
            base_url=base_url,
        ),
        verbose=True,
    )


def pipe_to_task(
    pipe: ParsedPipe,
    agent: Agent,
    kickoff_keys: set[str] | None = None,
) -> Task:
    """Convert a PipeLLM pipe into a CrewAI Task.

    - pipe.prompt   → task.description
    - pipe.output   → task.expected_output

    Only @var references whose names appear in *kickoff_keys* are converted
    to CrewAI ``{var}`` interpolation.  All other references are left as
    plain text so CrewAI does not try to resolve them from kickoff inputs
    (they arrive via sequential task context instead).
    """
    description = pipe.prompt if pipe.prompt else pipe.description
    kickoff_keys = kickoff_keys or set()

    def _replace_var(m: re.Match[str]) -> str:
        var_name = m.group(1)
        if var_name in kickoff_keys:
            return f"{{{var_name}}}"
        return var_name  # leave as readable plain text

    description = re.sub(r"[@$](\w+)(?:\.\w+)*", _replace_var, description)
    return Task(
        description=description,
        expected_output=f"Produce output of type: {pipe.output}",
        agent=agent,
    )


# ──────────────────────────────────────────────────────────────
# Crew assembly
# ──────────────────────────────────────────────────────────────


def resolve_execution_order(bundle: ParsedBundle) -> list[str]:
    """Walk the main_pipe's PipeSequence steps and resolve to an
    ordered list of PipeLLM pipe codes.

    For PipeBatch steps, the branch_pipe_code (the actual LLM
    operator) is used instead.
    """
    main = bundle.pipes.get(bundle.main_pipe)
    if not main or main.pipe_type != "PipeSequence":
        # Fallback: return all PipeLLM pipes in definition order
        return [
            code
            for code, p in bundle.pipes.items()
            if p.pipe_type == "PipeLLM"
        ]

    ordered: list[str] = []
    for step in main.steps:
        step_code = step.get("pipe", "")
        step_pipe = bundle.pipes.get(step_code)
        if not step_pipe:
            continue

        if step_pipe.pipe_type == "PipeBatch":
            branch = step_pipe.branch_pipe_code
            if branch and branch in bundle.pipes:
                ordered.append(branch)
        elif step_pipe.pipe_type == "PipeLLM":
            ordered.append(step_code)
        # PipeSequence nested inside another is not handled here
        # for simplicity — extend as needed.

    return ordered


def build_crew(
    bundle: ParsedBundle,
    base_url: str = DEFAULT_OLLAMA_BASE_URL,
) -> Crew:
    """Build a full CrewAI Crew from a parsed .mthds bundle.

    Each PipeLLM pipe becomes an Agent + Task pair.
    The PipeSequence controller determines execution order.
    """
    pipe_codes = resolve_execution_order(bundle)

    # Only the first pipe's input keys are provided via kickoff()
    first_pipe = bundle.pipes[pipe_codes[0]] if pipe_codes else None
    kickoff_keys = set(first_pipe.inputs) if first_pipe else set()

    agents: list[Agent] = []
    tasks: list[Task] = []

    for code in pipe_codes:
        pipe = bundle.pipes[code]
        agent = pipe_to_agent(pipe, base_url=base_url)
        task = pipe_to_task(pipe, agent, kickoff_keys=kickoff_keys)
        agents.append(agent)
        tasks.append(task)

    return Crew(
        agents=agents,
        tasks=tasks,
        process=Process.sequential,
        verbose=True,
    )
