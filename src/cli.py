#!/usr/bin/env python3
"""
CLI entry point: run a .mthds bundle as a CrewAI crew with Ollama.

Usage:
    python -m src.cli bundles/research.mthds
    python -m src.cli bundles/research.mthds --ollama-url http://localhost:11434
    python -m src.cli bundles/research.mthds --input "What causes aurora borealis?"
"""

from __future__ import annotations

import argparse
import json
import sys

from .runtime import build_crew, load_bundle


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a .mthds bundle as a CrewAI crew with Ollama inference.",
    )
    parser.add_argument(
        "bundle",
        help="Path to a .mthds bundle file",
    )
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
        help="Base URL for Ollama API (default: http://localhost:11434)",
    )
    parser.add_argument(
        "--input", "-i",
        default=None,
        help="Input text to pass to the first agent (e.g. a research question)",
    )
    parser.add_argument(
        "--inputs-json",
        default=None,
        help="JSON string of key-value inputs to pass to the crew",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse the bundle and show the crew structure without executing",
    )

    args = parser.parse_args()

    # ── Load bundle ──
    print(f"\n📦 Loading bundle: {args.bundle}")
    bundle = load_bundle(args.bundle)
    print(f"   Domain:      {bundle.domain}")
    print(f"   Description: {bundle.description}")
    print(f"   Main pipe:   {bundle.main_pipe}")
    print(f"   Concepts:    {len(bundle.concepts)}")
    print(f"   Pipes:       {len(bundle.pipes)}")

    # ── Build crew ──
    crew = build_crew(bundle, base_url=args.ollama_url)
    print(f"\n🤖 Crew assembled: {len(crew.agents)} agents, {len(crew.tasks)} tasks")

    for i, agent in enumerate(crew.agents, 1):
        print(f"   Agent {i}: {agent.role}")
        print(f"            LLM: {agent.llm.model}  temp={agent.llm.temperature}")  # type: ignore[attr-defined]

    if args.dry_run:
        print("\n✅ Dry run complete — no execution.")
        return

    # ── Prepare inputs ──
    inputs: dict[str, str] = {}
    if args.inputs_json:
        inputs = json.loads(args.inputs_json)
    elif args.input:
        # Guess the first pipe's first input key
        from .runtime import resolve_execution_order
        pipe_codes = resolve_execution_order(bundle)
        if pipe_codes:
            first_pipe = bundle.pipes[pipe_codes[0]]
            if first_pipe.inputs:
                first_key = next(iter(first_pipe.inputs))
                inputs[first_key] = args.input

    if not inputs:
        print("\n⚠️  No inputs provided. Use --input or --inputs-json.")
        print("   Example: --input 'What are the latest trends in AI?'")
        sys.exit(1)

    print(f"\n🚀 Running crew with inputs: {list(inputs.keys())}")
    print("─" * 60)

    # ── Execute ──
    result = crew.kickoff(inputs=inputs)

    print("─" * 60)
    print(f"\n✅ Final output:\n\n{result}")


if __name__ == "__main__":
    main()
