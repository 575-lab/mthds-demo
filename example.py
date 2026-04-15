#!/usr/bin/env python3
"""
Example: use the runtime as a library to load a bundle,
inspect its structure, and run the crew programmatically.
"""

from src.runtime import build_crew, load_bundle, resolve_execution_order


def main() -> None:
    # ── Load any .mthds bundle ──
    bundle = load_bundle("bundles/research.mthds")

    print("=" * 60)
    print(f"  Domain:      {bundle.domain}")
    print(f"  Description: {bundle.description}")
    print(f"  Main pipe:   {bundle.main_pipe}")
    print("=" * 60)

    # ── Inspect concepts ──
    print("\n📚 Concepts:")
    for name, defn in bundle.concepts.items():
        desc = defn.get("description", "")
        structure = defn.get("structure")
        print(f"   • {name}: {desc}")
        if structure:
            for field_name, field_def in structure.items():
                print(f"     └─ {field_name}: {field_def.get('description', '')}")

    # ── Inspect pipes ──
    print("\n🔧 Pipes:")
    for code, pipe in bundle.pipes.items():
        inputs_str = ", ".join(
            f"{k}: {v}" for k, v in pipe.inputs.items()
        ) if pipe.inputs else "none"
        print(f"   • {code}")
        print(f"     Type:   {pipe.pipe_type}")
        print(f"     Inputs: {inputs_str}")
        print(f"     Output: {pipe.output}")
        if pipe.pipe_type == "PipeLLM":
            print(f"     Model:  {pipe.model_handle} (temp={pipe.temperature})")

    # ── Show execution order ──
    order = resolve_execution_order(bundle)
    print(f"\n🔄 Execution order: {' → '.join(order)}")

    # ── Build crew (dry run) ──
    crew = build_crew(bundle)
    print(f"\n🤖 Crew: {len(crew.agents)} agents, {len(crew.tasks)} tasks")
    for i, agent in enumerate(crew.agents, 1):
        print(f"   Agent {i}: {agent.role}")

    # ── Uncomment to actually run (requires Ollama running) ──
    # result = crew.kickoff(inputs={
    #     "question": "What are the environmental impacts of deep-sea mining?"
    # })
    # print(f"\n✅ Result:\n{result}")


if __name__ == "__main__":
    main()
