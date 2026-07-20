#!/usr/bin/env python3
"""Validate the UAV prompt and tools against Laguna XS's official tokenizer.

This downloads tokenizer/configuration files only. It never downloads model
weights, calls a model endpoint, or launches Hosted Training.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from typing import Any

import uav_operator


MODEL = "poolside/Laguna-XS-2.1"
MAX_HOSTED_SEQUENCE_TOKENS = 65_536


def validate_tool_defs(tool_defs: Sequence[Mapping[str, Any]]) -> None:
    """Check invariants required by OpenAI-style function tool definitions."""

    names: set[str] = set()
    for tool in tool_defs:
        name = tool.get("name")
        parameters = tool.get("parameters")
        if not isinstance(name, str) or not name:
            raise ValueError("tool has no non-empty name")
        if name in names:
            raise ValueError(f"duplicate tool name: {name}")
        names.add(name)
        if not isinstance(parameters, Mapping) or parameters.get("type") != "object":
            raise ValueError(f"tool {name} does not have an object parameter schema")
        json.dumps(tool, allow_nan=False)


def initial_messages() -> list[dict[str, Any]]:
    """Return the exact initial system and T1 task messages from the environment."""

    environment = uav_operator.load_environment(
        tier="T1", dataset_split="train", max_examples=1, max_turns=40
    )
    row = next(iter(environment.get_dataset()))
    question = row.get("question")
    if not isinstance(question, str) or not question:
        raise ValueError("T1 dataset row has no non-empty question")
    return [
        {"role": "system", "content": uav_operator.SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


def _token_count(rendered: object) -> int:
    if not isinstance(rendered, Mapping):
        raise RuntimeError("Laguna tokenizer did not return a mapping")
    input_ids = rendered.get("input_ids")
    if input_ids is None:
        raise RuntimeError("Laguna tokenizer did not return input_ids")
    return len(input_ids[0]) if getattr(input_ids, "ndim", 1) > 1 else len(input_ids)


def render_initial_prompt() -> dict[str, Any]:
    """Render the initial prompt using the official Laguna tokenizer template."""

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Install an ephemeral compatible tokenizer dependency with: "
            "uv run --with 'transformers>=5.7.0' scripts/laguna_local_smoke.py"
        ) from exc

    tool_defs = uav_operator._tool_defs()
    validate_tool_defs(tool_defs)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    messages = initial_messages()
    rendered = tokenizer.apply_chat_template(
        messages,
        tools=list(tool_defs),
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
    )
    token_count = _token_count(rendered)
    tool_round_trip = [
        *messages,
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "id": "local-smoke-1",
                    # Laguna's native chat template expects a structured object.
                    # The renderer must bridge this from OpenAI's JSON-string form.
                    "function": {"name": "get_telemetry", "arguments": {}},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "local-smoke-1",
            "content": '{"ok":true,"smoke":"tool_result"}',
        },
    ]
    round_trip_tokens = _token_count(
        tokenizer.apply_chat_template(
            tool_round_trip,
            tools=list(tool_defs),
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
        )
    )
    if max(token_count, round_trip_tokens) >= MAX_HOSTED_SEQUENCE_TOKENS:
        raise RuntimeError(
            f"rendered prompt has {max(token_count, round_trip_tokens)} tokens, exceeding Hosted Training's "
            f"{MAX_HOSTED_SEQUENCE_TOKENS}-token sequence limit"
        )
    result: dict[str, Any] = {
        "kind": "local_laguna_tokenizer_smoke",
        "model": MODEL,
        "tool_count": len(tool_defs),
        "tool_names": [tool["name"] for tool in tool_defs],
        "initial_prompt_tokens": token_count,
        "synthetic_tool_round_trip_tokens": round_trip_tokens,
        "hosted_sequence_limit": MAX_HOSTED_SEQUENCE_TOKENS,
        "passed": True,
    }
    try:
        from renderers import LagunaXS21Renderer, LagunaXS21RendererConfig
    except ImportError:
        result["public_renderer_smoke"] = "not_installed"
        return result

    renderer = LagunaXS21Renderer(tokenizer, LagunaXS21RendererConfig())
    renderer_round_trip = [
        *messages,
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "id": "local-smoke-1",
                    # This matches the OpenAI-style string passed by verifiers.
                    "function": {"name": "get_telemetry", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "local-smoke-1",
            "content": '{"ok":true,"smoke":"tool_result"}',
        },
    ]
    rendered_by_renderer = renderer.render(
        renderer_round_trip, tools=list(tool_defs), add_generation_prompt=True
    )
    result["public_renderer_smoke"] = "passed"
    result["public_renderer_tool_round_trip_tokens"] = len(
        rendered_by_renderer.token_ids
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit JSON only.")
    args = parser.parse_args()
    result = render_initial_prompt()
    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
