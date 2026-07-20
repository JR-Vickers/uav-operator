#!/usr/bin/env python3
"""Generate a precise Prime support request from a captured Laguna XS run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAPTURE = REPO_ROOT / "assets" / "training" / "day8_laguna_t1_retry.json"
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "LAGUNA_XS_PLATFORM_PROBE.md"
MODEL = "poolside/Laguna-XS-2.1"
ENVIRONMENT = "jarrett/uav-operator@0.1.1"


def _nested(value: object, *keys: str | int) -> object | None:
    current = value
    for key in keys:
        if isinstance(current, dict) and isinstance(key, str):
            current = current.get(key)
        elif isinstance(current, list) and isinstance(key, int) and 0 <= key < len(current):
            current = current[key]
        else:
            return None
    return current


def build_escalation_request(capture: dict[str, Any]) -> str:
    """Return a support request containing only captured facts and probes."""

    run_id = capture.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("capture must contain a non-empty run_id")
    model = _nested(capture, "run", "run", "base_model")
    if model != MODEL:
        raise ValueError(f"capture base model must be {MODEL!r}")
    environment = _nested(capture, "run", "run", "environments", 0, "id")
    if environment != "jarrett/uav-operator":
        raise ValueError("capture must identify the uav-operator environment")

    latest_step = _nested(capture, "summary", "latest_step")
    training_tokens = _nested(capture, "summary", "training_tokens")
    model_errors = _nested(capture, "summary", "platform_model_errors")
    total_cost = _nested(capture, "usage", "total_cost_usd")
    input_tokens = _nested(capture, "usage", "inference", "input_tokens")
    output_tokens = _nested(capture, "usage", "inference", "output_tokens")
    components = "Both `uav-operator` and `eval-uav-operator` env-server components succeeded."

    return f"""# Laguna XS Hosted Training platform probe

## Captured failure

- Hosted Training run: `{run_id}`
- Model: `{MODEL}`
- Environment: `{ENVIRONMENT}`
- Latest optimizer step: `{latest_step}`
- Training tokens: `{training_tokens}`
- Dispatcher-level `ModelError` count: `{model_errors}`
- Inference tokens: `{input_tokens}` input / `{output_tokens}` output
- Reported total cost: `${total_cost}`
- {components}

The run produced no sampled rollouts, reward distributions, checkpoint, or
adapter. The public logs expose only generic `ModelError`; they do not contain
the underlying provider exception, request ID, or stack trace.

## Requested server-side probes

Please run these probes on the *same Hosted Training policy-inference image*
and report the underlying exception plus a correlation/request ID for each:

1. `poolside/Laguna-XS-2.1`: a plain one-message text completion.
2. The same chat completion with `enable_thinking=false`.
3. The same request with one trivial JSON tool definition.
4. If probe 3 fails, repeat with the serialized tool schema and system prompt
   from this run's saved rollout request.

The first failing probe distinguishes model loading/architecture support from
chat-template and tool-parser compatibility. Please do not substitute ordinary
Prime Inference for Hosted Training: that catalog has a different model entry.

## Deployment details requested

Please identify the deployed model and tokenizer revisions, inference engine
and version, renderer/chat-template commit, tool-parser version, and whether
`trust_remote_code` is enabled. Hosted Training exposes none of these controls
to the TOML configuration, so this cannot be pinned or tested from the
environment repository.

## Control matrix after the root exception is available

| Model | Simple known-good hosted environment | `{ENVIRONMENT}` |
| --- | --- | --- |
| `{MODEL}` | isolates model/deployment | reproduces current failure |
| Known-good, zero-cost Hosted model | isolates environment path | confirms environment path |

Run only zero-cost or explicitly approved matrix cells. Do not change simulator
logic, rewards, dataset seeds, or tool semantics to work around an unclassified
provider failure.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    capture = json.loads(args.capture.read_text())
    if not isinstance(capture, dict):
        raise ValueError("capture must be a JSON object")
    request = build_escalation_request(capture)
    args.output.write_text(request)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
