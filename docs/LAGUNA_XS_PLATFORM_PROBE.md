# Laguna XS Hosted Training platform probe

## Captured failure

- Hosted Training run: `txkxxxl1404r694dcmw2rzkh`
- Model: `poolside/Laguna-XS-2.1`
- Environment: `jarrett/uav-operator@0.1.1`
- Latest optimizer step: `0`
- Training tokens: `0`
- Dispatcher-level `ModelError` count: `200`
- Inference tokens: `735296` input / `3424` output
- Reported total cost: `$0.0`
- Both `uav-operator` and `eval-uav-operator` env-server components succeeded.

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

| Model | Simple known-good hosted environment | `jarrett/uav-operator@0.1.1` |
| --- | --- | --- |
| `poolside/Laguna-XS-2.1` | isolates model/deployment | reproduces current failure |
| Known-good, zero-cost Hosted model | isolates environment path | confirms environment path |

Run only zero-cost or explicitly approved matrix cells. Do not change simulator
logic, rewards, dataset seeds, or tool semantics to work around an unclassified
provider failure.
