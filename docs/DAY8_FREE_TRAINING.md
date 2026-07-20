# Day 8 free Hosted Training report

## Outcome

Day 8 is **blocked by the free Hosted Training catalog/eligibility policy**.
The environment/tool loop remains proven by eval `6d420fc1`, but no free
Hosted Training candidate completed an optimizer step.

The user manually attempted the one-step
`sprints/Llama-3.2-1B-Instruct` diagnostic after a preflight that confirmed the
model was available and its effective training, input, and output prices were
all exactly `$0/M`. The CLI repeated those free prices, confirmed the Hub
quality action, accepted interactive confirmation, and then received HTTP 400
before creating a run:

> Free-tier model `sprints/Llama-3.2-1B-Instruct`:
> `jarrett/uav-operator` does not meet the free-tier environment requirements.

No run ID exists. There were zero optimizer steps, training tokens,
checkpoints, adapters, or billing rows. The wallet remained `$57.9182` with 10
total billing rows. Evidence is in
`assets/training/day8_llama_1b_diagnostic.json` and the refreshed failing
preflight is in `assets/training/day8_llama_1b_preflight.json`.

## Preflight bug

The original preflight incorrectly treated these facts as sufficient:

- the exact Hosted model was available and all three effective prices were
  zero;
- public Hub version `0.1.1` had quality action `SUCCESS`;
- the wallet balance was available.

The backend applies a separate model/environment free-tier eligibility rule at
run creation. Prime CLI 0.6.16 does not expose that rule in `train models` or
`env status`. Its installed client contains a non-mutating
`/rft/runs/preview` method, but the live endpoint returned HTTP 405. Official
Hosted Training documentation describes published Hub environments and action
health as prerequisites but does not publish the promotional eligibility
criteria.

The preflight now fails closed unless eligibility is affirmatively exposed. It
also records this exact model/environment denial, so the same pairing cannot
produce another false green gate.

## Bounded failure decision

This is not a transient network, rate-limit, or capacity error and therefore
does not qualify for the single permitted unchanged retry. It is also not an
environment simulator, reward, seed, or packaging failure: the same published
wheel passed its Hub quality scan and the functional eval gate. The historical
Laguna runs remain separately documented in `docs/DAY8_SMOKE.md`.

The 50-step Llama smoke is not authorized. No paid model is permitted. Resume
only if Prime documents the free-tier environment requirements and this
environment satisfies them, or Prime explicitly enables the environment for a
currently free Hosted model. Any materially changed config must return through
the one-step diagnostic.

## Laguna compatibility follow-up

Separate platform tickets have been submitted for the Llama eligibility denial
and the Laguna XS policy-inference failure. They are awaiting a platform
response; ticket submission is not evidence that either condition has been
fixed and does not authorize a retry.

Upstream history makes a Laguna model/renderer revision mismatch a credible
hypothesis, but not a confirmed root cause. Poolside changed the Laguna XS 2.1
chat template on July 7 in
[`575f0f28`](https://huggingface.co/poolside/Laguna-XS-2.1/commit/575f0f288a8e61da7492135a678c52dc371a7936),
and Prime's
[`renderers` PR 99](https://github.com/PrimeIntellect-ai/renderers/pull/99)
rewrote the 2.1 renderer after the new template broke parity tests; its merge
commit was `4472b743`. The earlier support landed in
[`renderers` PR 97](https://github.com/PrimeIntellect-ai/renderers/pull/97).
The two failed Hosted runs had clean baseline evaluations followed by generic
`ModelError` failures on the training-policy path, so the available evidence
cannot establish which model revision, tokenizer revision, renderer commit, or
inference image that path used.

A rollback would be meaningful only if Prime pins a mutually compatible
model/tokenizer and renderer pair. Reverting this repository, the Hub
environment, or only one side of that pair cannot perform that repair and could
reintroduce the template mismatch. Hosted Training's documented configuration
accepts a model ID but exposes no model revision, tokenizer revision, renderer
commit, or inference-image control, so this is a platform-side action.

Before another Laguna diagnostic, obtain confirmation of a material deployment
change and, ideally, the exact model/tokenizer revision and `renderers` commit
used by the training-policy endpoint. A blind unchanged retry remains outside
the bounded retry rule. If Prime repairs or rolls back the endpoint, the next
attempt still starts at the committed one-step Laguna diagnostic with a fresh
free-price, capacity, wallet, and Hub preflight plus explicit user approval.

### 2026-07-19 retry result

After `poolside/Laguna-XS-2.1` remained listed in the live Hosted Training
catalog with zero effective training, input, and output prices, a user-launched
one-step diagnostic was run as `txkxxxl1404r694dcmw2rzkh`. It stopped at step
0 after roughly five minutes. The run recorded 200 dispatcher-level
`ModelError` failures, zero training tokens, no sampled rollouts, no reward
distributions, no checkpoint or adapter, and `$0.00` total reported cost. Both
environment-server components completed successfully; no environment exception
or more-specific provider stack trace was exposed. The captured run and fresh
preflight are `assets/training/day8_laguna_t1_retry.json` and
`assets/training/day8_laguna_t1_retry_preflight.json`.

This is a reproduction of the platform policy-inference failure, not evidence
that the model is trainable in this environment. Do not relaunch unchanged.
The ready-to-send platform probe request is in `docs/LAGUNA_XS_PLATFORM_PROBE.md`.

### Local prompt/renderer compatibility check

The environment-side interface was checked without model weights or another
Hosted Training run. Laguna's official tokenizer rendered the real initial T1
prompt plus all 17 tool definitions in 1,671 tokens, below the run's 65,536
token sequence limit. A synthetic OpenAI-style tool call (JSON-string
arguments) and tool result rendered in 1,699 tokens through the current public
Prime `renderers` checkout (`983901a`). The renderer explicitly converts those
JSON-string arguments to the structured form required by Laguna's native chat
template.

This rules out the current public renderer plus this environment's initial
prompt/tool schema as the observed failure. It does not prove the private
Hosted Training image used the same renderer/model/tokenizer revisions.

## Work while blocked

The active dependency plan is in `PLAN.md` under “Blocked-mode parallel plan.”
Training-independent evaluation tooling, writeup, media preparation, baseline
structure, and release hygiene may continue. Curriculum selection, trained
checkpoint evaluation, learning curves, before/after claims, and trained-model
red-team results remain blocked. Synthetic fixtures may validate tooling but
must be unmistakably labeled and cannot be published as experimental evidence.
