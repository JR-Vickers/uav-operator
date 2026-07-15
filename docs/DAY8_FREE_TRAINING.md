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
