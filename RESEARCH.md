# RESEARCH.md — positioning, audience, distribution

This file keeps the marketing constraint visible during the build. If a
technical decision makes the artifact less legible to the audience below,
that's a cost — weigh it.

## 1. The pitch (canonical forms)

- **One sentence:** I taught a small open model to run drone operations —
  shifting wind, pop-up flight restrictions, cascading failures — and it
  learned to stop crashing.
- **One paragraph (README/blog opener):** Drones already fly themselves.
  Autopilots hold trajectories, failsafes trigger RTL — none of that needs an
  LLM. What isn't automated is the chair: the remote pilot in command making
  contextual tradeoffs over semantic inputs — a NOTAM, a battery anomaly, a
  gust front arriving early — where the rulebook answer is wrong or ambiguous.
  `uav-operator` is an RL environment for exactly that layer: the model is
  the ops center, the sim owns the aircraft, and every reward is computed
  from physics, not prose.
- **Positioning sentence (vs prior art):** MultiUAV-Plat and UAVBench built
  the exam; this is the gym.

## 2. Prior art map (engage by name — this audience reads arXiv daily)

| Work | What it is | Delta |
|---|---|---|
| MultiUAV-Plat (arXiv 2606.31073, ~1wk old) | LLM-oriented multi-UAV *eval platform*: REST APIs, partial obs, hidden validation, 75 sessions/1500 tasks; task planning (coverage/search/assignment) | Not trainable (platform, not env); no reward design for RL; no operator-judgment tasks; not verifiers/Hub. Cite their ReAct-baseline ~31% pass rate as evidence of headroom. Credit generously. |
| UAVBench (arXiv 2511.11252) | 50k LLM-generated flight scenarios, JSON schema, risk labels | Static dataset, non-interactive. Potential future import as scenario seeds — mention as future work. |
| AeroGen / UAV-CodeAgents | Single-shot LLM→drone-SDK codegen | Open-loop; no mid-mission judgment; different layer. |
| AirSim / Flightmare / gym-pybullet-drones | Physics/control simulators | Wrong abstraction layer entirely — cite to *explain* the layer model, not as competitors. |
| Hub prior art | No UAV/aviation/ops-dispatch env found (checked July 2026) | "First UAV-operations environment on the Hub" — verify again on Day 13 before claiming in print. |

## 3. Audience + revealed quality bar (from PI's own signals)

Featured-env pattern: multi-turn, tool-using, sandboxed/agentic (opencode-
science, deepdive, mini-swe-agent-plus) — not benchmark wraps. Their moonshot
list = inventing verification for the unverified. Their tech report calls for
"real-world workloads." Bounty pricing: wraps $100–500; complex scope
$1k–5k+. Build to the featured/application-only tier or don't bother.
INTELLECT-3 trained on a small slice of the Hub → "training-grade" is the
scarce property. The four properties to hit: training-grade, invented
verification, real-world workload, adversarially hardened (+ watchable,
+ one-sentence-compressible).

## 4. Authority notes (what to claim, what not to)

- CLAIM: FAA Part 107; commanded real UAS ops in live wind (first drone-based
  cloud-seeding operations in the US, Rainmaker — public record). Failsafe/
  TFR/battery-margin realism comes from operating experience. This is the
  writeup's authenticity layer and the anti-"is this realistic?" shield.
- DO NOT: import any Infinrg material (per CLAUDE.md §6); overclaim fidelity
  ("simplified airspace geometry, real structure" — say it plainly); claim
  "first LLM UAV benchmark" (false — MultiUAV-Plat et al.); pretend the model
  flies the aircraft (the layer-model argument is the credibility core —
  sophisticated readers WILL probe exactly this, because it was our own first
  objection).

## 5. Distribution sequence (Day 14)

1. Blog post live (own site or HF community blog) — the deep artifact.
2. Thread, SF morning. Outline: (1) hook clip — TFR blooms mid-flight, model
   reroutes, 5-sec; (2) "drones fly themselves; here's the layer that
   doesn't" — layer model in 2 tweets; (3) reward-from-physics + best hack
   screenshot ("here's GPT-X talking its way around a TFR — and why that
   scored zero"); (4) training curve + before/after pair, same seed; (5) built
   on verifiers/prime-rl/PI compute end-to-end, link Hub listing; (6) what's
   next (fleet tier) + blog link.
3. DMs to Will Brown (@willccbb) + Johannes (X), same day, separately worded,
   one sentence + Hub link. Shape: "Built the first UAV-operations environment
   for the Hub — ops-center judgment layer, physics-verified rewards, trained
   Qwen3-4B on PI compute, red-team writeup included. [link]" No ask. The
   artifact is the ask.
4. Same-day replies to every substantive response. If PI engages, offer the
   fleet-tier RFC as the natural "we've already worked together" continuation.
5. Secondary surfaces (Day 15+): PI Discord #environments, r/LocalLLaMA if
   thread lands, HN only if blog post stands alone technically.

## 6. Reference shelf

- verifiers docs: github.com/PrimeIntellect-ai/verifiers (docs/environments.md
  for MultiTurnEnv/Rubric/stop conditions; pyproject + hatch include rules)
- prime-rl: github.com/PrimeIntellect-ai/prime-rl (small-run guides: Wordle
  2–4×H100 multi-turn example is the closest template to our run)
- Hub docs: docs.primeintellect.ai/tutorials-environments/environments
- Env walkthrough (community): huggingface.co/blog/anakin87/environments-hub
- INTELLECT-3 tech report: arXiv 2512.16144 (EnvGroup pattern; "real-world
  workloads" quote)
- MultiUAV-Plat: arXiv 2606.31073 · UAVBench: arXiv 2511.11252
- PI environments program + bounties: primeintellect.ai/blog/environments,
  /blog/scaling-environments-program