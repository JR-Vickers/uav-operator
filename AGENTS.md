# AGENTS.md — uav-operator

RL environment for the Prime Intellect Environments Hub. An LLM acts as the
remote pilot in command (ops-center operator) of small UAS missions over the
San Francisco Bay Area. The autopilot is part of the simulator; the model is
only invoked at judgment points. See SPEC.md for design, PLAN.md for schedule,
RESEARCH.md for positioning and audience.

## The one sentence (do not lose sight of it)

"I taught a small open model to run drone operations through shifting wind,
pop-up flight restrictions, and cascading failures — watch it learn to stop
crashing." Every technical decision must serve either (a) the credibility of
this sentence to RL researchers, or (b) the 5-second visual proof of it.

## Project geometry

- This repo root is `environments/uav_operator/` inside a `prime lab setup`
  workspace. The workspace (one level up) provides prime CLI auth, endpoint
  configs, and agent skills. NEVER modify files above this repo root. NEVER
  commit anything from the workspace level.
- This repo has its own git remote (public GitHub). The Hub wheel ships ONLY
  what `[tool.hatch.build]` includes: `uav_operator.py` and `pyproject.toml`.
  Renderer, scripts, assets, docs stay GitHub-only.

## Hard constraints (non-negotiable)

1. **Verifiers spec compliance.** `load_environment(**kwargs) -> vf.Environment`
   is the only entry point. Subclass `vf.MultiTurnEnv`. The env must pass
   `prime eval run` / `vf-eval` at every commit after Day 1. If a change breaks
   `vf-eval uav-operator -m <model> -n 2 -r 1`, fixing that outranks all other work.
2. **Reward from sim state only.** No reward component may ever read, parse, or
   judge the model's natural-language text. Success = physics + logged sim
   trajectory. Model claims are inert. This is the anti-reward-hacking
   foundation; violating it invalidates the writeup's central argument.
3. **Seeded determinism.** `(seed, action_sequence) -> identical outcome`, bit
   for bit. All randomness flows from one `np.random.Generator` seeded per
   episode. No wall-clock time, no unseeded sampling. CI test enforces this.
4. **Event-driven sim, not timestep physics.** The sim advances decision-point
   to decision-point. We never integrate flight dynamics. The autopilot layer
   is an analytic model (segment times, energy burn, failsafe triggers). If
   you find yourself writing a physics integrator, stop — wrong layer.
5. **Lean wheel.** Package deps: `verifiers`, `numpy`, and stdlib. Dataset ships
   as generated-at-load or bundled JSON. `geopandas`, `cartopy`, `matplotlib`,
   `contextily` are dev/renderer deps only — NEVER in `[project.dependencies]`.
6. **No Infinrg material.** Nothing from Infinrg biogas site-selection work —
   no code, data, framings, or file fragments — may appear in this repo, ever.
   Wind-field and ops realism draws on public knowledge and Rainmaker
   operational experience (public record), cited as experience, not artifacts.
7. **The autopilot is competent.** The sim's autopilot flies commanded routes
   correctly, holds altitude, executes RTL, enforces its dumb failsafes. The
   model must never need to micro-correct trajectories. If a task is solvable
   by classical control, it belongs inside the sim, not in the action space.
   ("Drones already fly themselves; this is the layer that doesn't.")

## Conventions

- Python ≥3.10, `uv` for everything. Type hints on all public functions.
  `ruff` clean. Tests in `tests/`, run with `uv run pytest -q`.
- Sim core is pure functions + dataclasses where possible. `SimState` is a
  single serializable dataclass; every decision point logs a full snapshot to
  `state["sim_log"]` (verifiers state dict) — the renderer and red-team
  tooling consume ONLY these logs, never live sim objects.
- Every episode artifact (rollout JSON) must be renderable offline:
  `uv run python scripts/render.py <rollout.json> -o out.mp4`.
- Tool errors return structured error messages to the model (never raise).
  Invalid actions cost sim time (realism + anti-spam) but don't crash.
- Coordinates: WGS84 lat/lon, bounding box (37.55, -122.60) to (37.95, -122.15).
  Distances via a haversine helper in `uav_operator.py`. Altitudes in ft MSL
  (aviation convention), battery in Wh, wind in kt/deg-from.
- Commit style: imperative, scoped (`sim:`, `env:`, `reward:`, `render:`,
  `scenario:`, `docs:`). Commit at every green-test checkpoint.

## Working agreements for code sessions

- Read SPEC.md before touching sim/env/reward code; read PLAN.md at session
  start to know today's gate; update HANDOFF.md at session end (what changed,
  what's broken, next action).
- Ugliness discipline: until the Day 4 gate, prefer the ugliest implementation
  that produces real decisions. Polish is scheduled (Days 6–7, 12–13), not
  ambient.
- When adding any reward component, simultaneously add: (a) a unit test, (b) a
  line in SPEC.md §Reward table, (c) an entry in `docs/HACKS.md` describing how
  it could be gamed and why it isn't (or an open TODO if it can be).
- Never delete red-team evidence. Failed rubrics, exploit transcripts, and
  pre-fix rollouts go to `assets/redteam/` — they are writeup material.

## Definition of done (v0.1.0 public)

- [ ] `prime env push` succeeds; env installable and runnable from Hub by a stranger
- [ ] README with: one-paragraph pitch, GIF, quickstart, task/reward tables,
      leaderboard (≥4 frontier models + Qwen3-4B base), prior-art positioning
- [ ] Tier 0–3 scenarios, ≥300 train / ≥60 eval episodes, difficulty curve
      shown (frontier models score meaningfully below ceiling on Tier 3)
- [ ] Red-team round documented in `docs/HACKS.md` with ≥3 closed exploits
- [ ] GRPO LoRA run on Qwen3-4B-Instruct shows a rising eval curve; curve PNG
      + config committed; run executed on Prime Intellect compute
- [ ] ≥1 rendered mp4 of a Tier 2+ episode (before/after training pair preferred)
