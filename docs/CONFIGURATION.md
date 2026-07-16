# Configuration

`uav_operator.load_environment(**kwargs)` is the only public environment entry
point. It returns a Verifiers `MultiTurnEnv` and accepts the implemented
settings below.

## Environment settings

| Setting | Type | Default | Behavior |
| --- | --- | --- | --- |
| `seed` | integer | `0` | Base seed used to derive deterministic per-example seeds |
| `max_examples` | integer | `-1` | Number of generated rows; `-1` selects the split default |
| `max_turns` | integer | `40` | Maximum model turns before the harness terminates the episode |
| `sim_time_cap_min` | integer | `90` | Maximum simulated mission duration in minutes |
| `tier` | string | `mixed_day5` | `T0`, `T1`, `T2`, `T3`, `mixed_day5`/`mixed`, or legacy `mixed_day4`/`day4` |
| `dataset_split` | string | `default` | Dataset ownership mode: `default`, `train`, `dev`, or `eval` |
| `system_prompt` | string | built-in operator manual | Replaces the environment system prompt |
| `wind_enabled` | boolean | `true` | Enables the seeded wind field and altitude shear |
| `gust_front_probability` | float | `0.5` | Episode probability of a seeded gust front |

Additional keyword arguments are forwarded to `vf.MultiTurnEnv`. Invalid
values rejected by Verifiers remain harness errors rather than environment
configuration fields.

## Dataset splits

The generator has fixed default sizes:

| Split | Rows | Seed offset from `seed` | Intended use |
| --- | ---: | ---: | --- |
| `train` | 300 | 0 | Training and development of policies |
| `dev` | 60 | 10,000 | Calibration and model selection |
| `eval` | 60 | 20,000 | Held-out final evaluation |

Each mixed split cycles evenly through T0, T1, T2, and T3. A fixed tier
override generates every selected row at that tier.

`max_examples` replaces the split's default row count; it does not sample from
or truncate a prebuilt file. Rows are generated deterministically in index
order.

`dataset_split = "default"` gives the environment a 300-row train dataset and a
separate 60-row eval dataset. Selecting `train`, `dev`, or `eval` explicitly
uses that same generated dataset for both `dataset` and `eval_dataset`. This is
useful when an evaluation or training harness needs exact split ownership.

The split seed ranges are disjoint for a common base seed. Changing the base
seed shifts all three ranges while preserving the offsets.

## Taskset and harness ownership

Environment/taskset settings are values passed to `load_environment`: scenario
tier, split, seed, generated row count, simulator limits, wind behavior, and
the system prompt.

Evaluation-harness settings control model execution around the environment:
model and provider selection, number of examples and rollouts, sampling tokens
and temperature, concurrency, retries, result saving, and requested state
columns.

In the current Prime eval TOML format:

- `env_args` contains environment/taskset settings.
- Top-level fields such as `model`, `num_examples`, `rollouts_per_example`,
  `max_tokens`, and `temperature` belong to the evaluation harness.
- `state_columns = ["sim_state", "sim_log"]` requests saved simulator evidence.
- `[[eval]] id = "uav-operator"` selects the local environment entry point.

Do not put model sampling settings in `env_args`; they are not consumed by
`load_environment`.

## Example evaluation TOML

```toml
model = "poolside/laguna-m.1"
provider = "prime"
api_client_type = "openai_chat_completions"

env_args = { tier = "T2", dataset_split = "dev", max_examples = 15, max_turns = 40, seed = 0 }

num_examples = 15
rollouts_per_example = 2
max_concurrent = 8
max_retries = 0
max_tokens = 512
temperature = 0.2

save_results = true
state_columns = ["sim_state", "sim_log"]
disable_tui = true

[[eval]]
id = "uav-operator"
```

Run it with:

```bash
prime --plain eval run path/to/config.toml
```

The committed
[`configs/eval/day8_laguna_m1_t1_functional.toml`](../configs/eval/day8_laguna_m1_t1_functional.toml)
is a smaller working example using T1 development rows.

## Direct Python loading

For tests and repository scripts:

```python
import uav_operator

env = uav_operator.load_environment(
    tier="T3",
    dataset_split="dev",
    max_examples=8,
    max_turns=40,
    seed=0,
)
```

The public Hub package includes the same loader and generated datasets; no
external dataset download is required.
