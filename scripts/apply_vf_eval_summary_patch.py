"""Apply the local vf-eval summary patch to the active environment."""

from __future__ import annotations

import sys
import sysconfig
from pathlib import Path


OLD_BLOCK = """        )
    finally:
        if not config.disable_env_server:
            await vf_env.stop_server()

    metadata_changed = _attach_metadata_name(outputs["metadata"], config.name)
    if _attach_metadata_cost(outputs["metadata"], model_pricing, outputs["outputs"]):
        metadata_changed = True
    if metadata_changed and config.save_results:
        await asyncio.to_thread(save_metadata, outputs["metadata"], results_path)

    return outputs


async def run_evaluations(config: EvalRunConfig) -> None:
    # load event loop lag monitor
    event_loop_lag_monitor = EventLoopLagMonitor(max_measurements=int(1e5))
"""

NEW_BLOCK = """        )
        outputs["metadata"]["results_path"] = str(results_path)
        outputs["metadata"]["run_id"] = results_path.name
    finally:
        if not config.disable_env_server:
            await vf_env.stop_server()

    metadata_changed = True
    metadata_changed = _attach_metadata_name(outputs["metadata"], config.name) or metadata_changed
    if _attach_metadata_cost(outputs["metadata"], model_pricing, outputs["outputs"]):
        metadata_changed = True
    if metadata_changed and config.save_results:
        await asyncio.to_thread(save_metadata, outputs["metadata"], results_path)

    return outputs


async def run_evaluations(config: EvalRunConfig) -> None:
    # load event loop lag monitor
    event_loop_lag_monitor = EventLoopLagMonitor(max_measurements=int(1e5))
"""

OLD_RUN_RESULTS = """    if n > 0:
        lags_arr = np.array(lags)
        mean_lag = float(lags_arr.mean())
        p99_lag = float(np.percentile(lags_arr, 99))
        max_lag = float(lags_arr.max())
        print(
            f"\\nPerformance:\\nevent_loop_lag: mean={print_time(mean_lag)}, p99={print_time(p99_lag)}, max={print_time(max_lag)} (n={n})"
        )


async def run_evaluations_tui(
"""

NEW_RUN_RESULTS = """    if n > 0:
        lags_arr = np.array(lags)
        mean_lag = float(lags_arr.mean())
        p99_lag = float(np.percentile(lags_arr, 99))
        max_lag = float(lags_arr.max())
        print(
            f"\\nPerformance:\\nevent_loop_lag: mean={print_time(mean_lag)}, p99={print_time(p99_lag)}, max={print_time(max_lag)} (n={n})"
        )
    print("\\nRun Results:")
    for results in all_results:
        metadata = results.get("metadata") or {}
        run_id = metadata.get("run_id")
        results_path = metadata.get("results_path")
        if run_id and results_path:
            print(f"run_id: {run_id}")
            print(f"results_path: {results_path}")


async def run_evaluations_tui(
"""


def main() -> int:
    purelib = Path(sysconfig.get_paths()["purelib"])
    target = purelib / "verifiers" / "utils" / "eval_utils.py"
    if not target.exists():
        print(f"missing installed file: {target}", file=sys.stderr)
        return 1

    text = target.read_text()
    if 'outputs["metadata"]["results_path"] = str(results_path)' in text and 'print("\\nRun Results:")' in text:
        print("vf-eval summary patch already present")
        return 0

    if OLD_BLOCK not in text or OLD_RUN_RESULTS not in text:
        print(
            "vf-eval summary patch could not be applied cleanly; "
            "the installed verifiers version may have changed.",
            file=sys.stderr,
        )
        return 1

    text = text.replace(OLD_BLOCK, NEW_BLOCK, 1)
    text = text.replace(OLD_RUN_RESULTS, NEW_RUN_RESULTS, 1)
    target.write_text(text)
    print(f"patched {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
