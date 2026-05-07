"""Worker for evaluating exactly one LiveCodeBench eval lock.

This file is intentionally separate from eval_lcb_from_locks.py so the
orchestrator can invoke it as a fresh Python process for each lock. The worker
writes a finish-status JSON file before exiting; the orchestrator waits for and
reads that file before launching the next worker.
"""

import argparse
import json
import os
import time
from pathlib import Path


def _claim_lock(lock_path: Path) -> Path | None:
    running_path = lock_path.with_suffix(lock_path.suffix + ".running")
    try:
        fd = os.open(str(running_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(f"claimed_at={time.time()}\npid={os.getpid()}\n")
        return running_path
    except FileExistsError:
        return None


def _is_debug_lock(lock_path: Path) -> bool:
    try:
        with open(lock_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        rid = str(payload.get("run_id", "")).strip()
    except Exception as exc:
        print(f"Could not read {lock_path} for debug check: {exc}")
        rid = ""
    return rid.startswith("debug-") or "/debug-" in str(lock_path)


def _write_finish(finish_file: Path, payload: dict) -> None:
    finish_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **payload,
        "finished_at_unix": time.time(),
        "pid": os.getpid(),
    }
    tmp_path = finish_file.with_suffix(finish_file.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    tmp_path.replace(finish_file)


def _process_single_lock(lock_path: Path, args) -> int:
    """Claim, evaluate, finalise one lock file. Returns process exit code."""
    # Heavy imports stay in the worker so the orchestrator process stays small.
    import wandb
    from eval_metrics import evaluate_saved_livecodebench_generation, safe_wandb_log

    running_marker = _claim_lock(lock_path)
    if running_marker is None:
        print(f"Skipping {lock_path} (already claimed).")
        _write_finish(
            Path(args.finish_file),
            {
                "status": "skipped_already_claimed",
                "return_code": 0,
                "lock_path": str(lock_path),
            },
        )
        return 0

    run = None
    try:
        with open(lock_path, "r", encoding="utf-8") as f:
            lock_payload = json.load(f)

        run_id = str(lock_payload.get("run_id", "")).strip()
        steer_mode = str(lock_payload.get("steer_mode", "none"))
        if run_id:
            run = wandb.init(
                project=args.wandb_project,
                id=run_id,
                resume="allow",
            )
            print(f"Resumed W&B run: {run_id}")
        else:
            print(f"No run_id in {lock_path}; evaluating without W&B resume.")

        result = evaluate_saved_livecodebench_generation(
            lock_payload["lcb_output_path"],
            livecodebench_release=lock_payload["livecodebench_release"],
            lcb_num_process_evaluate=int(args.lcb_num_process_evaluate),
            lcb_timeout=int(args.lcb_timeout),
            lcb_eval_path=lock_payload.get("lcb_eval_path"),
            lcb_eval_all_path=lock_payload.get("lcb_eval_all_path"),
        )
        print(
            f"[{steer_mode}] pass@1={result['pass@1']:.4f}, pass@5={result['pass@5']:.4f} "
            f"| eval={result['lcb_eval_all_path']}"
        )

        safe_wandb_log(
            {
                f"lcb/{steer_mode}/pass@1": result["pass@1"],
                f"lcb/{steer_mode}/pass@5": result["pass@5"],
                f"lcb/{steer_mode}/output_path": lock_payload["lcb_output_path"],
                f"lcb/{steer_mode}/eval_all_path": result["lcb_eval_all_path"],
                f"lcb/{steer_mode}/evaluated_from_lock": 1,
            }
        )

        done_payload = {
            **lock_payload,
            "status": "completed",
            "completed_at_unix": time.time(),
            "pass@1": result["pass@1"],
            "pass@5": result["pass@5"],
            "lcb_eval_path": result["lcb_eval_path"],
            "lcb_eval_all_path": result["lcb_eval_all_path"],
        }
        with open(lock_path, "w", encoding="utf-8") as f:
            json.dump(done_payload, f, indent=2)

        done_path = lock_path.with_suffix(".done.json")
        lock_path.rename(done_path)
        print(f"Completed {done_path}")
        _write_finish(
            Path(args.finish_file),
            {
                "status": "completed",
                "return_code": 0,
                "lock_path": str(lock_path),
                "done_path": str(done_path),
                "pass@1": result["pass@1"],
                "pass@5": result["pass@5"],
            },
        )
        return 0
    except Exception as exc:
        print(f"Failed lock {lock_path}: {exc}")
        _write_finish(
            Path(args.finish_file),
            {
                "status": "failed",
                "return_code": 1,
                "lock_path": str(lock_path),
                "error": str(exc),
            },
        )
        return 1
    finally:
        if run is not None:
            run.finish()
        if running_marker.exists():
            running_marker.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock_path", required=True, help="One *_eval.lock.json file to evaluate.")
    parser.add_argument("--finish_file", required=True, help="Status JSON written before worker exit.")
    parser.add_argument("--wandb_project", default="coding-qa", help="W&B project for resumed logging.")
    parser.add_argument("--lcb_num_process_evaluate", type=int, default=2)
    parser.add_argument("--lcb_timeout", type=int, default=6)
    parser.add_argument("--lcb_recursion_limit", type=int, default=12000)
    parser.add_argument(
        "--include_debug",
        action="store_true",
        help="Allow evaluating debug-* locks. Off by default.",
    )
    args = parser.parse_args()

    os.environ["LCB_RECURSION_LIMIT"] = str(int(args.lcb_recursion_limit))
    lock_path = Path(args.lock_path)
    finish_file = Path(args.finish_file)

    if not lock_path.is_file():
        print(f"Lock not found: {lock_path}")
        _write_finish(
            finish_file,
            {"status": "missing_lock", "return_code": 2, "lock_path": str(lock_path)},
        )
        raise SystemExit(2)

    if not args.include_debug and _is_debug_lock(lock_path):
        print(f"Skipping debug lock: {lock_path}")
        _write_finish(
            finish_file,
            {"status": "skipped_debug", "return_code": 0, "lock_path": str(lock_path)},
        )
        raise SystemExit(0)

    raise SystemExit(_process_single_lock(lock_path, args))


if __name__ == "__main__":
    main()
