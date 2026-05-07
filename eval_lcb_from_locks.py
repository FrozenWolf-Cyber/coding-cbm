"""Orchestrate evaluation of LiveCodeBench `*_eval.lock.json` files.

This script stays lightweight: it discovers lock files, filters debug runs, then
launches `eval_lcb_single_lock.py` once per lock via `os.system`. Each worker is
a separate Python process and writes a `*.finish.json` file before exiting; this
script waits for that finish file before moving to the next lock.
"""

import argparse
import glob
import json
import os
import shlex
import sys
import time
from pathlib import Path


def _is_debug_lock(lock_path: Path) -> bool:
    """Return True if the lock corresponds to a debug training run.

    Detection looks at both the `run_id` field inside the JSON and the
    directory segment of the lock path (which is also `{run_id}` because
    `eval_metrics.run_codecontests_evaluation_for_cbm` writes
    `output/{model_label}-{steer_mode}/{run_id}/...`).
    """
    try:
        with open(lock_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        rid = str(payload.get("run_id", "")).strip()
    except Exception as exc:
        print(f"Could not read {lock_path} for debug check: {exc}")
        rid = ""
    return rid.startswith("debug-") or "/debug-" in str(lock_path)


def _finish_file_for_lock(lock_path: Path) -> Path:
    return lock_path.with_suffix(lock_path.suffix + ".finish.json")


def _build_worker_args(args) -> list[str]:
    """Args forwarded to the separate single-lock worker script."""
    forwarded = [
        "--wandb_project", args.wandb_project,
        "--lcb_num_process_evaluate", str(args.lcb_num_process_evaluate),
        "--lcb_timeout", str(args.lcb_timeout),
        "--lcb_recursion_limit", str(args.lcb_recursion_limit),
    ]
    if args.include_debug:
        forwarded.append("--include_debug")
    return forwarded


def _shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def _wait_for_finish_file_forever(finish_file: Path) -> dict:
    """Wait indefinitely for the worker's finish JSON and return its payload."""
    next_log_at = time.time() + 60
    while True:
        if finish_file.is_file():
            with open(finish_file, "r", encoding="utf-8") as f:
                return json.load(f)
        now = time.time()
        if now >= next_log_at:
            print(f"Still waiting for finish file: {finish_file}", flush=True)
            next_log_at = now + 60
        time.sleep(1)


def _system_exit_code(status: int) -> int:
    if hasattr(os, "waitstatus_to_exitcode"):
        return os.waitstatus_to_exitcode(status)
    if status == 0:
        return 0
    return status >> 8


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--locks_glob",
        type=str,
        default="LiveCodeBench/output/*/*/codegeneration_*_eval.lock.json",
        help="Glob for pending LiveCodeBench eval lock files.",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="coding-qa",
        help="W&B project for resumed logging.",
    )
    parser.add_argument(
        "--max_locks",
        type=int,
        default=0,
        help="Optional max number of lock files to process (0 means all).",
    )
    parser.add_argument(
        "--lcb_num_process_evaluate",
        type=int,
        default=2,
        help="Parallel workers for LCB pass@k grading.",
    )
    parser.add_argument(
        "--lcb_timeout",
        type=int,
        default=6,
        help="Per-test-case timeout (seconds) for LCB grading.",
    )
    parser.add_argument(
        "--lcb_recursion_limit",
        type=int,
        default=12000,
        help="Recursion limit used by LCB sandbox during grading.",
    )
    parser.add_argument(
        "--include_debug",
        action="store_true",
        help=(
            "Also evaluate locks whose run_id starts with 'debug-' (produced by "
            "train_combined_finegrained.py --debug/--debug_0_step). Off by default."
        ),
    )
    args = parser.parse_args()
    os.environ["LCB_RECURSION_LIMIT"] = str(int(args.lcb_recursion_limit))

    # ── Orchestrator mode: discover, filter, then run one worker per lock ─
    lock_files = sorted(glob.glob(args.locks_glob))
    if args.max_locks > 0:
        lock_files = lock_files[: args.max_locks]

    if not lock_files:
        print(f"No lock files matched: {args.locks_glob}")
        return

    print(f"Found {len(lock_files)} lock files.")

    pending: list[Path] = []
    for lock_file in lock_files:
        lock_path = Path(lock_file)
        if not args.include_debug and _is_debug_lock(lock_path):
            print(f"Skipping debug lock: {lock_path}")
            continue
        pending.append(lock_path)

    skipped = len(lock_files) - len(pending)
    print(f"{len(pending)} locks to evaluate ({skipped} debug skipped).")

    forwarded = _build_worker_args(args)
    worker_path = Path(__file__).with_name("eval_lcb_single_lock.py").resolve()
    if not worker_path.is_file():
        raise FileNotFoundError(f"Worker script not found: {worker_path}")

    n_ok = 0
    n_fail = 0
    n_missing_finish = 0
    for idx, lock_path in enumerate(pending, 1):
        print(f"\n[{idx}/{len(pending)}] Evaluating {lock_path} ...", flush=True)
        t0 = time.perf_counter()
        finish_file = _finish_file_for_lock(lock_path)
        if finish_file.exists():
            finish_file.unlink()

        cmd = _shell_join(
            [
                sys.executable,
                str(worker_path),
                "--lock_path", str(lock_path),
                "--finish_file", str(finish_file),
                *forwarded,
            ]
        )
        print(f"Running: {cmd}", flush=True)
        rc = _system_exit_code(os.system(cmd))
        finish_payload = _wait_for_finish_file_forever(finish_file)
        finish_status = str(finish_payload.get("status", "unknown"))

        elapsed = time.perf_counter() - t0
        if rc == 0:
            n_ok += 1
            print(
                f"[{idx}/{len(pending)}] {finish_status} in {elapsed:.1f}s "
                f"(finish={finish_file})",
                flush=True,
            )
        else:
            n_fail += 1
            print(
                f"[{idx}/{len(pending)}] worker exited with code {rc} "
                f"after {elapsed:.1f}s (finish_status={finish_status}, finish={finish_file})",
                flush=True,
            )

    print(
        f"\nOrchestrator summary: ok={n_ok} fail={n_fail} missing_finish={n_missing_finish} "
        f"debug_skipped={skipped} total_locks={len(lock_files)}"
    )


if __name__ == "__main__":
    main()
