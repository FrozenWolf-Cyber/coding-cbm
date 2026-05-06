import argparse
import glob
import json
import os
import time
from pathlib import Path

import wandb

from eval_metrics import evaluate_saved_livecodebench_generation, safe_wandb_log


def _claim_lock(lock_path: Path) -> Path | None:
    running_path = lock_path.with_suffix(lock_path.suffix + ".running")
    try:
        fd = os.open(str(running_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(f"claimed_at={time.time()}\n")
        return running_path
    except FileExistsError:
        return None


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
    args = parser.parse_args()
    os.environ["LCB_RECURSION_LIMIT"] = str(int(args.lcb_recursion_limit))

    lock_files = sorted(glob.glob(args.locks_glob))
    if args.max_locks > 0:
        lock_files = lock_files[: args.max_locks]

    if not lock_files:
        print(f"No lock files matched: {args.locks_glob}")
        return

    print(f"Found {len(lock_files)} lock files.")

    for lock_file in lock_files:
        lock_path = Path(lock_file)
        running_marker = _claim_lock(lock_path)
        if running_marker is None:
            print(f"Skipping {lock_path} (already claimed).")
            continue

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
        except Exception as exc:
            print(f"Failed lock {lock_path}: {exc}")
        finally:
            if run is not None:
                run.finish()
            if running_marker.exists():
                running_marker.unlink()


if __name__ == "__main__":
    main()
