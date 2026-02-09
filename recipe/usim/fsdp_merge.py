"""
Merge a VERL (FSDP) checkpoint into a directory under a provided target root

Output layout:
  <target_root>/merged_checkpoints/<label>/<label>_<run_hash>/

NOTES:
  - Deletes existing contents under <target_root>/merged_checkpoints/<label>/ by default
  - If --train_type rl, uses <local_dir>/actor as the merge source
  - Runs:
      python -m verl.model_merger merge --backend fsdp --local_dir ... --target_dir ...

  ./merge_model.py --local_dir /path/to/ckpt --label reddit_sft --target_root /tmp
  ./merge_model.py --local_dir /path/to/ckpt --label reddit_rl --train_type rl --target_root /models
  ./merge_model.py --local_dir /path/to/ckpt --label run42 --target_root /models --clean
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _clear_dir_contents(dir_path: Path) -> None:
    """Delete all contents of dir_path (but keep dir_path itself)."""
    if not dir_path.exists():
        return
    for child in dir_path.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()


def merge_model_to_target_root(
    local_dir: str,
    label: str,
    target_root: Path,
    train_type: str | None = None,
    *,
    clean: bool = True,
    dry_run: bool = False,
) -> Path:
    """
    Merge a VERL FSDP checkpoint into a given target root.
    """
    assert label.strip() != "", "label must be non-empty string"
    target_root = Path(target_root).expanduser().resolve()
    
    # import hashlib
    # run_id_source = f"{time.time_ns()}"
    # run_hash = hashlib.sha1(run_id_source.encode("utf-8")).hexdigest()[:8]

    if clean:
        _clear_dir_contents(target_root / "merged_checkpoints")

    target_dir = target_root / "merged_checkpoints" / label
    target_dir.mkdir(parents=True, exist_ok=True)

    if train_type == "rl":
        local_dir = os.path.join(local_dir, "actor")
    
    cmd = [
        sys.executable,
        "-m",
        "verl.model_merger",
        "merge",
        "--backend",
        "fsdp",
        "--local_dir",
        local_dir,
        "--target_dir",
        str(target_dir),
    ]

    print(f"[merge_model] target_root : {target_root}")
    print(f"[merge_model] target_dir  : {target_dir}")
    print(f"[merge_model] running     : {' '.join(cmd)}")

    if dry_run:
        print("[merge_model] dry-run: not executing.")
        return target_dir

    subprocess.run(cmd, check=True)
    print(f"[merge_model] done. merged model written to: {target_dir}")
    return target_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge a VERL FSDP checkpoint into a given target root.")
    parser.add_argument("--local_dir", required=True, help="Checkpoint dir (for RL, will use <local_dir>/actor).")
    parser.add_argument("--label", required=True, help="Label for output folder naming.")
    parser.add_argument("--target_root", required=True, help="Root directory to write merged_checkpoints under.")
    parser.add_argument(
        "--train_type",
        default=None,
        choices=[None, "rl", "sft", "pretrain"],
        help="If 'rl', merges from <local_dir>/actor. Other values treated like None.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Do not delete existing contents under merged_checkpoints/<label>/ before merging.",
    )
    parser.add_argument("--dry_run", action="store_true", help="Print command/paths but do not execute.")

    args = parser.parse_args()

    if args.label.strip() == "":
        # take the last two levels
        args.label = args.local_dir.strip("/").split("/")[-2:]
        args.label = "/".join(args.label)

    out_dir = merge_model_to_target_root(
        local_dir=args.local_dir,
        label=args.label,
        target_root=Path(args.target_root),
        train_type=args.train_type,
        clean=args.clean,
        dry_run=args.dry_run,
    )
                        
    print(str(out_dir))


if __name__ == "__main__":
    main()
