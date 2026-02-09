"""
Merge a VERL (FSDP) checkpoint and upload it to Hugging Face Hub.

Usage:
  python upload_model.py --local_dir /path/to/checkpoint --repo_name model-name
  python upload_model.py --local_dir /path/to/checkpoint --repo_name model-name --train_type rl
  python upload_model.py --local_dir /path/to/checkpoint --repo_name model-name --org my-org
  python upload_model.py --local_dir /path/to/checkpoint --repo_name model-name --private
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, create_repo


def _clear_dir_contents(dir_path: Path) -> None:
    """Delete all contents of dir_path (but keep dir_path itself)."""
    if not dir_path.exists():
        return
    for child in dir_path.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()


def merge_fsdp_checkpoint(
    local_dir: str,
    target_dir: Path,
    train_type: str | None = None,
    *,
    dry_run: bool = False,
) -> Path:
    """
    Merge a VERL FSDP checkpoint into a target directory.
    """
    target_dir = Path(target_dir).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    merge_source = local_dir
    if train_type == "rl":
        merge_source = os.path.join(local_dir, "actor")

    cmd = [
        sys.executable,
        "-m",
        "verl.model_merger",
        "merge",
        "--backend",
        "fsdp",
        "--local_dir",
        merge_source,
        "--target_dir",
        str(target_dir),
    ]

    print(f"[merge] source dir : {merge_source}")
    print(f"[merge] target dir : {target_dir}")
    print(f"[merge] running    : {' '.join(cmd)}")

    if dry_run:
        print("[merge] dry-run: not executing.")
        return target_dir

    subprocess.run(cmd, check=True)
    print(f"[merge] done. merged model written to: {target_dir}")
    return target_dir


def upload_to_hf(
    model_dir: Path,
    repo_id: str,
    *,
    private: bool = False,
    commit_message: str | None = None,
    dry_run: bool = False,
) -> str:
    """
    Upload a merged model directory to Hugging Face Hub.

    Returns:
        The URL of the uploaded model.
    """
    api = HfApi()

    print(f"[upload] repo_id    : {repo_id}")
    print(f"[upload] model_dir  : {model_dir}")
    print(f"[upload] private    : {private}")

    if dry_run:
        print("[upload] dry-run: not uploading.")
        return f"https://huggingface.co/{repo_id}"

    # Create repo if it doesn't exist
    try:
        create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
        print(f"[upload] repo created/verified: {repo_id}")
    except Exception as e:
        print(f"[upload] warning: could not create repo: {e}")

    # Upload the folder
    if commit_message is None:
        commit_message = "Upload merged VERL checkpoint"

    api.upload_folder(
        folder_path=str(model_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_message,
    )

    url = f"https://huggingface.co/{repo_id}"
    print(f"[upload] done. model uploaded to: {url}")
    return url


def merge_and_upload(
    local_dir: str,
    repo_name: str,
    org: str = "hf-org",
    train_type: str | None = None,
    target_root: str | None = None,
    *,
    private: bool = False,
    commit_message: str | None = None,
    keep_merged: bool = False,
    dry_run: bool = False,
) -> tuple[Path, str]:
    """
    Merge FSDP checkpoint and upload to HF Hub.

    Args:
        local_dir: Path to the VERL checkpoint directory.
        repo_name: Name of the HF repo (without org prefix).
        org: HF organization name (default: hf-org).
        train_type: Type of training checkpoint ('rl', 'sft', 'pretrain', or None).
        target_root: Root directory for merged checkpoints. If None, uses a temp directory.
        private: Whether to make the HF repo private.
        commit_message: Custom commit message for the upload.
        keep_merged: If True, keep the merged checkpoint directory after upload.
        dry_run: If True, print commands but don't execute.

    Returns:
        Tuple of (merged_dir, hf_url).
    """
    repo_id = f"{org}/{repo_name}"

    # Determine target directory for merging
    if target_root is None:
        # Use local scratch if available
        id_file = [f for f in os.listdir("/lfs") if f.startswith("ampere")]
        if id_file:
            ampere_id = id_file[0].replace("ampere", "")
            target_root = f"/lfs/ampere{ampere_id}/0/{os.environ.get('USER', 'merged')}"
        else:
            target_root = tempfile.mkdtemp(prefix="verl_merge_")

    target_dir = Path(target_root) / "merged_checkpoints" / repo_name

    # Clear existing merged content
    if target_dir.exists():
        print(f"[merge_and_upload] clearing existing dir: {target_dir}")
        if not dry_run:
            _clear_dir_contents(target_dir)

    # Step 1: Merge the checkpoint
    print("=" * 60)
    print("Step 1: Merging FSDP checkpoint")
    print("=" * 60)
    merged_dir = merge_fsdp_checkpoint(
        local_dir=local_dir,
        target_dir=target_dir,
        train_type=train_type,
        dry_run=dry_run,
    )

    # Step 2: Upload to HF
    print()
    print("=" * 60)
    print("Step 2: Uploading to Hugging Face Hub")
    print("=" * 60)
    hf_url = upload_to_hf(
        model_dir=merged_dir,
        repo_id=repo_id,
        private=private,
        commit_message=commit_message,
        dry_run=dry_run,
    )

    # Cleanup if requested
    if not keep_merged and not dry_run:
        print(f"[cleanup] removing merged dir: {merged_dir}")
        shutil.rmtree(merged_dir, ignore_errors=True)

    print()
    print("=" * 60)
    print(f"Done! Model available at: {hf_url}")
    print("=" * 60)

    return merged_dir, hf_url


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge a VERL FSDP checkpoint and upload to Hugging Face Hub."
    )
    parser.add_argument(
        "--local_dir",
        required=True,
        help="Checkpoint dir (for RL, will use <local_dir>/actor).",
    )
    parser.add_argument(
        "--repo_name",
        required=True,
        help="Name of the HF repo (without org prefix).",
    )
    parser.add_argument(
        "--org",
        default="hf-org",
        help="HF organization name (default: hf-org).",
    )
    parser.add_argument(
        "--train_type",
        default="rl",
        choices=["rl", "sft", "pretrain"],
        help="Type of checkpoint. If 'rl', merges from <local_dir>/actor. Default: rl.",
    )
    parser.add_argument(
        "--target_root",
        default=None,
        help="Root directory for merged checkpoints. If not specified, uses local scratch.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the HF repo private.",
    )
    parser.add_argument(
        "--commit_message",
        default=None,
        help="Custom commit message for the upload.",
    )
    parser.add_argument(
        "--keep_merged",
        action="store_true",
        help="Keep the merged checkpoint directory after upload.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands/paths but do not execute.",
    )

    args = parser.parse_args()

    merge_and_upload(
        local_dir=args.local_dir,
        repo_name=args.repo_name,
        org=args.org,
        train_type=args.train_type,
        target_root=args.target_root,
        private=args.private,
        commit_message=args.commit_message,
        keep_merged=args.keep_merged,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
