#!/usr/bin/env python3
"""
install_collab.py — Colab installer for Depth-Anything-collab-notebook

What it does:
- Colab-compatible: no venv, installs into current runtime.
- Ensures ffmpeg is available (apt-get if missing).
- Installs python deps (requirements-colab.txt / requirements.txt if present, else a minimal set).
- Ensures Depth-Anything-V2 repo is cloned into ./Depth-Anything-V2 (expected by test_install.py).
- Installs Depth-Anything-V2 requirements if present.
- Prepares runtime folders: /content/jobs, /content/work, /content/checkpoints
- Copies bundled checkpoints from repo if present (optional; non-destructive).
- Optional smoke test: imports + torch/cuda check.

Usage in Colab:
  !python install_collab.py --with-widgets --smoke-test
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# --- Paths / repo layout ---
REPO_ROOT = Path(__file__).resolve().parent

DEPTH_REPO_NAME = "Depth-Anything-V2"
DEPTH_REPO_URL = "https://github.com/DepthAnything/Depth-Anything-V2.git"
DEPTH_REPO_DIR = REPO_ROOT / DEPTH_REPO_NAME  # where test_install.py expects it

# Runtime dirs used by your pipeline / jobs
JOBS_DIR = Path("/content/jobs")
WORK_DIR = Path("/content/work")
CHECKPOINTS_TARGET = Path("/content/checkpoints")

# Bundled checkpoints folder candidates inside your repo (if you ship a small model)
BUNDLED_CHECKPOINTS_DIR_CANDIDATES = [
    REPO_ROOT / "checkpoints",
    REPO_ROOT / "checkpoint",
    REPO_ROOT / "models",
    REPO_ROOT / "weights",
]

# Requirements files preference order (in your repo)
REQUIREMENTS_CANDIDATES = [
    REPO_ROOT / "requirements-colab.txt",
    REPO_ROOT / "requirements_colab.txt",
    REPO_ROOT / "requirements.txt",
]


def run(cmd: list[str] | str, check: bool = True, env: dict | None = None) -> subprocess.CompletedProcess:
    if isinstance(cmd, str):
        shell = True
        printable = cmd
    else:
        shell = False
        printable = " ".join(cmd)
    print(f"\n🟦 RUN: {printable}")
    return subprocess.run(cmd, shell=shell, check=check, env=env, text=True)


def which(exe: str) -> Optional[str]:
    from shutil import which as _which
    return _which(exe)


def is_colab() -> bool:
    # Common heuristics
    return (
        os.path.exists("/content")
        and ("COLAB_GPU" in os.environ or "google.colab" in sys.modules or "COLAB_RELEASE_TAG" in os.environ)
    )


def print_env_info() -> None:
    print("\n🧾 Environment info")
    print(" - timestamp:", datetime.utcnow().isoformat() + "Z")
    print(" - python:", sys.version.replace("\n", " "))
    print(" - repo:", REPO_ROOT)
    print(" - colab:", is_colab())
    print(" - ffmpeg:", which("ffmpeg") or "missing")
    print(" - nvidia-smi:", which("nvidia-smi") or "missing")
    if which("nvidia-smi"):
        run(["bash", "-lc", "nvidia-smi -L || true"], check=False)


def apt_install_if_missing(args: argparse.Namespace) -> None:
    if args.no_apt:
        print("🟨 --no-apt set: skipping apt installs.")
        return

    if which("ffmpeg"):
        print("✅ ffmpeg found.")
    else:
        print("🟨 ffmpeg not found → installing via apt-get...")
        run(["bash", "-lc", "apt-get update -y"])
        run(["bash", "-lc", "apt-get install -y ffmpeg"])


def pip_install_base(args: argparse.Namespace) -> None:
    # Always use python -m pip to avoid mismatches.
    # We upgrade pip/setuptools/wheel but keep it simple.
    run([sys.executable, "-m", "pip", "install", "-U", "pip", "setuptools", "wheel"])

    req_file = None
    for cand in REQUIREMENTS_CANDIDATES:
        if cand.exists():
            req_file = cand
            break

    if req_file:
        print(f"✅ Using requirements file: {req_file}")
        run([sys.executable, "-m", "pip", "install", "-r", str(req_file)])
    else:
        print("🟨 No requirements*.txt found in repo. Installing minimal dependency set.")
        base = [
            "pyyaml",
            "tqdm",
            "numpy",
            "opencv-python",
            "imageio",
            "imageio-ffmpeg",
            "yt-dlp",
            "requests",
        ]
        run([sys.executable, "-m", "pip", "install", *base])

    if args.with_widgets:
        run([sys.executable, "-m", "pip", "install", "ipywidgets", "ipyfilechooser"])


def prepare_dirs() -> None:
    for d in (JOBS_DIR, WORK_DIR, CHECKPOINTS_TARGET):
        d.mkdir(parents=True, exist_ok=True)
    print(f"✅ Prepared dirs:\n- {JOBS_DIR}\n- {WORK_DIR}\n- {CHECKPOINTS_TARGET}")


def ensure_repo_cloned(name: str, url: str, target_dir: Path, args: argparse.Namespace) -> None:
    if target_dir.exists():
        print(f"✅ {name} already present: {target_dir}")
        return
    if args.no_git:
        print(f"🟨 --no-git set: skipping clone of {name}.")
        return
    print(f"🟨 Cloning {name} into: {target_dir}")
    run(["bash", "-lc", f"git clone --depth 1 {url} {target_dir}"])


def install_requirements_if_present(req_file: Path) -> None:
    if req_file.exists():
        print(f"✅ Installing requirements: {req_file}")
        run([sys.executable, "-m", "pip", "install", "-r", str(req_file)])
    else:
        print(f"🟨 Requirements not found: {req_file} (skipping)")


def ensure_depth_anything_v2(args: argparse.Namespace) -> Path:
    ensure_repo_cloned(DEPTH_REPO_NAME, DEPTH_REPO_URL, DEPTH_REPO_DIR, args)

    # Install Depth-Anything-V2 requirements if they exist
    install_requirements_if_present(DEPTH_REPO_DIR / "requirements.txt")

    return DEPTH_REPO_DIR


def copy_bundled_checkpoints(args: argparse.Namespace) -> None:
    if args.no_copy_checkpoints:
        print("🟨 --no-copy-checkpoints set: skipping checkpoint copy.")
        return

    src_dir = None
    for cand in BUNDLED_CHECKPOINTS_DIR_CANDIDATES:
        if cand.exists() and cand.is_dir() and any(cand.iterdir()):
            src_dir = cand
            break

    if not src_dir:
        print("🟨 No bundled checkpoints directory found in repo (checked: "
              + ", ".join(str(p) for p in BUNDLED_CHECKPOINTS_DIR_CANDIDATES) + ").")
        print("   If you rely on external checkpoints, place them in /content/checkpoints.")
        return

    print(f"✅ Found bundled checkpoints in: {src_dir}")
    copied = 0

    for item in src_dir.rglob("*"):
        if item.is_dir():
            continue
        rel = item.relative_to(src_dir)
        dst = CHECKPOINTS_TARGET / rel
        dst.parent.mkdir(parents=True, exist_ok=True)

        if dst.exists() and not args.overwrite_checkpoints:
            continue

        shutil.copy2(item, dst)
        copied += 1

    print(f"✅ Checkpoints copy complete. Copied/updated: {copied} file(s)")
    print(f"   Target: {CHECKPOINTS_TARGET}")


def smoke_test(args: argparse.Namespace) -> None:
    if not args.smoke_test:
        return

    print("\n🧪 Smoke test: imports + torch/cuda check")
    code = r"""
import sys
print("python:", sys.version)
mods = ["yaml","tqdm","numpy","cv2","imageio","requests"]
for m in mods:
    __import__(m)
print("imports ok:", mods)

import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
"""
    run([sys.executable, "-c", code], check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Install dependencies & prepare workspace for Colab.")
    ap.add_argument("--smoke-test", action="store_true", help="Run import checks and torch/cuda check.")
    ap.add_argument("--no-apt", action="store_true", help="Skip apt-get installs (ffmpeg).")
    ap.add_argument("--with-widgets", action="store_true", help="Install ipywidgets + ipyfilechooser.")
    ap.add_argument("--no-git", action="store_true", help="Skip git cloning external repos (Depth-Anything-V2).")
    ap.add_argument("--no-copy-checkpoints", action="store_true", help="Don't copy bundled checkpoints from repo to /content/checkpoints.")
    ap.add_argument("--overwrite-checkpoints", action="store_true", help="Overwrite existing checkpoint files when copying.")
    args = ap.parse_args()

    print_env_info()

    apt_install_if_missing(args)
    pip_install_base(args)
    prepare_dirs()

    # Ensure Depth-Anything-V2 exists where test_install.py expects it
    depth_dir = ensure_depth_anything_v2(args)
    if depth_dir.exists():
        print(f"✅ Depth repo ready: {depth_dir}")
    else:
        print(f"⚠️ Depth repo missing: {depth_dir} (clone may have been skipped)")

    copy_bundled_checkpoints(args)

    smoke_test(args)

    print("\n✅ Colab install done.")
    print("Next steps:")
    print(f" - Depth repo: {DEPTH_REPO_DIR}")
    print(f" - Put/verify checkpoints in: {CHECKPOINTS_TARGET}")
    print(f" - Create jobs in: {JOBS_DIR}")
    print(" - Run your runner: python run_job.py (as you planned)")


if __name__ == "__main__":
    main()

