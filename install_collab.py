#!/usr/bin/env python3
"""
Colab installer for Depth-Anything-collab-notebook

Goals:
- Colab-compatible: no venv, installs into current runtime.
- Idempotent: safe to re-run.
- Optional apt install for ffmpeg (usually present, but we verify).
- Install python deps from requirements files if present, else fallback minimal deps.
- Prepare folders: jobs/, work/, checkpoints/
- Copy bundled checkpoints from repo if present.
- Optional smoke test (imports + basic sanity).

Usage (in Colab):
  !python install_collab.py --smoke-test
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


# ---- Tunables (adjust if your repo structure differs) ----
REPO_ROOT = Path(__file__).resolve().parent

# If your repo contains model(s) here, they'll be copied to CHECKPOINTS_TARGET
BUNDLED_CHECKPOINTS_DIR_CANDIDATES = [
    REPO_ROOT / "checkpoints",
    REPO_ROOT / "checkpoint",
    REPO_ROOT / "models",
    REPO_ROOT / "weights",
]

# Where your runtime should find checkpoints on Colab
CHECKPOINTS_TARGET = Path("/content/checkpoints")

# Work dirs used by jobs/pipeline
JOBS_DIR = Path("/content/jobs")
WORK_DIR = Path("/content/work")

# Requirements file preference order
REQUIREMENTS_CANDIDATES = [
    REPO_ROOT / "requirements-colab.txt",
    REPO_ROOT / "requirements_colab.txt",
    REPO_ROOT / "requirements.txt",
]


def is_colab() -> bool:
    return os.path.exists("/content") and "COLAB_GPU" in os.environ or "google.colab" in sys.modules


def run(cmd: list[str] | str, check: bool = True, env: dict | None = None) -> subprocess.CompletedProcess:
    if isinstance(cmd, str):
        shell = True
        printable = cmd
    else:
        shell = False
        printable = " ".join(cmd)

    print(f"\n🟦 RUN: {printable}")
    return subprocess.run(cmd, shell=shell, check=check, env=env, text=True)


def which(exe: str) -> str | None:
    from shutil import which as _which
    return _which(exe)


def apt_install_if_missing(args: argparse.Namespace) -> None:
    if args.no_apt:
        print("🟨 --no-apt set: skipping apt installs.")
        return

    # ffmpeg is frequently preinstalled; only install if missing
    if which("ffmpeg"):
        print("✅ ffmpeg found.")
    else:
        print("🟨 ffmpeg not found → installing via apt-get...")
        run(["bash", "-lc", "apt-get update -y"])
        run(["bash", "-lc", "apt-get install -y ffmpeg"])


def pip_install(args: argparse.Namespace) -> None:
    # Always use python -m pip to avoid mismatches
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
        print("🟨 No requirements*.txt found. Installing minimal dependency set.")
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
        # Torch is usually present in Colab; do not force reinstall unless requested.
        if args.force_torch:
            # WARNING: this may change over time; user can choose to force if needed.
            base += ["torch", "torchvision", "torchaudio"]
        run([sys.executable, "-m", "pip", "install", *base])

    # Helpful extras (optional)
    if args.with_widgets:
        run([sys.executable, "-m", "pip", "install", "ipywidgets", "ipyfilechooser"])


def prepare_dirs() -> None:
    for d in [JOBS_DIR, WORK_DIR, CHECKPOINTS_TARGET]:
        d.mkdir(parents=True, exist_ok=True)
    print(f"✅ Prepared dirs:\n- {JOBS_DIR}\n- {WORK_DIR}\n- {CHECKPOINTS_TARGET}")


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
        print("   If you rely on external checkpoints, place them in /content/checkpoints or update this script.")
        return

    print(f"✅ Found bundled checkpoints in: {src_dir}")
    # Copy files (not deleting existing). Idempotent.
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


def print_env_info() -> None:
    print("\n🧾 Environment info")
    print(" - python:", sys.version.replace("\n", " "))
    print(" - repo:", REPO_ROOT)
    print(" - colab:", is_colab())
    print(" - ffmpeg:", which("ffmpeg") or "missing")
    print(" - nvidia-smi:", "found" if which("nvidia-smi") else "missing")
    if which("nvidia-smi"):
        try:
            run(["bash", "-lc", "nvidia-smi -L || true"], check=False)
        except Exception:
            pass


def smoke_test(args: argparse.Namespace) -> None:
    if not args.smoke_test:
        return

    print("\n🧪 Smoke test: imports + torch cuda check")
    code = r"""
import sys
print("python:", sys.version)
mods = ["yaml","tqdm","numpy","cv2","imageio","requests"]
for m in mods:
    __import__(m)
print("imports ok:", mods)

try:
    import torch
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))
except Exception as e:
    print("torch check skipped/failed:", e)
"""
    run([sys.executable, "-c", code])

    # If you have test scripts, we can optionally run them.
    if args.run_tests:
        # try the common ones; ignore if missing
        candidates = ["test_install.py", "test_all.py"]
        for t in candidates:
            p = REPO_ROOT / t
            if p.exists():
                print(f"\n🧪 Running {t} ...")
                run([sys.executable, str(p)], check=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Install dependencies & prepare workspace for Colab.")
    ap.add_argument("--smoke-test", action="store_true", help="Run import checks and basic torch/cuda check.")
    ap.add_argument("--run-tests", action="store_true", help="Run repo tests if present (best-effort).")
    ap.add_argument("--no-apt", action="store_true", help="Skip apt-get installs (ffmpeg).")
    ap.add_argument("--force-torch", action="store_true", help="Force install torch/vision/audio via pip (usually not needed on Colab).")
    ap.add_argument("--with-widgets", action="store_true", help="Install ipywidgets + ipyfilechooser.")
    ap.add_argument("--no-copy-checkpoints", action="store_true", help="Don't copy bundled checkpoints from repo to /content/checkpoints.")
    ap.add_argument("--overwrite-checkpoints", action="store_true", help="Overwrite existing checkpoint files when copying.")
    args = ap.parse_args()

    print_env_info()
    apt_install_if_missing(args)
    pip_install(args)
    prepare_dirs()
    copy_bundled_checkpoints(args)
    smoke_test(args)

    print("\n✅ Colab install done.")
    print("Next steps:")
    print(f" - Put/verify checkpoints in: {CHECKPOINTS_TARGET}")
    print(f" - Create jobs in: {JOBS_DIR}")
    print(" - Run your runner: python run_job.py (as you planned)")


if __name__ == "__main__":
    main()

