#!/usr/bin/env python3
"""
install_collab.py — Colab installer for Depth-Anything-collab-notebook

- Colab-friendly: no venv.
- Ensures ffmpeg exists.
- Installs python deps from requirements*.txt if present, else minimal deps.
- Clones Depth-Anything-V2 into ./Depth-Anything-V2 (expected by test_install.py).
- Downloads official Depth Anything V2 checkpoints into ./checkpoints/:
    - depth_anything_v2_vits.pth
    - depth_anything_v2_vitb.pth
    - depth_anything_v2_vitl.pth
- Prepares /content/jobs and /content/work.

Usage (in Colab):
  !python install_collab.py --with-widgets --smoke-test
  !python install_collab.py --models vitb        # only Base
  !python install_collab.py --models all         # vits+vitb+vitl (default)
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Iterable

# --- Repo paths ---
REPO_ROOT = Path(__file__).resolve().parent

DEPTH_REPO_NAME = "Depth-Anything-V2"
DEPTH_REPO_URL = "https://github.com/DepthAnything/Depth-Anything-V2.git"
DEPTH_REPO_DIR = REPO_ROOT / DEPTH_REPO_NAME  # where test_install.py expects it

# --- Runtime dirs (your pipeline uses these) ---
JOBS_DIR = Path("/content/jobs")
WORK_DIR = Path("/content/work")

# --- Where your converter expects checkpoints ---
REPO_CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"

# Official checkpoints (Small/Base/Large) hosted on Hugging Face
# (keep URLs in code to avoid raw links in chat text)
CKPT_URLS = {
    "vits": "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth?download=true",
    "vitb": "https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth?download=true",
    "vitl": "https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true",
}
CKPT_FILES = {
    "vits": "depth_anything_v2_vits.pth",
    "vitb": "depth_anything_v2_vitb.pth",
    "vitl": "depth_anything_v2_vitl.pth",
}

# Requirements file preference order (in your repo)
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
    return os.path.exists("/content") and (
        "COLAB_GPU" in os.environ or "google.colab" in sys.modules or "COLAB_RELEASE_TAG" in os.environ
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
        return
    print("🟨 ffmpeg not found → installing via apt-get...")
    run(["bash", "-lc", "apt-get update -y"])
    run(["bash", "-lc", "apt-get install -y ffmpeg"])


def pip_install_base(args: argparse.Namespace) -> None:
    # Upgrade pip toolchain only; avoid forcing torch in Colab (it comes preinstalled for GPU runtimes).
    run([sys.executable, "-m", "pip", "install", "-U", "pip", "setuptools", "wheel"])

    req_file = next((p for p in REQUIREMENTS_CANDIDATES if p.exists()), None)
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
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    REPO_CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    print("✅ Prepared dirs:")
    print(" -", JOBS_DIR)
    print(" -", WORK_DIR)
    print(" -", REPO_CHECKPOINTS_DIR)


def ensure_depth_repo(args: argparse.Namespace) -> None:
    if DEPTH_REPO_DIR.exists():
        print(f"✅ {DEPTH_REPO_NAME} already present: {DEPTH_REPO_DIR}")
    else:
        if args.no_git:
            raise SystemExit(f"Depth repo missing at {DEPTH_REPO_DIR} and --no-git was set.")
        print(f"🟨 Cloning {DEPTH_REPO_NAME} into: {DEPTH_REPO_DIR}")
        run(["bash", "-lc", f"git clone --depth 1 {DEPTH_REPO_URL} {DEPTH_REPO_DIR}"])

    # Install Depth-Anything-V2 requirements if present
    req = DEPTH_REPO_DIR / "requirements.txt"
    if req.exists():
        print(f"✅ Installing Depth-Anything-V2 requirements: {req}")
        run([sys.executable, "-m", "pip", "install", "-r", str(req)])
    else:
        print("🟨 Depth-Anything-V2 requirements.txt not found (skipping).")


def download_file(url: str, dst: Path) -> None:
    """
    Robust downloader using curl if available (progress + resume),
    else wget, else python requests (fallback).
    """
    dst.parent.mkdir(parents=True, exist_ok=True)

    # If already there and non-trivial size, skip
    if dst.exists() and dst.stat().st_size > 10_000_000:  # >10MB
        print(f"✅ Already downloaded: {dst.name} ({dst.stat().st_size/1e6:.1f} MB)")
        return

    if which("curl"):
        # -L follow redirects, -C - resume, --fail to error on HTTP errors, --progress-bar for progress
        run(["bash", "-lc", f'curl -L --fail -C - --progress-bar -o "{dst}" "{url}"'])
        return
    if which("wget"):
        run(["bash", "-lc", f'wget -c -O "{dst}" "{url}"'])
        return

    # Fallback: python requests streaming
    import requests
    from tqdm import tqdm

    print(f"🟨 Downloading (python): {url}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", "0") or 0)
        tmp = dst.with_suffix(dst.suffix + ".part")
        with open(tmp, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dst.name) as bar:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                bar.update(len(chunk))
        tmp.replace(dst)


def resolve_models_selection(sel: str) -> list[str]:
    sel = (sel or "").strip().lower()
    if sel in ("all", "common", ""):
        return ["vits", "vitb", "vitl"]
    parts = [p.strip() for p in sel.replace(",", " ").split() if p.strip()]
    ok = []
    for p in parts:
        if p not in CKPT_URLS:
            raise SystemExit(f"Unknown model '{p}'. Use: vits, vitb, vitl, all")
        ok.append(p)
    return ok


def ensure_checkpoints(args: argparse.Namespace) -> None:
    wanted = resolve_models_selection(args.models)
    print(f"✅ Will ensure checkpoints: {wanted} in {REPO_CHECKPOINTS_DIR}")

    for key in wanted:
        url = CKPT_URLS[key]
        fname = CKPT_FILES[key]
        dst = REPO_CHECKPOINTS_DIR / fname
        download_file(url, dst)

    # Quick listing
    print("\n📦 Checkpoints present:")
    for p in sorted(REPO_CHECKPOINTS_DIR.glob("depth_anything_v2_*.pth")):
        print(f" - {p.name} ({p.stat().st_size/1e6:.1f} MB)")


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
    ap.add_argument("--no-git", action="store_true", help="Skip git clone for Depth-Anything-V2.")
    ap.add_argument(
        "--models",
        default="all",
        help="Which checkpoints to download: 'vits', 'vitb', 'vitl', or 'all' (default: all). Example: --models vitb",
    )
    args = ap.parse_args()

    print_env_info()
    apt_install_if_missing(args)
    pip_install_base(args)
    prepare_dirs()

    ensure_depth_repo(args)
    ensure_checkpoints(args)

    smoke_test(args)

    print("\n✅ Colab install done.")
    print("Next steps:")
    print(f" - Depth repo: {DEPTH_REPO_DIR}")
    print(f" - Checkpoints: {REPO_CHECKPOINTS_DIR}  (expected by your converter)")
    print(f" - Jobs: {JOBS_DIR}")
    print(f" - Work: {WORK_DIR}")
    print(" - Run: python test_install.py ... or python run_job.py ...")


if __name__ == "__main__":
    main()

