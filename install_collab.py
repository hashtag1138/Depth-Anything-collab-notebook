#!/usr/bin/env python3
"""
install_collab.py — minimal Colab installer (inference-only)

What it does (minimal):
- Print env info (GPU, ffmpeg).
- Ensure ffmpeg exists.
- Optionally install ipywidgets + ipyfilechooser (for notebook UI).
- Clone Depth-Anything-V2 into ./Depth-Anything-V2 (expected by test_install.py).
- Download Depth-Anything-V2 checkpoints into ./checkpoints/ (expected by your converter):
    - depth_anything_v2_vits.pth
    - depth_anything_v2_vitb.pth
    - depth_anything_v2_vitl.pth
- Prepare /content/jobs and /content/work
- Optional smoke test: torch cuda check + imports.

It intentionally DOES NOT install Depth-Anything-V2/requirements.txt (Gradio pins, numpy/websockets downgrades).
It also avoids upgrading setuptools/wheel to reduce environment churn on Colab.

Usage:
  !python install_collab.py --with-widgets --smoke-test
  !python install_collab.py --models vitb
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# --- Repo paths ---
REPO_ROOT = Path(__file__).resolve().parent

DEPTH_REPO_NAME = "Depth-Anything-V2"
DEPTH_REPO_URL = "https://github.com/DepthAnything/Depth-Anything-V2.git"
DEPTH_REPO_DIR = REPO_ROOT / DEPTH_REPO_NAME

# --- Runtime dirs ---
JOBS_DIR = Path("/content/jobs")
WORK_DIR = Path("/content/work")

# --- Where your converter expects checkpoints ---
REPO_CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"

# Official checkpoints on Hugging Face (Small/Base/Large)
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


def run(cmd: list[str] | str, check: bool = True) -> subprocess.CompletedProcess:
    if isinstance(cmd, str):
        shell = True
        printable = cmd
    else:
        shell = False
        printable = " ".join(cmd)
    print(f"\n🟦 RUN: {printable}")
    return subprocess.run(cmd, shell=shell, check=check, text=True)


def which(exe: str) -> Optional[str]:
    from shutil import which as _which
    return _which(exe)


def print_env_info() -> None:
    print("\n🧾 Environment info")
    print(" - timestamp:", datetime.now(timezone.utc).isoformat())
    print(" - python:", sys.version.replace("\n", " "))
    print(" - repo:", REPO_ROOT)
    print(" - ffmpeg:", which("ffmpeg") or "missing")
    print(" - nvidia-smi:", which("nvidia-smi") or "missing")
    if which("nvidia-smi"):
        run(["bash", "-lc", "nvidia-smi -L || true"], check=False)


def ensure_ffmpeg(args: argparse.Namespace) -> None:
    if which("ffmpeg"):
        print("✅ ffmpeg found.")
        return
    if args.no_apt:
        raise SystemExit("ffmpeg not found and --no-apt was set.")
    print("🟨 ffmpeg not found → installing via apt-get...")
    run(["bash", "-lc", "apt-get update -y"])
    run(["bash", "-lc", "apt-get install -y ffmpeg"])


def ensure_widgets(args: argparse.Namespace) -> None:
    if not args.with_widgets:
        return
    # Only install if missing (light touch)
    try:
        import ipywidgets  # noqa
        import ipyfilechooser  # noqa
        print("✅ widgets already installed.")
        return
    except Exception:
        pass
    run([sys.executable, "-m", "pip", "install", "-q", "ipywidgets", "ipyfilechooser"])


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
        return
    if args.no_git:
        raise SystemExit(f"{DEPTH_REPO_NAME} missing and --no-git was set.")
    print(f"🟨 Cloning {DEPTH_REPO_NAME} into: {DEPTH_REPO_DIR}")
    run(["bash", "-lc", f"git clone --depth 1 {DEPTH_REPO_URL} {DEPTH_REPO_DIR}"])


def download_file(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)

    # Skip if already downloaded and looks real (>10MB)
    if dst.exists() and dst.stat().st_size > 10_000_000:
        print(f"✅ Already downloaded: {dst.name} ({dst.stat().st_size/1e6:.1f} MB)")
        return

    if which("curl"):
        run(["bash", "-lc", f'curl -L --fail -C - --progress-bar -o "{dst}" "{url}"'])
        return
    if which("wget"):
        run(["bash", "-lc", f'wget -c -O "{dst}" "{url}"'])
        return

    # Very rare fallback
    run([sys.executable, "-m", "pip", "install", "-q", "requests"])
    import requests
    from tqdm import tqdm

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
    if sel in ("all", ""):
        return ["vits", "vitb", "vitl"]
    parts = [p.strip() for p in sel.replace(",", " ").split() if p.strip()]
    for p in parts:
        if p not in CKPT_URLS:
            raise SystemExit(f"Unknown model '{p}'. Use: vits, vitb, vitl, all")
    return parts


def ensure_checkpoints(args: argparse.Namespace) -> None:
    wanted = resolve_models_selection(args.models)
    print(f"✅ Will ensure checkpoints: {wanted} in {REPO_CHECKPOINTS_DIR}")

    for key in wanted:
        url = CKPT_URLS[key]
        fname = CKPT_FILES[key]
        dst = REPO_CHECKPOINTS_DIR / fname
        download_file(url, dst)

    print("\n📦 Checkpoints present:")
    for p in sorted(REPO_CHECKPOINTS_DIR.glob("depth_anything_v2_*.pth")):
        print(f" - {p.name} ({p.stat().st_size/1e6:.1f} MB)")


def smoke_test(args: argparse.Namespace) -> None:
    if not args.smoke_test:
        return
    print("\n🧪 Smoke test: torch + cuda + minimal imports")
    code = r"""
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))

import yaml, cv2, numpy, imageio, requests
print("imports ok")
"""
    run([sys.executable, "-c", code], check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Minimal Colab install (inference-only).")
    ap.add_argument("--smoke-test", action="store_true", help="Run torch/cuda + imports check.")
    ap.add_argument("--with-widgets", action="store_true", help="Install ipywidgets + ipyfilechooser if missing.")
    ap.add_argument("--no-apt", action="store_true", help="Skip apt-get installs (ffmpeg).")
    ap.add_argument("--no-git", action="store_true", help="Skip git clone Depth-Anything-V2.")
    ap.add_argument("--models", default="vitb", help="Which checkpoints to download: vits, vitb, vitl, all. Default: vitb.")
    args = ap.parse_args()

    print_env_info()
    ensure_ffmpeg(args)
    ensure_widgets(args)
    prepare_dirs()
    ensure_depth_repo(args)
    ensure_checkpoints(args)
    smoke_test(args)

    print("\n✅ Colab install done.")
    print("Next steps:")
    print(f" - Depth repo: {DEPTH_REPO_DIR}")
    print(f" - Checkpoints: {REPO_CHECKPOINTS_DIR} (expected by your converter)")
    print(f" - Jobs: {JOBS_DIR}")
    print(f" - Work: {WORK_DIR}")


if __name__ == "__main__":
    main()

