#!/usr/bin/env python3
"""
install.py (Local / Linux)

Stage 1 installer for a LOCAL setup:
- Creates a Python venv
- Installs Python dependencies into the venv
- Verifies system tools (ffmpeg/ffprobe/git)
- Clones Depth-Anything-V2 (needed for depth_anything_v2.* imports)
- Downloads Depth-Anything-V2 model checkpoints into ./checkpoints/
- (Optional) runs a quick import sanity-check inside the venv

This script does NOT download your converter (mono_to_sbs_*.py). In local dev you provide it yourself.

Example:
  python install.py --venv .venv --encoder vits,vitb --depth_repo ./Depth-Anything-V2

Torch installation:
- Default: does NOT install torch automatically (because CUDA wheels depend on your system).
- Use --torch cpu to install CPU-only torch.
- Use --torch cu121 (or cu118, cu124...) to install from PyTorch's official CUDA index.

Examples:
  python install.py --torch cpu
  python install.py --torch cu121
"""

from __future__ import annotations
import argparse
import os
import platform
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Dict, List, Tuple


# ---- Defaults ----

# Minimal deps used by notebook + mono_to_sbs script
DEFAULT_PIP_DEPS = [
    'pip',
    'setuptools',
    'wheel',
    'numpy>=2.0,<2.2',
    'tqdm',
    'opencv-python-headless',
    'pyyaml',          # used later for config/run pipeline
    'requests',        # for downloads
    'yt-dlp',          # optional, but handy locally too
    'gdown',           # optional
    'prompt_toolkit',  # wizard UI (new_job.py)
]

DEPTH_REPO_URL = "https://github.com/DepthAnything/Depth-Anything-V2.git"

# Model URLs (Hugging Face "resolve" links)
# NOTE: vitg (giant) has been intermittently unavailable; we treat it as "custom only".
MODEL_URLS: Dict[str, str] = {
    "vits": "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth?download=true",
    "vitb": "https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth?download=true",
    "vitl": "https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true",
    # "vitg": custom only for now
}


# ---- Utilities ----

def sh(cmd: List[str], cwd: Path | None = None, env: dict | None = None, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, check=check)

def which_or_warn(name: str) -> bool:
    exe = shutil.which(name)
    return bool(exe)

def venv_python(venv_dir: Path) -> Path:
    if platform.system().lower().startswith("win"):
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"

def venv_pip(venv_dir: Path) -> Path:
    if platform.system().lower().startswith("win"):
        return venv_dir / "Scripts" / "pip.exe"
    return venv_dir / "bin" / "pip"

def ensure_venv(venv_dir: Path) -> None:
    py = venv_python(venv_dir)
    if py.exists():
        return
    print(f"[*] Creating venv: {venv_dir}")
    sh([sys.executable, "-m", "venv", str(venv_dir)])
    if not py.exists():
        raise SystemExit(f"Failed to create venv (python not found): {py}")

def pip_install(venv_dir: Path, pkgs: List[str]) -> None:
    pip = venv_pip(venv_dir)
    print(f"[*] Installing packages into venv ({len(pkgs)}):")
    print("    " + " ".join(pkgs))
    sh([str(pip), "install", "--upgrade"] + pkgs)

def pip_install_torch(venv_dir: Path, torch_mode: str) -> None:
    """
    torch_mode:
      - "skip": do nothing
      - "cpu": install CPU wheels from official index
      - "cu121"/"cu118"/...: install CUDA wheels from the corresponding index
    """
    pip = venv_pip(venv_dir)

    if torch_mode == "skip":
        print("[*] Torch: skip (you will install torch manually)")
        return

    if torch_mode == "cpu":
        print("[*] Torch: installing CPU-only wheels (torch + torchvision)")
        sh([str(pip), "install", "--upgrade", "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cpu"])
        return

    # CUDA
    if torch_mode.startswith("cu"):
        cu = torch_mode[2:]
        idx = f"https://download.pytorch.org/whl/cu{cu}"
        print(f"[*] Torch: installing CUDA wheels from {idx}")
        sh([str(pip), "install", "--upgrade", "torch", "torchvision", "--index-url", idx])
        return

    raise SystemExit(f"Unknown --torch mode: {torch_mode} (use skip|cpu|cu121|cu118|...)")

def git_clone_or_update(repo_url: str, dest: Path, update: bool) -> None:
    if dest.exists() and (dest / ".git").exists():
        if update:
            print(f"[*] Updating repo: {dest}")
            sh(["git", "pull", "--rebase"], cwd=dest)
        else:
            print(f"[*] Repo already present: {dest} (update disabled)")
        return
    if dest.exists() and not (dest / ".git").exists():
        raise SystemExit(f"Destination exists but is not a git repo: {dest}")
    print(f"[*] Cloning repo: {repo_url} -> {dest}")
    sh(["git", "clone", repo_url, str(dest)])

def download(url: str, dest: Path) -> None:
    """
    Download using curl/wget if available; fallback to python requests inside venv later.
    We use curl/wget here because it avoids importing extra deps in the installer itself.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and dest.stat().st_size > 0:
        print(f"[*] Model already present: {dest.name} ({dest.stat().st_size/1e6:.1f} MB)")
        return

    curl = shutil.which("curl")
    wget = shutil.which("wget")

    print(f"[*] Downloading model: {dest.name}")
    print(f"    URL: {url}")

    if curl:
        sh([curl, "-L", "-o", str(dest), url])
        return
    if wget:
        sh([wget, "-O", str(dest), url])
        return

    raise SystemExit("Neither curl nor wget found. Install one of them, or download models manually.")

def import_check(venv_dir: Path, depth_repo: Path) -> None:
    py = venv_python(venv_dir)
    code = textwrap.dedent(f"""
    import sys, os
    import numpy as np
    import cv2
    try:
        import torch
        torch_ok = True
    except Exception as e:
        torch_ok = False
        print("torch import failed:", e)

    # ensure Depth-Anything-V2 is importable
    sys.path.insert(0, r"{depth_repo}")
    from depth_anything_v2.dpt import DepthAnythingV2

    print("numpy", np.__version__)
    print("cv2", cv2.__version__)
    print("torch", "OK" if torch_ok else "MISSING")
    print("DepthAnythingV2 import OK")
    """).strip()
    print("[*] Running import check inside venv...")
    sh([str(py), "-c", code])



def add_repo_to_venv_path(venv_dir: Path, depth_repo: Path, pth_name: str = "depth_anything_v2_repo.pth") -> Path:
    """
    Permanently make Depth-Anything-V2 importable inside the venv by writing a .pth file
    into the venv's site-packages that points to the repo directory.

    This avoids having to set PYTHONPATH at runtime.
    """
    py = venv_python(venv_dir)
    code = textwrap.dedent(f"""
    import site
    from pathlib import Path
    depth_repo = Path(r"{depth_repo}").resolve()

    candidates = []
    try:
        candidates += site.getsitepackages()
    except Exception:
        pass
    try:
        sp = site.getusersitepackages()
        if sp:
            candidates.append(sp)
    except Exception:
        pass

    site_pkgs = None
    for c in candidates:
        p = Path(c)
        if p.exists() and "site-packages" in p.as_posix():
            site_pkgs = p
            break
    if site_pkgs is None:
        raise SystemExit("Could not locate site-packages. Are you running inside the venv?")

    pth_path = site_pkgs / "{pth_name}"
    pth_path.write_text(str(depth_repo) + "\\n", encoding="utf-8")
    print(str(pth_path))
    """).strip()
    # Returns the pth path printed by the subprocess.
    p = subprocess.run([str(py), "-c", code], check=True, stdout=subprocess.PIPE, text=True)
    return Path(p.stdout.strip())


def print_next_steps(venv_dir: Path, depth_repo: Path) -> None:
    act = f"source {venv_dir}/bin/activate"
    if platform.system().lower().startswith("win"):
        act = f"{venv_dir}\\Scripts\\activate"
    msg = f"""
    ✅ Install finished.

    Next steps:
      1) Activate venv:
           {act}

      2) Run your converter (example):
           python mono_to_sbs_pico4_v2_autosize.py input.mp4 output_sbs.mp4 --sbs_w 3840 --encoder vitb

    Notes:
      - Depth-Anything-V2 repo is at: {depth_repo}
      - Checkpoints are in: ./checkpoints (expected by our converter by default)
      - If torch was skipped, install it now (CPU or CUDA):
           pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
           # or CUDA example:
           pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    """
    print(textwrap.dedent(msg).strip())


def main() -> None:
    ap = argparse.ArgumentParser(description="Local installer for Depth-Anything mono→SBS pipeline (stage 1).")
    ap.add_argument("--venv", default=".venv", help="Venv directory (default: .venv)")
    ap.add_argument("--depth_repo", default="Depth-Anything-V2", help="Where to clone Depth-Anything-V2 (default: ./Depth-Anything-V2)")
    ap.add_argument("--update_repo", action="store_true", help="If repo exists, git pull --rebase")
    ap.add_argument("--checkpoints", default="checkpoints", help="Where to store .pth models (default: ./checkpoints)")
    ap.add_argument("--encoder", default="vits,vitb", help="Comma list of encoders to download (vits,vitb,vitl). Default: vits,vitb")
    ap.add_argument("--torch", default="skip", help="Install torch: skip|cpu|cu121|cu118|... (default: skip)")
    ap.add_argument("--no_import_check", action="store_true", help="Skip import sanity-check step")
    ap.add_argument("--require_system_tools", action="store_true",
                    help="Fail immediately if required system tools are missing (git, ffmpeg, ffprobe).")
    ap.add_argument("--no_pth", action="store_true", help="Do not write a .pth file to make Depth-Anything-V2 importable inside the venv.")
    args = ap.parse_args()

    venv_dir = Path(args.venv).expanduser().resolve()
    depth_repo = Path(args.depth_repo).expanduser().resolve()
    ckpt_dir = Path(args.checkpoints).expanduser().resolve()

    # System tool checks (warn-only for some)
    # System tool checks
    missing_tools = []
    for tool in ["git", "ffmpeg", "ffprobe"]:
        if not which_or_warn(tool):
            missing_tools.append(tool)

    if missing_tools:
        print("⚠️ Missing system tools:", ", ".join(missing_tools))
        print("   Install them with your distro package manager (example Ubuntu):")
        print("     sudo apt-get update && sudo apt-get install -y git ffmpeg")
        print("   (ffprobe comes with ffmpeg)")
        if args.require_system_tools:
            raise SystemExit("Missing required system tools. Re-run after installing them, or omit --require_system_tools.")



    # 1) venv
    ensure_venv(venv_dir)

    # 2) python deps
    pip_install(venv_dir, DEFAULT_PIP_DEPS)

    # 3) torch (optional)
    pip_install_torch(venv_dir, args.torch)

    # 4) clone Depth-Anything-V2
    if not which_or_warn("git"):
        print("⚠️ git missing; cannot auto-clone Depth-Anything-V2. Please clone it manually:")
        print(f"    git clone {DEPTH_REPO_URL}")
    else:
        git_clone_or_update(DEPTH_REPO_URL, depth_repo, args.update_repo)


    # 4b) Make Depth-Anything-V2 importable inside the venv (writes a .pth file)
    if not args.no_pth:
        if depth_repo.exists():
            try:
                pth_path = add_repo_to_venv_path(venv_dir, depth_repo)
                print(f"[*] Wrote venv path file: {pth_path}")
            except subprocess.CalledProcessError as e:
                print("⚠️ Failed to write .pth file for Depth-Anything-V2.")
                raise SystemExit(e.returncode)
        else:
            print("⚠️ Depth-Anything-V2 repo not present; cannot write .pth file.")

    # 5) models
    wanted = [x.strip().lower() for x in args.encoder.split(",") if x.strip()]
    for enc in wanted:
        if enc not in MODEL_URLS:
            print(f"⚠️ Unknown/unsupported encoder '{enc}' for auto-download.")
            if enc == "vitg":
                print("   vitg (giant) is not consistently available on Hugging Face; download it manually and place it in ./checkpoints.")
            continue
        url = MODEL_URLS[enc]
        dest = ckpt_dir / f"depth_anything_v2_{enc}.pth"
        download(url, dest)

    # 6) import check
    if not args.no_import_check:
        if depth_repo.exists():
            try:
                import_check(venv_dir, depth_repo)
            except subprocess.CalledProcessError as e:
                print("⚠️ Import check failed.")
                raise SystemExit(e.returncode)
        else:
            print("⚠️ Depth-Anything-V2 repo not present; skipping import check.")

    print_next_steps(venv_dir, depth_repo)


if __name__ == "__main__":
    main()
