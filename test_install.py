#!/usr/bin/env python3
"""
test_install_progress_v3.py

Fixes the "unrecognized arguments: --crf/--preset" problem by:
- Detecting which quality flags your converter supports (via `--help`)
- Using NVENC-style flags when available: --cq and --nv_preset
- Otherwise skipping quality flags entirely

Also keeps:
- ffmpeg % progress for input video generation
- live streaming of converter output

Usage:
  source .venv/bin/activate
  python test_install_progress_v3.py \
    --image calibration_pattern_3840x2160.png \
    --converter ./mono_to_sbs_pico4_v2_autosize.py \
    --depth_repo ./Depth-Anything-V2

Optional quality flags (applied only if supported by converter):
  --out_cq 24 --nv_preset p5
"""

from __future__ import annotations
import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional


def which_or_die(name: str) -> str:
    exe = shutil.which(name)
    if not exe:
        raise SystemExit(f"Required tool not found in PATH: {name}")
    return exe


def human_mb(n: int) -> str:
    return f"{n/1e6:.2f} MB"


def run_ffmpeg_with_progress(cmd: List[str], total_seconds: float, label: str) -> None:
    cmd = cmd[:]
    cmd.insert(1, "-hide_banner")
    cmd.insert(2, "-loglevel"); cmd.insert(3, "error")
    cmd.insert(4, "-progress"); cmd.insert(5, "pipe:1")
    cmd.insert(6, "-nostats")

    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)

    last_pct = -1
    last_line = ""
    try:
        if not p.stdout:
            raise RuntimeError("ffmpeg stdout unavailable")
        for line in p.stdout:
            line = line.strip()
            if not line:
                continue
            last_line = line
            if line.startswith("out_time_ms="):
                try:
                    out_ms = int(line.split("=", 1)[1])
                    out_s = out_ms / 1_000_000.0
                    pct = int(min(100, max(0, round((out_s / max(0.0001, total_seconds)) * 100))))
                    if pct != last_pct:
                        last_pct = pct
                        print(f"\r{label}: {pct:3d}% ({out_s:6.2f}s / {total_seconds:.2f}s)", end="", flush=True)
                except Exception:
                    pass
            elif line.startswith("progress=") and "end" in line:
                break

        rc = p.wait()
        if rc != 0:
            err = (p.stderr.read() if p.stderr else "") or ""
            raise SystemExit(f"\nffmpeg failed (rc={rc}). Last progress line: '{last_line}'\n{err}")
    finally:
        if p.stdout:
            p.stdout.close()
        if p.stderr:
            p.stderr.close()

    print(f"\r{label}: 100% ({total_seconds:.2f}s / {total_seconds:.2f}s) ✔")


def make_1080p_video(ffmpeg: str, image: Path, out_mp4: Path, seconds: int, fps: int, crf: int, preset: str) -> None:
    vf = "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2"
    cmd = [
        ffmpeg, "-y",
        "-loop", "1",
        "-t", str(seconds),
        "-i", str(image),
        "-vf", vf,
        "-r", str(fps),
        "-pix_fmt", "yuv420p",
        "-c:v", "libx264",
        "-preset", preset,
        "-crf", str(crf),
        "-tune", "stillimage",
        "-movflags", "+faststart",
        str(out_mp4),
    ]
    run_ffmpeg_with_progress(cmd, total_seconds=float(seconds), label="Generate 1080p input")


def stream_process(cmd: List[str], label: str, env: dict) -> None:
    print(f"{label}: running:\n  " + " ".join(cmd))
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
    if not p.stdout:
        raise RuntimeError("subprocess stdout unavailable")
    try:
        for line in p.stdout:
            print(line, end="")
    finally:
        p.stdout.close()
    rc = p.wait()
    if rc != 0:
        raise SystemExit(f"{label}: FAILED (exit code {rc})")


def converter_help(converter: Path) -> str:
    p = subprocess.run([sys.executable, str(converter), "--help"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return p.stdout or ""


def supports_flag(help_text: str, flag: str) -> bool:
    # naive but effective: argparse prints flags like "--cq CQ" or "--nv_preset NV_PRESET"
    return flag in help_text


def convert_to_sbs(
    converter: Path,
    depth_repo: Path,
    inp: Path,
    outp: Path,
    sbs_w: int,
    encoder: str,
    batch: int,
    input_size: int,
    max_shift: int,
    alpha: float,
    out_cq: Optional[int],
    nv_preset: Optional[str],
    preview: bool,
) -> None:
    help_text = converter_help(converter)

    cmd = [
        sys.executable, str(converter),
        str(inp), str(outp),
        "--sbs_w", str(sbs_w),
        "--encoder", encoder,
        "--batch", str(batch),
        "--input_size", str(input_size),
        "--max_shift", str(max_shift),
        "--alpha", str(alpha),
    ]
    if preview and supports_flag(help_text, "--preview"):
        cmd.append("--preview")

    # Quality flags: use what the converter supports
    if out_cq is not None and supports_flag(help_text, "--cq"):
        cmd += ["--cq", str(out_cq)]
    if nv_preset is not None and supports_flag(help_text, "--nv_preset"):
        cmd += ["--nv_preset", nv_preset]

    # Inject Depth-Anything-V2 path into PYTHONPATH so "import depth_anything_v2" works
    env = dict(os.environ)
    py_path = env.get("PYTHONPATH", "")
    parts = [str(depth_repo)]
    if py_path:
        parts.append(py_path)
    env["PYTHONPATH"] = os.pathsep.join(parts)

    stream_process(cmd, label="Convert to SBS", env=env)


def main() -> None:
    ap = argparse.ArgumentParser(description="Local install smoke-test with progress (auto-detect converter flags).")
    ap.add_argument("--image", required=True, help="Calibration image path (png/jpg).")
    ap.add_argument("--converter", required=True, help="Path to converter script.")
    ap.add_argument("--depth_repo", default="Depth-Anything-V2", help="Path to Depth-Anything-V2 repo (default: ./Depth-Anything-V2)")
    ap.add_argument("--outdir", default="_test_install", help="Output working directory.")

    ap.add_argument("--duration", type=int, default=10, help="Test video duration seconds.")
    ap.add_argument("--fps", type=int, default=30, help="FPS for test video.")

    ap.add_argument("--video_crf", type=int, default=34, help="CRF for generated input video.")
    ap.add_argument("--video_preset", default="ultrafast", help="Preset for generated input video.")

    ap.add_argument("--sbs_w", type=int, default=3840, help="SBS total width (3840 => 1080p per eye).")
    ap.add_argument("--encoder", default="vitb", help="DepthAnything encoder.")
    ap.add_argument("--batch", type=int, default=1, help="Depth batch size.")
    ap.add_argument("--input_size", type=int, default=518, help="Depth model input size.")
    ap.add_argument("--max_shift", type=int, default=24, help="Stereo max shift.")
    ap.add_argument("--alpha", type=float, default=0.90, help="Temporal smoothing alpha.")
    ap.add_argument("--out_cq", type=int, default=24, help="NVENC constant quality (only if converter supports --cq). Default: 24")
    ap.add_argument("--nv_preset", default="p5", help="NVENC preset p1..p7 (only if converter supports --nv_preset). Default: p5")
    ap.add_argument("--preview", action="store_true", help="Run converter in --preview mode (if supported).")

    args = ap.parse_args()

    ffmpeg = which_or_die("ffmpeg")

    image = Path(args.image).expanduser().resolve()
    converter = Path(args.converter).expanduser().resolve()
    depth_repo = Path(args.depth_repo).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()

    if not image.exists():
        raise SystemExit(f"Image not found: {image}")
    if not converter.exists():
        raise SystemExit(f"Converter not found: {converter}")
    if not depth_repo.exists():
        raise SystemExit(f"Depth-Anything-V2 repo not found: {depth_repo} (did you run install.py?)")

    outdir.mkdir(parents=True, exist_ok=True)

    input_vid = outdir / "input_1080p_16x9.mp4"
    out_sbs = outdir / "sbs_1080p.mp4"

    print("== Step 1/2: Generate 16:9 1080p test video ==")
    t0 = time.perf_counter()
    make_1080p_video(ffmpeg, image, input_vid, args.duration, args.fps, args.video_crf, args.video_preset)
    t1 = time.perf_counter()
    print(f"Created: {input_vid} ({human_mb(input_vid.stat().st_size)}) in {t1-t0:.2f}s\n")

    print("== Step 2/2: Convert to SBS 1080p ==")
    t2 = time.perf_counter()
    convert_to_sbs(
        converter=converter,
        depth_repo=depth_repo,
        inp=input_vid,
        outp=out_sbs,
        sbs_w=args.sbs_w,
        encoder=args.encoder,
        batch=args.batch,
        input_size=args.input_size,
        max_shift=args.max_shift,
        alpha=args.alpha,
        out_cq=args.out_cq,
        nv_preset=args.nv_preset,
        preview=args.preview,
    )
    t3 = time.perf_counter()

    size_in = input_vid.stat().st_size
    size_out = out_sbs.stat().st_size if out_sbs.exists() else 0
    ratio = (size_out / size_in) if size_in > 0 and size_out > 0 else None

    print("\n== Result ==")
    print(f"Output: {out_sbs} ({human_mb(size_out)})")
    if ratio is not None:
        print(f"Ratio (out/in): {ratio:.3f}")
    print(f"Convert time: {t3-t2:.2f}s")
    print("\n✅ Smoke-test complete.")


if __name__ == "__main__":
    main()
