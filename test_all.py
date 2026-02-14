#!/usr/bin/env python3
"""
test_all.py

End-to-end regression test:
1) Generate calibration videos across multiple ratios + resolution tiers
   (calls make_calibration_video_multi_res.py)
2) Create corresponding local "job" YAML files (default params, overridable by flags)
3) Run them with run_job.py (runner) and collect logs/results

Defaults chosen for broad compatibility (small GPUs):
- preview mode, preview_interval=3.0s, encoder=vits, batch=1
- input_size=518, max_shift=24, alpha=0.90
- sbs_w=3840
- quality auto with cq=28, nv_preset=p5
- calibration video generation: 10s, 30fps, crf=34, preset ultrafast, tune stillimage

Usage:
  source .venv/bin/activate

  python test_all.py \
    --source calibration_pattern_3840x2160.png \
    --converter ./mono_to_sbs_pico4_v2_autosize.py \
    --depth_repo ./Depth-Anything-V2 \
    --checkpoints ./checkpoints

You can override any conversion parameter, plus generation params.

Output:
- calibration videos in ./calib_videos (or --gen_outdir)
- jobs in ./jobs_test_all_<timestamp> (or --jobs_dir)
- successful jobs moved to --done_dir
- logs in --log_dir and per-job logs in runtime.log_dir
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

try:
    import yaml  # type: ignore
except Exception:
    yaml = None


def stamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def which_or_die(name: str) -> str:
    exe = shutil.which(name)
    if not exe:
        raise SystemExit(f"Required tool not found in PATH: {name}")
    return exe


def safe_dump_yaml(data: Dict[str, Any]) -> str:
    if yaml is None:
        raise SystemExit("PyYAML not installed in venv. Install with: pip install pyyaml")
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)


def write_job_yaml(path: Path, job: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(safe_dump_yaml(job), encoding="utf-8")


def build_job(name: str, input_path: Path, args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "job": {"name": name},
        "source": {
            "kind": "local",
            "path": str(input_path),
        },
        "output": {
            "dir": args.output_dir,
            "name_mode": "auto",
        },
        "runtime": {
            "converter_path": args.converter,
            "depth_repo": args.depth_repo,
            "checkpoints_dir": args.checkpoints,
            "log_dir": args.log_dir,
        },
        "convert": {
            "mode": args.mode,
            "preview_interval": args.preview_interval,
            "sbs_w": args.sbs_w,
            "encoder": args.encoder,
            "batch": args.batch,
            "input_size": args.input_size,
            "max_shift": args.max_shift,
            "alpha": args.alpha,
        },
        "quality": {
            "mode": args.quality_mode,
            "cq": args.cq,
            "nv_preset": args.nv_preset,
        },
    }


def run_generate(args: argparse.Namespace) -> None:
    which_or_die("ffmpeg")
    src = Path(args.source).expanduser().resolve()
    if not src.exists():
        raise SystemExit(f"Calibration image not found: {src}")

    gen_script = Path(args.gen_script).expanduser().resolve()
    if not gen_script.exists():
        raise SystemExit(f"Generator script not found: {gen_script}")

    outdir = Path(args.gen_outdir).expanduser().resolve()
    ensure_dir(outdir)

    # Skip generation if we already have generated videos and overwrite not requested
    existing = sorted(outdir.rglob("*.mp4"))
    if existing and not args.gen_overwrite:
        print(f"== Skipping generation (found {len(existing)} existing video(s) under {outdir}) ==")
        print("   Use --gen_overwrite to force regeneration.\n")
        return

    cmd = [
        sys.executable, str(gen_script),
        "--source", str(src),
        "--duree", str(args.gen_duration),
        "--outdir", str(outdir),
        "--fps", str(args.gen_fps),
        "--crf", str(args.gen_crf),
        "--preset", str(args.gen_preset),
        "--tune", str(args.gen_tune),
    ]
    if args.gen_overwrite:
        cmd.append("--overwrite")

    print("== Generating calibration videos ==")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)
    print("== Generation done ==\n")

def discover_videos(gen_outdir: Path) -> List[Path]:
    # Videos are in subfolders per tier. Find mp4 recursively.
    vids = sorted(gen_outdir.rglob("*.mp4"))
    return vids


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate calibration videos, create jobs, run them all.")

    # Inputs & scripts
    ap.add_argument("--source", default="calibration_pattern_3840x2160.png", help="Calibration image path.")
    ap.add_argument("--gen_script", default="make_calibration_video_multi_res.py", help="Generator script path.")
    ap.add_argument("--gen_outdir", default="calib_videos", help="Where to write generated calibration videos.")
    ap.add_argument("--converter", default="mono_to_sbs_pico4_v2_autosize.py", help="Converter script path.")
    ap.add_argument("--depth_repo", default="Depth-Anything-V2", help="Depth-Anything-V2 repo path.")
    ap.add_argument("--checkpoints", default="checkpoints", help="Checkpoints directory path.")

    # Generation defaults
    ap.add_argument("--gen_duration", type=int, default=10, help="Calibration video duration in seconds (default: 10).")
    ap.add_argument("--gen_fps", type=int, default=30, help="Calibration video FPS (default: 30).")
    ap.add_argument("--gen_crf", type=int, default=34, help="Calibration x264 CRF (default: 34).")
    ap.add_argument("--gen_preset", default="ultrafast", help="Calibration x264 preset (default: ultrafast).")
    ap.add_argument("--gen_tune", default="stillimage", help="Calibration x264 tune (default: stillimage).")
    ap.add_argument("--gen_overwrite", action="store_true", help="Overwrite existing calibration videos.")

    # Job / runner directories
    ap.add_argument("--jobs_dir", default="", help="Jobs directory. Default: ./jobs_test_all_<timestamp>")
    ap.add_argument("--done_dir", default="job_done", help="Done jobs directory (default: ./job_done)")
    ap.add_argument("--log_dir", default="logs", help="Log directory (default: ./logs)")
    ap.add_argument("--output_dir", default="outputs", help="Output SBS directory (default: ./outputs)")
    ap.add_argument("--runner", default="run_job.py", help="Runner script path (default: run_job.py)")
    ap.add_argument("--max_jobs", type=int, default=0, help="If >0, limit number of jobs processed.")

    # Conversion defaults (overridable)
    ap.add_argument("--mode", choices=["preview", "full"], default="preview", help="Conversion mode (default: preview)")
    ap.add_argument("--preview_interval", type=float, default=3.0, help="Preview interval in seconds (default: 3.0)")
    ap.add_argument("--encoder", choices=["vits", "vitb", "vitl", "vitg"], default="vits", help="DepthAnything encoder (default: vits)")
    ap.add_argument("--batch", type=int, default=1, help="Batch size (default: 1 for small GPUs)")
    ap.add_argument("--input_size", type=int, default=518, help="Model input size (default: 518)")
    ap.add_argument("--max_shift", type=int, default=24, help="Stereo max shift (default: 24)")
    ap.add_argument("--alpha", type=float, default=0.90, help="Temporal smoothing alpha (default: 0.90)")
    ap.add_argument("--sbs_w", type=int, default=3840, help="SBS total width (default: 3840)")

    ap.add_argument("--quality_mode", choices=["auto", "nvenc", "none"], default="auto", help="Quality mode (default: auto)")
    ap.add_argument("--cq", type=int, default=28, help="NVENC CQ (default: 28)")
    ap.add_argument("--nv_preset", default="p5", help="NVENC preset p1..p7 (default: p5)")

    args = ap.parse_args()

    # Resolve important paths
    gen_outdir = Path(args.gen_outdir).expanduser().resolve()
    ensure_dir(gen_outdir)

    # Create a unique jobs dir unless provided
    if not args.jobs_dir:
        args.jobs_dir = f"jobs_test_all_{stamp()}"
    jobs_dir = Path(args.jobs_dir).expanduser().resolve()
    ensure_dir(jobs_dir)

    # Ensure required tools
    which_or_die("ffmpeg")
    which_or_die("ffprobe")

    # Generate calibration videos
    run_generate(args)

    videos = discover_videos(gen_outdir)
    if not videos:
        raise SystemExit(f"No generated videos found under: {gen_outdir}")

    # Optionally limit number of jobs created
    if args.max_jobs and args.max_jobs > 0:
        videos = videos[: args.max_jobs]

    # Create jobs
    print(f"== Creating {len(videos)} job(s) in {jobs_dir} ==")
    for v in videos:
        job_name = "job_" + v.stem
        job = build_job(job_name, v, args)
        job_path = jobs_dir / f"{job_name}.yaml"
        write_job_yaml(job_path, job)
    print("== Jobs created ==\n")

    # Run jobs
    runner = Path(args.runner).expanduser().resolve()
    if not runner.exists():
        raise SystemExit(f"Runner script not found: {runner}")

    cmd = [
        sys.executable, str(runner),
        "--jobs_dir", str(jobs_dir),
        "--done_dir", str(Path(args.done_dir).expanduser().resolve()),
        "--log_dir", str(Path(args.log_dir).expanduser().resolve()),
    ]
    if args.max_jobs and args.max_jobs > 0:
        cmd += ["--max_jobs", str(args.max_jobs)]

    print("== Running jobs ==")
    print(" ".join(cmd))
    subprocess.run(cmd, check=False)
    print("\n== test_all done ==")
    print(f"Jobs dir:  {jobs_dir}")
    print(f"Videos:    {gen_outdir}")
    print(f"Outputs:   {Path(args.output_dir).expanduser().resolve()}")
    print(f"Done jobs: {Path(args.done_dir).expanduser().resolve()}")
    print(f"Logs:      {Path(args.log_dir).expanduser().resolve()}")


if __name__ == "__main__":
    main()
