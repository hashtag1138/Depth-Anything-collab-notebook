#!/usr/bin/env python3
"""
run_job.py

Batch runner for job YAML files created by new_job.py.

Behavior:
- Scans ./jobs (or --jobs_dir) for *.yaml
- For each job:
  - Validates required files/tools (ffmpeg/ffprobe, converter, Depth-Anything-V2 repo, checkpoints)
  - Prepares input video:
      * local: verify exists
      * ytdlp: download via yt-dlp into download_dir with a sanitized filename
  - Computes output filename (name_mode=auto): <sanitized_stem>_sbs.mp4 in output.dir
  - Launches the converter, streaming output
  - On success: moves the job yaml to ./job_done/
  - On failure: logs error and continues to next job
- Prints progress + writes a log file with successes/errors

Usage:
  source .venv/bin/activate
  python run_job.py
  python run_job.py --jobs_dir jobs --done_dir job_done

Notes:
- This runner intentionally keeps failed jobs in place so you can fix and re-run.
- It injects Depth-Anything-V2 into PYTHONPATH for the converter.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml  # type: ignore
except Exception:
    yaml = None

try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None


VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi", ".ts", ".m2ts"}


def ts() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def stamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def which(name: str) -> Optional[str]:
    return shutil.which(name)


def sanitize_filename(name: str, max_len: int = 140) -> str:
    """
    Make a filesystem-friendly basename (no extension).
    Keeps: letters, digits, _, -, .
    """
    name = name.strip()
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("._-")
    if not name:
        name = "video"
    return name[:max_len]


def read_yaml(path: Path) -> Dict[str, Any]:
    if yaml is None:
        raise SystemExit("PyYAML is required to run jobs. Install it in the venv: pip install pyyaml")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("Job YAML root must be a mapping")
    return data


def write_log_line(log_path: Path, line: str) -> None:
    ensure_dir(log_path.parent)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line.rstrip("\n") + "\n")


def job_log_path(job: Dict[str, Any], default_log_dir: Path) -> Path:
    # prefer runtime.log_dir if present
    log_dir = default_log_dir
    try:
        ld = job.get("runtime", {}).get("log_dir")
        if ld:
            log_dir = Path(ld).expanduser().resolve()
    except Exception:
        pass
    ensure_dir(log_dir)
    return log_dir / f"run_job_{stamp()}.log"


def convert_help(converter: Path) -> str:
    p = subprocess.run([sys.executable, str(converter), "--help"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return p.stdout or ""


def supports_flag(help_text: str, flag: str) -> bool:
    return flag in help_text


def validate_job(job: Dict[str, Any]) -> Tuple[bool, str]:
    # required structure
    if "source" not in job or "runtime" not in job or "convert" not in job or "output" not in job:
        return False, "Missing required top-level keys (source/runtime/convert/output)"

    src = job["source"]
    if src.get("kind") not in {"local", "ytdlp"}:
        return False, f"Unsupported source.kind: {src.get('kind')}"

    rt = job["runtime"]
    conv_path = Path(rt.get("converter_path", "")).expanduser()
    if not conv_path.exists():
        return False, f"Converter not found: {conv_path}"

    depth_repo = Path(rt.get("depth_repo", "")).expanduser()
    if not depth_repo.exists():
        return False, f"Depth-Anything-V2 repo not found: {depth_repo}"

    ckpt_dir = Path(rt.get("checkpoints_dir", "")).expanduser()
    if not ckpt_dir.exists():
        return False, f"Checkpoints dir not found: {ckpt_dir}"

    enc = job.get("convert", {}).get("encoder", "vitb")
    if enc in {"vits", "vitb", "vitl"}:
        pth = ckpt_dir / f"depth_anything_v2_{enc}.pth"
        if not pth.exists():
            return False, f"Missing checkpoint for encoder '{enc}': {pth}"
    # vitg may be custom; we don't hard fail if missing

    if which("ffmpeg") is None or which("ffprobe") is None:
        return False, "ffmpeg/ffprobe not found in PATH"

    if src["kind"] == "local":
        ip = Path(str(src.get("path", ""))).expanduser()
        if not ip.exists():
            return False, f"Input file does not exist: {ip}"
        if ip.suffix.lower() not in VIDEO_EXTS:
            return False, f"Input extension not recognized: {ip.suffix}"


    # preview interval sanity
    try:
        if str(job.get("convert", {}).get("mode", "preview")).lower() == "preview":
            pi = job.get("convert", {}).get("preview_interval", 30)
            if pi is not None:
                pi = float(pi)
                if pi <= 0:
                    return False, "convert.preview_interval must be >= 1"
    except Exception:
        return False, "convert.preview_interval must be an integer"
    # video codec sanity (optional; only if provided)
    try:
        q = job.get("quality", {}) or {}
        vc = q.get("video_codec", None) or job.get("convert", {}).get("video_codec", None)
        if vc is not None:
            vc_s = str(vc).lower()
            if vc_s not in {"auto", "nvenc", "x264"}:
                return False, "quality.video_codec must be one of: auto|nvenc|x264"
        crf = q.get("crf", None) or job.get("convert", {}).get("crf", None)
        if crf is not None:
            _ = int(crf)
    except Exception:
        return False, "quality.video_codec/crf invalid"

    if src["kind"] == "ytdlp":
        if which("yt-dlp") is None:
            # You can also have yt_dlp module, but CLI is simplest.
            # We fail early so user knows.
            return False, "yt-dlp not found in PATH (pip install yt-dlp)"

    return True, "ok"


def download_with_ytdlp(url: str, download_dir: Path, log_path: Path) -> Path:
    """
    Download URL using yt-dlp into download_dir with a sanitized filename.
    We use the video id to avoid weird titles. Output template:
      <download_dir>/video_<id>.%(ext)s

    Progress:
      - Uses yt-dlp native progress output (with --newline so it's line-based)
      - Streams output to console and to the log file

    Returns the downloaded file path.
    """
    ensure_dir(download_dir)

    # First get id (fast) to determine a stable output path
    p = subprocess.run(["yt-dlp", "--print", "id", url], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"yt-dlp failed to probe id: {p.stderr.strip() or p.stdout.strip()}")
    vid = sanitize_filename(p.stdout.strip() or "video")

    out_tmpl = str(download_dir / f"video_{vid}.%(ext)s")

    # Download with progress lines
    cmd = ["yt-dlp", "--newline", "-o", out_tmpl, "--no-playlist", url]
    write_log_line(log_path, f"[{ts()}] YTDLP: {' '.join(cmd)}")

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    if not proc.stdout:
        raise RuntimeError("Failed to capture yt-dlp output")

    try:
        for line in proc.stdout:
            # Show progress as yt-dlp prints it (line-based with --newline)
            print(line, end="")
            write_log_line(log_path, line.rstrip("\n"))
    finally:
        proc.stdout.close()

    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"yt-dlp download failed (exit code {rc}). See log: {log_path}")

    # Find the resulting file: match video_<id>.* and pick the newest
    matches = sorted(download_dir.glob(f"video_{vid}.*"), key=lambda x: x.stat().st_mtime, reverse=True)
    if not matches:
        raise RuntimeError(f"yt-dlp reported success but no file found for template: {out_tmpl}")
    return matches[0]


def run_converter(job: Dict[str, Any], input_video: Path, output_video: Path, log_path: Path) -> None:
    rt = job["runtime"]
    cvt = job["convert"]
    qual = job.get("quality", {}) or {}

    converter = Path(rt["converter_path"]).expanduser().resolve()
    depth_repo = Path(rt["depth_repo"]).expanduser().resolve()

    help_text = convert_help(converter)

    cmd: List[str] = [
        sys.executable, str(converter),
        str(input_video), str(output_video),
        "--sbs_w", str(int(cvt.get("sbs_w", 3840))),
        "--encoder", str(cvt.get("encoder", "vitb")),
        "--batch", str(int(cvt.get("batch", 8))),
        "--input_size", str(int(cvt.get("input_size", 518))),
        "--max_shift", str(int(cvt.get("max_shift", 24))),
        "--alpha", str(float(cvt.get("alpha", 0.90))),
    ]

    # Optional: video codec selection / quality for x264 (auto-detect supported flags)
    # - video_codec: auto|nvenc|x264
    # - crf: used for x264
    # - x264_preset: ultrafast/superfast/veryfast/faster/fast/medium/slow/slower/veryslow
    q = job.get("quality", {}) or {}
    vc = q.get("video_codec", None) or cvt.get("video_codec", None)
    if vc is not None and supports_flag(help_text, "--video_codec"):
        cmd += ["--video_codec", str(vc)]

    crf = q.get("crf", None) or cvt.get("crf", None)
    if crf is not None and supports_flag(help_text, "--crf"):
        try:
            cmd += ["--crf", str(int(crf))]
        except Exception:
            pass

    x264_preset = q.get("x264_preset", None) or cvt.get("x264_preset", None)
    if x264_preset is not None and supports_flag(help_text, "--x264_preset"):
        cmd += ["--x264_preset", str(x264_preset)]

    # preview/full
    if str(cvt.get("mode", "preview")).lower() == "preview" and supports_flag(help_text, "--preview"):
        cmd.append("--preview")
        # Optional: preview interval (seconds between preview samples; converter may accept float)
        pi = cvt.get("preview_interval", None)
        if pi is not None and supports_flag(help_text, "--preview_interval"):
            try:
                pi_f = float(pi)
                if pi_f > 0:
                    # keep minimal string (avoid scientific)
                    pi_s = str(int(pi_f)) if abs(pi_f - int(pi_f)) < 1e-9 else str(pi_f)
                    cmd += ["--preview_interval", pi_s]
            except Exception:
                pass

    # quality flags (auto-detect supported flags)
    qmode = str(qual.get("mode", "auto")).lower()
    cq = qual.get("cq")
    nv_preset = qual.get("nv_preset")
    if qmode in {"auto", "nvenc"}:
        if cq is not None and supports_flag(help_text, "--cq"):
            cmd += ["--cq", str(int(cq))]
        if nv_preset and supports_flag(help_text, "--nv_preset"):
            cmd += ["--nv_preset", str(nv_preset)]

    # env: inject Depth-Anything-V2 repo so import depth_anything_v2 works
    env = dict(os.environ)
    env["PYTHONPATH"] = str(depth_repo) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    # run + stream output, also log
    write_log_line(log_path, f"[{ts()}] RUN: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
    if not proc.stdout:
        raise RuntimeError("Failed to capture converter output")

    try:
        for line in proc.stdout:
            print(line, end="")
            write_log_line(log_path, line.rstrip("\n"))
    finally:
        proc.stdout.close()

    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"Converter failed (exit code {rc}). See log: {log_path}")


def compute_auto_output_name(input_path: Path) -> str:
    return sanitize_filename(input_path.stem) + "_sbs.mp4"


def process_one(job_path: Path, done_dir: Path, default_log_dir: Path, idx: int, total: int) -> Tuple[bool, str]:
    """
    Returns (success, message)
    """
    try:
        job = read_yaml(job_path)
    except Exception as e:
        return False, f"YAML parse error: {e}"

    log_path = job_log_path(job, default_log_dir)

    ok, msg = validate_job(job)
    if not ok:
        write_log_line(log_path, f"[{ts()}] JOB {job_path.name} INVALID: {msg}")
        return False, msg

    src = job["source"]
    out_cfg = job["output"]
    rt = job["runtime"]

    # Resolve output dir
    out_dir = Path(out_cfg.get("dir", "./outputs")).expanduser().resolve()
    ensure_dir(out_dir)

    # Prepare input
    try:
        if src["kind"] == "local":
            input_video = Path(src["path"]).expanduser().resolve()
        else:
            url = str(src["url"])
            dl_dir = Path(src.get("download_dir", "./work/downloads")).expanduser().resolve()
            write_log_line(log_path, f"[{ts()}] Downloading via yt-dlp: {url}")
            input_video = download_with_ytdlp(url, dl_dir, log_path).resolve()
            write_log_line(log_path, f"[{ts()}] Downloaded: {input_video}")
    except Exception as e:
        write_log_line(log_path, f"[{ts()}] DOWNLOAD FAILED: {e}")
        return False, f"download failed: {e}"

    # Compute output filename
    out_name_mode = str(out_cfg.get("name_mode", "auto")).lower()
    if out_name_mode != "auto":
        # future: support custom; for now fallback to auto
        out_name_mode = "auto"
    output_video = (out_dir / compute_auto_output_name(input_video)).resolve()

    # Convert
    t0 = time.perf_counter()
    try:
        write_log_line(log_path, f"[{ts()}] START job={job_path.name} input={input_video} output={output_video}")
        run_converter(job, input_video, output_video, log_path)
        dt_s = time.perf_counter() - t0
        in_sz = input_video.stat().st_size if input_video.exists() else 0
        out_sz = output_video.stat().st_size if output_video.exists() else 0
        ratio = (out_sz / in_sz) if in_sz > 0 and out_sz > 0 else 0.0
        write_log_line(log_path, f"[{ts()}] SUCCESS time={dt_s:.2f}s out={output_video} ratio={ratio:.3f}")
    except Exception as e:
        dt_s = time.perf_counter() - t0
        write_log_line(log_path, f"[{ts()}] CONVERT FAILED after {dt_s:.2f}s: {e}")
        return False, f"convert failed: {e}"

    # Move job file to done dir
    try:
        ensure_dir(done_dir)
        dest = done_dir / job_path.name
        # avoid overwrite
        if dest.exists():
            dest = done_dir / f"{job_path.stem}_{stamp()}{job_path.suffix}"
        job_path.replace(dest)
        write_log_line(log_path, f"[{ts()}] Moved job to done: {dest}")
    except Exception as e:
        # Conversion succeeded; moving job is non-fatal
        write_log_line(log_path, f"[{ts()}] WARNING: could not move job to done: {e}")

    return True, str(output_video)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run conversion jobs from YAML files.")
    ap.add_argument("--jobs_dir", default="jobs", help="Directory containing job YAML files (default: ./jobs)")
    ap.add_argument("--done_dir", default="job_done", help="Directory to move successful jobs into (default: ./job_done)")
    ap.add_argument("--log_dir", default="logs", help="Default directory for log files (default: ./logs)")
    ap.add_argument("--pattern", default="*.yaml", help="Glob pattern for job files (default: *.yaml)")
    ap.add_argument("--max_jobs", type=int, default=0, help="If >0, process at most N jobs (useful for testing).")
    args = ap.parse_args()

    jobs_dir = Path(args.jobs_dir).expanduser().resolve()
    done_dir = Path(args.done_dir).expanduser().resolve()
    log_dir = Path(args.log_dir).expanduser().resolve()
    ensure_dir(jobs_dir)
    ensure_dir(log_dir)

    jobs = sorted(jobs_dir.glob(args.pattern))
    if args.max_jobs and args.max_jobs > 0:
        jobs = jobs[: args.max_jobs]

    if not jobs:
        print(f"No job files found in {jobs_dir} matching {args.pattern}")
        return

    # Main session log
    session_log = log_dir / f"run_job_session_{stamp()}.log"
    write_log_line(session_log, f"[{ts()}] START session jobs={len(jobs)} jobs_dir={jobs_dir}")

    successes = 0
    failures = 0

    iterator = jobs
    if tqdm is not None:
        iterator = tqdm(jobs, desc="Jobs", unit="job")

    for i, job_path in enumerate(iterator, start=1):
        header = f"[{ts()}] ({i}/{len(jobs)}) {job_path.name}"
        print("\n" + "=" * len(header))
        print(header)
        print("=" * len(header))
        write_log_line(session_log, header)

        try:
            ok, msg = process_one(job_path, done_dir, log_dir, i, len(jobs))
        except Exception as e:
            ok, msg = False, f"unexpected error: {e}"

        if ok:
            successes += 1
            print(f"✅ SUCCESS: {msg}")
            write_log_line(session_log, f"[{ts()}] SUCCESS {job_path.name} -> {msg}")
        else:
            failures += 1
            print(f"❌ FAIL: {msg}")
            write_log_line(session_log, f"[{ts()}] FAIL {job_path.name}: {msg}")

    write_log_line(session_log, f"[{ts()}] END session success={successes} fail={failures}")
    print("\n=== DONE ===")
    print(f"Success: {successes}")
    print(f"Fail:    {failures}")
    print(f"Session log: {session_log}")


if __name__ == "__main__":
    main()
