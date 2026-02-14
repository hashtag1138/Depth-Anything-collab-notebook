#!/usr/bin/env python3
"""
new_job.py

Interactive wizard to create a conversion "job" YAML in ./jobs/

Features
- Asks a series of questions with defaults
- Supports source: local or ytdlp
- Path autocompletion when prompt_toolkit is available
- Validates inputs (paths, ints, floats)
- Prints a readable summary + the YAML preview
- Asks confirmation before writing

Usage:
  source .venv/bin/activate
  python new_job.py

Output:
  ./jobs/job_YYYYmmdd_HHMMSS.yaml

Notes:
- For best UX, install prompt_toolkit:
    pip install prompt_toolkit
  (You can add it to install.py later.)
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# --- Optional dependencies ---
try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

try:
    from prompt_toolkit.shortcuts import radiolist_dialog  # type: ignore
    from prompt_toolkit.completion import PathCompleter  # type: ignore
    from prompt_toolkit import prompt  # type: ignore
    PTK = True
except Exception:  # pragma: no cover
    PTK = False


VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi", ".ts", ".m2ts"}


def now_stamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def yn(s: str) -> bool:
    return s.strip().lower() in {"y", "yes", "o", "oui"}


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def is_probably_url(u: str) -> bool:
    return bool(re.match(r"^https?://", u.strip(), flags=re.I))


def printable_path(p: Path) -> str:
    try:
        return str(p.resolve())
    except Exception:
        return str(p)


# ---------- Prompt helpers (PTK + fallback) ----------

def choose(title: str, text: str, options: Tuple[Tuple[str, str], ...], default_key: str) -> str:
    """
    options: ((key, label), ...)
    returns selected key
    """
    if PTK:
        values = [(k, label) for k, label in options]
        res = radiolist_dialog(title=title, text=text, values=values).run()
        return res if res is not None else default_key

    # Fallback: CLI numeric choice
    print(f"\n{title}\n{text}")
    for i, (k, label) in enumerate(options, start=1):
        star = " (default)" if k == default_key else ""
        print(f"  {i}. {label}{star}")
    raw = input("> ").strip()
    if not raw:
        return default_key
    try:
        idx = int(raw)
        if 1 <= idx <= len(options):
            return options[idx - 1][0]
    except Exception:
        pass
    # Also accept entering the key directly
    for k, _ in options:
        if raw == k:
            return k
    return default_key


def ask_str(label: str, default: str, help_text: str = "", path: bool = False) -> str:
    if PTK:
        completer = PathCompleter(expanduser=True) if path else None
        prompt_text = f"{label} [{default}]: "
        if help_text:
            print(help_text)
        return (prompt(prompt_text, default=default, completer=completer).strip() or default)

    if help_text:
        print(help_text)
    raw = input(f"{label} [{default}]: ").strip()
    return raw or default


def ask_int(label: str, default: int, min_v: Optional[int] = None, max_v: Optional[int] = None) -> int:
    def validate(x: str) -> int:
        x = x.strip()
        if not x:
            return default
        try:
            n = int(x)
        except Exception:
            raise ValueError("Not an integer")
        if min_v is not None and n < min_v:
            raise ValueError(f"Must be >= {min_v}")
        if max_v is not None and n > max_v:
            raise ValueError(f"Must be <= {max_v}")
        return n

    while True:
        raw = ask_str(label, str(default))
        try:
            return validate(raw)
        except ValueError as e:
            print(f"⚠️  {e}")


def ask_float(label: str, default: float, min_v: Optional[float] = None, max_v: Optional[float] = None) -> float:
    def validate(x: str) -> float:
        x = x.strip()
        if not x:
            return default
        try:
            n = float(x.replace(",", "."))
        except Exception:
            raise ValueError("Not a number")
        if min_v is not None and n < min_v:
            raise ValueError(f"Must be >= {min_v}")
        if max_v is not None and n > max_v:
            raise ValueError(f"Must be <= {max_v}")
        return n

    while True:
        raw = ask_str(label, f"{default}")
        try:
            return validate(raw)
        except ValueError as e:
            print(f"⚠️  {e}")


# ---------- Data model ----------

@dataclass
class SourceLocal:
    kind: str = "local"
    path: str = ""  # required


@dataclass
class SourceYtdlp:
    kind: str = "ytdlp"
    url: str = ""  # required
    download_dir: str = "./work/downloads"
    format: str = "best"


@dataclass
class OutputCfg:
    dir: str = "./outputs"
    name_mode: str = "auto"


@dataclass
class RuntimeCfg:
    converter_path: str = "./mono_to_sbs_pico4_v2_autosize.py"
    depth_repo: str = "./Depth-Anything-V2"
    checkpoints_dir: str = "./checkpoints"
    log_dir: str = "./logs"


@dataclass
class ConvertCfg:
    mode: str = "preview"
    preview_interval: int = 30  # every N frames (only used in preview mode)
    sbs_w: int = 3840
    sbs_h: Optional[int] = None
    encoder: str = "vitb"
    batch: int = 8
    input_size: int = 518
    max_shift: int = 24
    alpha: float = 0.90


@dataclass
class QualityCfg:
    mode: str = "auto"  # auto|nvenc|none
    cq: int = 24
    nv_preset: str = "p5"
    video_codec: str = "auto"  # auto|nvenc|x264
    crf: int = 28               # used when x264 is selected
    x264_preset: str = "fast"  # used when x264 is selected


def build_job_dict(name: str,
                   source: Dict[str, Any],
                   output: OutputCfg,
                   runtime: RuntimeCfg,
                   convert: ConvertCfg,
                   quality: QualityCfg) -> Dict[str, Any]:
    return {
        "job": {"name": name},
        "source": source,
        "output": asdict(output),
        "runtime": asdict(runtime),
        "convert": asdict(convert),
        "quality": asdict(quality),
    }


def to_yaml(data: Dict[str, Any]) -> str:
    if yaml is not None:
        return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
    import json
    return "# PyYAML not installed; using JSON fallback\n" + json.dumps(data, indent=2, ensure_ascii=False)


def validate_paths(job: Dict[str, Any]) -> Tuple[bool, str]:
    src = job["source"]
    if src["kind"] == "local":
        p = Path(src["path"]).expanduser()
        if not p.exists():
            return False, f"Input file does not exist: {p}"
        if p.suffix.lower() not in VIDEO_EXTS:
            return False, f"Input file extension not recognized as video: {p.suffix}"
    else:
        url = src.get("url", "")
        if not is_probably_url(url):
            return False, f"URL doesn't look valid: {url}"

    conv = Path(job["runtime"]["converter_path"]).expanduser()
    if not conv.exists():
        return False, f"Converter not found: {conv}"

    depth_repo = Path(job["runtime"]["depth_repo"]).expanduser()
    if not depth_repo.exists():
        return False, f"Depth-Anything-V2 repo not found: {depth_repo}"

    ckpt = Path(job["runtime"]["checkpoints_dir"]).expanduser()
    if not ckpt.exists():
        return False, f"Checkpoints dir not found: {ckpt}"

    return True, "ok"


def print_summary(job: Dict[str, Any]) -> None:
    s = job["source"]
    print("\n=== JOB SUMMARY ===")
    print(f"Name: {job['job']['name']}")
    print(f"Source kind: {s['kind']}")
    if s["kind"] == "local":
        print(f"  Input:  {s['path']}")
    else:
        print(f"  URL:    {s['url']}")
        print(f"  DL dir: {s['download_dir']}")
        print(f"  Format: {s.get('format','best')}")
    print(f"Output dir: {job['output']['dir']}  (name_mode={job['output']['name_mode']})")
    print("Runtime:")
    print(f"  Converter:   {job['runtime']['converter_path']}")
    print(f"  Depth repo:  {job['runtime']['depth_repo']}")
    print(f"  Checkpoints: {job['runtime']['checkpoints_dir']}")
    print(f"  Logs:        {job['runtime']['log_dir']}")
    c = job["convert"]
    print("Convert:")
    print(f"  Mode: {c['mode']} | encoder={c['encoder']} | sbs_w={c['sbs_w']} | batch={c['batch']} | input_size={c['input_size']}")
    print(f"  max_shift={c['max_shift']} | alpha={c['alpha']}")
    q = job["quality"]
    print("Quality:")
    lineq = f"  mode={q['mode']} | cq={q['cq']} | nv_preset={q['nv_preset']}"
    if q.get('video_codec') is not None:
        lineq += f" | video_codec={q.get('video_codec')} | crf={q.get('crf')} | x264_preset={q.get('x264_preset')}"
    print(lineq)
    print("===================\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Create a new job YAML via an interactive wizard.")
    ap.add_argument("--jobs_dir", default="jobs", help="Directory to store jobs (default: ./jobs)")
    ap.add_argument("--name", default=None, help="Optional job name. Default: job_<timestamp>")
    args = ap.parse_args()

    if not PTK:
        print("ℹ️  prompt_toolkit not installed: using basic prompts (no path autocompletion).")
        print("   For a nicer wizard, install it: pip install prompt_toolkit\n")

    job_name = args.name or f"job_{now_stamp()}"

    # A) Source
    sk = choose(
        title="Source",
        text="Choose the input source type:",
        options=(("local", "Local file (recommended)"), ("ytdlp", "YouTube/URL via yt-dlp")),
        default_key="local",
    )

    if sk == "local":
        src_path = ask_str("Input video path", "./inputs/input.mp4", help_text="Tip: you can paste a full path.", path=True)
        source: Dict[str, Any] = asdict(SourceLocal(path=str(Path(src_path).expanduser())))
    else:
        url = ask_str("yt-dlp URL", "https://", help_text="Paste a URL (YouTube or compatible site).", path=False)
        dl = ask_str("Download directory", "./work/downloads", help_text="Where to store the downloaded file.", path=True)
        fmt = ask_str("yt-dlp format", "best", help_text="Format selector. Default 'best' is fine.", path=False)
        source = asdict(SourceYtdlp(url=url, download_dir=str(Path(dl).expanduser()), format=fmt))

    # B) Output
    out_dir = ask_str("Output directory", "./outputs", help_text="Where run_job will write the final SBS video.", path=True)
    output = OutputCfg(dir=str(Path(out_dir).expanduser()), name_mode="auto")

    # C) Runtime / converter paths
    converter_path = ask_str("Converter script path", "./mono_to_sbs_pico4_v2_autosize.py", help_text="Path to your converter script.", path=True)
    depth_repo = ask_str("Depth-Anything-V2 repo path", "./Depth-Anything-V2", help_text="Path to the cloned Depth-Anything-V2 repo.", path=True)
    checkpoints_dir = ask_str("Checkpoints dir", "./checkpoints", help_text="Folder containing depth_anything_v2_*.pth models.", path=True)
    log_dir = ask_str("Log dir", "./logs", help_text="Where run_job will write logs.", path=True)
    runtime = RuntimeCfg(
        converter_path=str(Path(converter_path).expanduser()),
        depth_repo=str(Path(depth_repo).expanduser()),
        checkpoints_dir=str(Path(checkpoints_dir).expanduser()),
        log_dir=str(Path(log_dir).expanduser()),
    )

    # D) Convert params
    mode = choose(
        title="Mode",
        text="Choose conversion mode:",
        options=(("preview", "Preview (faster, lower fidelity)"), ("full", "Full (slower, best quality)")),
        default_key="preview",
    )

    # Preview interval: how many frames to skip between preview samples (only if converter supports it).
    # Typical: 30 for 30fps => ~1 sample per second.
    preview_interval = 30
    if mode == "preview":
        preview_interval = ask_int("Preview interval (frames to skip between samples)", 30, min_v=1, max_v=10000)
    sbs_w = ask_int("SBS total width (px)", 3840, min_v=320, max_v=16384)
    encoder = choose(
        title="Encoder",
        text="DepthAnything encoder (model size):",
        options=(("vits", "vits (small/fast)"), ("vitb", "vitb (base, good default)"), ("vitl", "vitl (large)"), ("vitg", "vitg (giant, optional)")),
        default_key="vitb",
    )
    batch = ask_int("Batch size", 8, min_v=1, max_v=512)
    input_size = ask_int("Model input_size", 518, min_v=128, max_v=2048)
    max_shift = ask_int("max_shift", 24, min_v=1, max_v=256)
    alpha = ask_float("alpha (temporal smoothing 0..1)", 0.90, min_v=0.0, max_v=1.0)

    convert = ConvertCfg(
        mode=mode,
        preview_interval=preview_interval,
        sbs_w=sbs_w,
        sbs_h=None,
        encoder=encoder,
        batch=batch,
        input_size=input_size,
        max_shift=max_shift,
        alpha=alpha,
    )

    # E) Quality
    q_mode = choose(
        title="Quality",
        text="Quality tuning mode:\n- auto: run_job detects supported flags and applies cq/preset if possible\n- nvenc: force NVENC-style settings\n- none: do not pass quality flags",
        options=(("auto", "auto (recommended)"), ("nvenc", "nvenc (use cq + nv_preset)"), ("none", "none")),
        default_key="auto",
    )

    # Video codec selection (works only if your converter supports it; run_job will auto-detect)
    video_codec = choose(
        title="Video codec",
        text="Select ffmpeg video codec strategy:\n"
             "- auto: use NVENC if available, else x264\n"
             "- nvenc: force h264_nvenc\n"
             "- x264: force libx264",
        options=(("auto", "auto (recommended)"), ("nvenc", "nvenc (h264_nvenc)"), ("x264", "x264 (libx264)")),
        default_key="auto",
    )

    # x264 knobs (only meaningful if x264 is selected; safe defaults otherwise)
    crf = ask_int("crf (x264 constant quality)", 28, min_v=0, max_v=51)
    x264_preset = ask_str("x264_preset (ultrafast..veryslow)", "fast")

    cq_default = 28 if mode == "preview" else 24
    cq = ask_int("cq (NVENC constant quality)", cq_default, min_v=1, max_v=51) if q_mode in {"auto", "nvenc"} else cq_default
    nv_preset = ask_str("nv_preset (p1..p7)", "p5") if q_mode in {"auto", "nvenc"} else "p5"
    quality = QualityCfg(mode=q_mode, cq=cq, nv_preset=nv_preset, video_codec=video_codec, crf=crf, x264_preset=x264_preset)

    job = build_job_dict(job_name, source, output, runtime, convert, quality)

    # Summary + YAML preview
    print_summary(job)
    yml = to_yaml(job)
    print("=== YAML PREVIEW ===")
    print(yml.strip())
    print("====================\n")

    ok, msg = validate_paths(job)
    if not ok:
        print(f"⚠️  Validation warning: {msg}")
        print("   You can still save the job, but run_job may fail until it's fixed.\n")

    confirm = ask_str("Write this job file? (y/N)", "N")
    if not yn(confirm):
        print("Cancelled. No file written.")
        return

    jobs_dir = Path(args.jobs_dir).expanduser().resolve()
    ensure_dir(jobs_dir)
    out_path = jobs_dir / f"{job_name}.yaml"
    out_path.write_text(yml, encoding="utf-8")
    print(f"✅ Wrote job: {printable_path(out_path)}")

    if not PTK:
        print("\nTip: for path autocompletion, install prompt_toolkit:")
        print("  pip install prompt_toolkit")


if __name__ == "__main__":
    main()
