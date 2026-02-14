#!/usr/bin/env python3
"""
make_calibration_multi.py

Generate MANY lightweight test videos from a single calibration image:
- multiple aspect ratios (16:9, 9:16, 4:3, 1:1, 21:9, 2.39:1, etc.)
- multiple "resolution tiers" from ~360p up to 4K bounds (3840x2160)

Key idea:
For each tier, we define a bounding box (e.g. 1280x720 for 720p).
For each ratio, we compute the largest WxH that fits *inside that box*
while preserving the ratio (no stretch), then pad to exactly WxH.

Example:
  python make_calibration_multi.py --source calibration_pattern_3840x2160.png --duree 30

Requires:
  - ffmpeg available in PATH
"""

from __future__ import annotations
import argparse
import shutil
import subprocess
from pathlib import Path
from typing import List, Tuple


DEFAULT_RATIOS: List[Tuple[str, float]] = [
    ("16x9", 16/9),
    ("9x16", 9/16),
    ("4x3", 4/3),
    ("3x4", 3/4),
    ("1x1", 1.0),
    ("21x9", 21/9),
    ("239x100", 2.39),
    ("2x1", 2.0),
    ("1x2", 0.5),
]

# "From 360p to 4K" as *bounds* you fit into (classic tiers).
# (tag, max_w, max_h)
DEFAULT_TIERS: List[Tuple[str, int, int]] = [
    ("360p",  640,  360),
    ("480p",  854,  480),
    ("720p",  1280, 720),
    ("1080p", 1920, 1080),
    ("1440p", 2560, 1440),
    ("2160p", 3840, 2160),  # 4K UHD bounds
]

def even(n: int) -> int:
    return n if n % 2 == 0 else n - 1

def pick_dims_inside_box(ar: float, max_w: int, max_h: int) -> Tuple[int, int]:
    """
    Compute the largest (w,h) that fits within max_w/max_h while keeping aspect ratio ar.
    """
    # width-limited
    w = max_w
    h = int(round(w / ar))
    if h > max_h:
        h = max_h
        w = int(round(h * ar))
    w = max(2, even(w))
    h = max(2, even(h))
    return w, h

def require_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise SystemExit("ffmpeg not found in PATH. Please install ffmpeg and retry.")
    return exe

def run(cmd: List[str], dry_run: bool = False) -> None:
    if dry_run:
        print("[dry-run] " + " ".join(cmd))
        return
    subprocess.run(cmd, check=True)

def make_one_video(
    ffmpeg: str,
    src: Path,
    out: Path,
    seconds: int,
    w: int,
    h: int,
    fps: int,
    crf: int,
    preset: str,
    tune: str | None,
    overwrite: bool,
    dry_run: bool,
) -> None:
    """
    Loop the image for `seconds`, scale to fit inside w:h (preserve AR), then pad to w:h.
    Encodes a fast/low-quality H.264 MP4 for quick testing.
    """
    vf = (
        f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
        f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2"
    )

    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error"]
    cmd.append("-y" if overwrite else "-n")

    cmd += [
        "-loop", "1",
        "-t", str(seconds),
        "-i", str(src),
        "-vf", vf,
        "-r", str(fps),
        "-pix_fmt", "yuv420p",
        "-c:v", "libx264",
        "-preset", preset,
        "-crf", str(crf),
    ]
    if tune:
        cmd += ["-tune", tune]

    cmd += ["-movflags", "+faststart", str(out)]
    out.parent.mkdir(parents=True, exist_ok=True)
    run(cmd, dry_run=dry_run)

def parse_ratios(ratio_args: List[str]) -> List[Tuple[str, float]]:
    """
    Parse ratios like:
      --ratios 16:9 4:3 1:1 2.39:1 9:16
    """
    res: List[Tuple[str, float]] = []
    for r in ratio_args:
        r = r.strip().lower()
        if ":" in r:
            a, b = r.split(":", 1)
            ar = float(a) / float(b)
            tag = f"{a}x{b}".replace(".", "_")
        else:
            ar = float(r)
            tag = r.replace(".", "_")
        res.append((tag, ar))
    return res

def parse_tiers(tier_args: List[str]) -> List[Tuple[str, int, int]]:
    """
    Parse tiers like:
      --tiers 640x360 1280x720 1920x1080 3840x2160
    Each entry becomes (tag, max_w, max_h) where tag is "640x360".
    """
    res: List[Tuple[str, int, int]] = []
    for t in tier_args:
        t = t.strip().lower()
        if "x" not in t:
            raise SystemExit(f"Invalid tier '{t}'. Use like 1280x720.")
        w_s, h_s = t.split("x", 1)
        mw, mh = int(float(w_s)), int(float(h_s))
        if mw < 2 or mh < 2:
            raise SystemExit(f"Invalid tier '{t}': width/height must be >=2.")
        res.append((f"{mw}x{mh}", mw, mh))
    return res

def main() -> None:
    ap = argparse.ArgumentParser(description="Generate calibration videos across ratios AND resolution tiers.")
    ap.add_argument("--source", required=True, help="Path to the calibration image (png/jpg).")
    ap.add_argument("--duree", type=int, required=True, help="Duration in seconds (e.g., 60).")
    ap.add_argument("--outdir", default="calib_videos", help="Output directory. Default: calib_videos")

    ap.add_argument("--fps", type=int, default=30, help="FPS for test videos. Default: 30")
    ap.add_argument("--crf", type=int, default=34, help="CRF (higher = smaller/worse). Default: 34 (fast tests)")
    ap.add_argument("--preset", default="ultrafast", help="x264 preset. Default: ultrafast")
    ap.add_argument("--tune", default="stillimage", help="Optional x264 tune. Default: stillimage")

    ap.add_argument("--ratios", nargs="*", default=None,
                    help="Optional ratios to test (e.g., 16:9 4:3 2.39:1 9:16).")
    ap.add_argument("--tiers", nargs="*", default=None,
                    help="Optional explicit tiers as WxH bounds (e.g., 640x360 1280x720 3840x2160).")

    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing files.")
    ap.add_argument("--dry_run", action="store_true", help="Print ffmpeg commands without running them.")

    args = ap.parse_args()

    src = Path(args.source).expanduser().resolve()
    if not src.exists():
        raise SystemExit(f"Source image not found: {src}")

    ffmpeg = require_ffmpeg()

    ratios = parse_ratios(args.ratios) if args.ratios else DEFAULT_RATIOS
    tiers = parse_tiers(args.tiers) if args.tiers else DEFAULT_TIERS

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Source:   {src}")
    print(f"Outdir:   {outdir}")
    print(f"Duration: {args.duree}s | FPS: {args.fps} | CRF {args.crf} | preset {args.preset} | tune {args.tune}")
    print(f"Ratios:   {', '.join([t for t,_ in ratios])}")
    print(f"Tiers:    {', '.join([t for t,_,_ in tiers])}")
    print()

    for tier_tag, max_w, max_h in tiers:
        tier_dir = outdir / tier_tag
        tier_dir.mkdir(parents=True, exist_ok=True)

        print(f"== Tier {tier_tag} (bounds {max_w}x{max_h}) ==")
        for ratio_tag, ar in ratios:
            w, h = pick_dims_inside_box(ar, max_w, max_h)
            out = tier_dir / f"calib_{tier_tag}_{ratio_tag}_{w}x{h}_{args.fps}fps_{args.duree}s_crf{args.crf}.mp4"
            print(f"- {ratio_tag:8s} -> {w}x{h}  =>  {out.name}")
            make_one_video(
                ffmpeg=ffmpeg,
                src=src,
                out=out,
                seconds=args.duree,
                w=w,
                h=h,
                fps=args.fps,
                crf=args.crf,
                preset=args.preset,
                tune=args.tune,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
            )
        print()

    print("Done.")

if __name__ == "__main__":
    main()
