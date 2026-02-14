#!/usr/bin/env python3
"""mono_to_sbs_pico4_v4_stream_nvenc_batch.py

Streaming (no-PNG) mono->SBS converter with:
- ffmpeg rawvideo pipe in/out
- NVIDIA NVENC encode (h264_nvenc)
- Depth-Anything-V2
- Torch reprojection (grid_sample)
- **Batching**: runs Depth + reprojection on batches of frames to better saturate the GPU.

Notes
- This script assumes you run it from the Depth-Anything-V2 repo root so that
  `depth_anything_v2/` is importable and checkpoints are in ./checkpoints/.
- Batching improves throughput when the GPU is under-utilized (common with 1-frame forwards).

Examples
  python mono_to_sbs_pico4_v4_stream_nvenc_batch.py input.mp4 out.mp4 --fp16 --batch 8
  python mono_to_sbs_pico4_v4_stream_nvenc_batch.py input.mp4 out.mp4 --preview --preview_interval 3 --batch 16 --no_audio
"""

from __future__ import annotations

import argparse
import math
import subprocess
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F

from depth_anything_v2.dpt import DepthAnythingV2


# -------------------------
# Defaults (Pico 4 friendly)
# -------------------------
UPSCALE_W, UPSCALE_H = 2560, 1440
SBS_W, SBS_H = 3840, 2160
DEFAULT_MAX_SHIFT = 24
DEFAULT_ALPHA = 0.90
DEFAULT_INPUT_SIZE = 518
DEFAULT_CQ = 18
DEFAULT_NV_PRESET = "p6"  # p1 fastest .. p7 best
DEFAULT_BATCH = 8

DEFAULT_VIDEO_CODEC = "auto"  # auto|nvenc|x264
DEFAULT_CRF = 28
DEFAULT_X264_PRESET = "fast"
MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64,  "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)

def ffmpeg_has_encoder(encoder: str) -> bool:
    """Return True if ffmpeg reports the given encoder in `ffmpeg -encoders`. Cached."""
    global _FFMPEG_ENCODERS_CACHE
    if _FFMPEG_ENCODERS_CACHE is None:
        try:
            out = subprocess.check_output(["ffmpeg", "-hide_banner", "-encoders"], stderr=subprocess.STDOUT).decode("utf-8", errors="ignore")
        except Exception:
            out = ""
        _FFMPEG_ENCODERS_CACHE = out
    return encoder in _FFMPEG_ENCODERS_CACHE


_FFMPEG_ENCODERS_CACHE: str | None = None


def pick_video_encoder(requested: str) -> str:
    """Resolve requested video encoder: auto -> nvenc if available else libx264."""
    requested = (requested or "auto").lower()
    if requested == "auto":
        return "h264_nvenc" if ffmpeg_has_encoder("h264_nvenc") else "libx264"
    if requested in {"nvenc", "h264_nvenc"}:
        return "h264_nvenc"
    if requested in {"x264", "libx264"}:
        return "libx264"
    # allow passing raw ffmpeg encoder names
    return requested



def ffprobe_fps(input_path: Path) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(input_path),
    ]
    out = subprocess.check_output(cmd).decode("utf-8").strip()
    if "/" in out:
        n, d = out.split("/")
        n_i, d_i = int(n), int(d)
        return (n_i / d_i) if d_i != 0 else float(n_i)
    return float(out)


def ffprobe_wh(input_path: Path) -> tuple[int, int]:
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=s=x:p=0",
        str(input_path),
    ]
    out = subprocess.check_output(cmd).decode("utf-8").strip()
    w_s, h_s = out.split("x")
    return int(w_s), int(h_s)


def ffprobe_duration(input_path: Path) -> float | None:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(input_path),
    ]
    try:
        out = subprocess.check_output(cmd).decode("utf-8").strip()
        return float(out) if out else None
    except Exception:
        return None


def estimate_total_frames(duration_s: float | None, fps_in: float, preview: bool, preview_interval: float) -> int | None:
    if duration_s is None or duration_s <= 0:
        return None
    if preview:
        return max(1, int(math.ceil(duration_s / max(preview_interval, 1e-6))))
    return max(1, int(round(duration_s * fps_in)))


def open_ffmpeg_reader(input_path: Path, fps: float, preview: bool, preview_interval: float) -> subprocess.Popen:
    vf = f"fps=1/{preview_interval}" if preview else f"fps={fps}"
    cmd = [
        "ffmpeg", "-v", "error", "-nostdin",
        "-i", str(input_path),
        "-vf", vf,
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "pipe:1",
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=10**8)


def open_ffmpeg_writer(
    out_path: Path,
    fps_out: float,
    w: int,
    h: int,
    video_encoder: str,
    *,
    cq: int,
    nv_preset: str,
    crf: int,
    x264_preset: str,
) -> subprocess.Popen:
    """Open an ffmpeg rawvideo->H.264 writer.

    - If video_encoder == 'h264_nvenc': uses NVENC with --cq + preset (p1..p7)
    - If video_encoder == 'libx264': uses x264 with -crf + preset (ultrafast..veryslow)

    The output is MP4 yuv420p with faststart.
    """
    enc = video_encoder

    base = [
        "ffmpeg", "-y", "-v", "error", "-nostdin",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s:v", f"{w}x{h}",
        "-r", str(fps_out),
        "-i", "pipe:0",
    ]

    if enc == "h264_nvenc":
        cmd = base + [
            "-c:v", "h264_nvenc",
            "-preset", nv_preset,
            "-cq", str(int(cq)),
        ]
    elif enc == "libx264":
        cmd = base + [
            "-c:v", "libx264",
            "-preset", x264_preset,
            "-crf", str(int(crf)),
        ]
    else:
        # Best-effort: pass through encoder name without extra quality flags
        cmd = base + ["-c:v", enc]

    cmd += [
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(out_path),
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, bufsize=10**8)


def mux_audio_from_source(src_video: Path, sbs_video_no_audio: Path, out_path: Path) -> None:
    run([
        "ffmpeg", "-y",
        "-i", str(sbs_video_no_audio),
        "-i", str(src_video),
        "-map", "0:v:0",
        "-map", "1:a:0?",
        "-c:v", "copy",
        "-c:a", "copy",
        str(out_path),
    ])


def load_model(encoder: str, ckpt_path: Path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DepthAnythingV2(**MODEL_CONFIGS[encoder])
    # Safer loading (avoid unpickling arbitrary objects) when the checkpoint is a plain state_dict.
    try:
        sd = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    except TypeError:
        # Older PyTorch versions don't support weights_only
        sd = torch.load(str(ckpt_path), map_location="cpu")
    model.load_state_dict(sd)
    model = model.to(device).eval()
    return model, device


def build_base_grid(eye_h: int, eye_w: int, device: str, dtype: torch.dtype) -> torch.Tensor:
    ys = torch.linspace(-1.0, 1.0, eye_h, device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, eye_w, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).contiguous()  # 1xHxWx2


def make_eye_canvas_and_depth(img_bgr: np.ndarray, depth_01: np.ndarray, eye_w: int, eye_h: int) -> tuple[np.ndarray, np.ndarray]:
    """Fit input into (eye_w, eye_h) WITHOUT cropping (contain).
    - Resizes with preserved aspect ratio
    - Pads with black (image) / 1.0 (depth) to center
    This avoids the 'zoom' effect caused by center-cropping wide inputs.
    """
    h, w = img_bgr.shape[:2]
    if h <= 0 or w <= 0:
        raise ValueError(f"Invalid frame size: {w}x{h}")

    # Scale to fit inside the target (contain)
    scale = min(eye_w / w, eye_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    interp_img = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    img_rs = cv2.resize(img_bgr, (new_w, new_h), interpolation=interp_img)
    d_rs = cv2.resize(depth_01, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.zeros((eye_h, eye_w, 3), dtype=np.uint8)
    depth_c = np.ones((eye_h, eye_w), dtype=np.float32)

    x0 = (eye_w - new_w) // 2
    y0 = (eye_h - new_h) // 2

    canvas[y0:y0 + new_h, x0:x0 + new_w, :] = img_rs
    depth_c[y0:y0 + new_h, x0:x0 + new_w] = d_rs

    return canvas, depth_c


# -------------------------
# Batch depth inference
# -------------------------

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


@torch.no_grad()
def infer_depth_batch(
    model: DepthAnythingV2,
    frames_bgr: list[np.ndarray],
    input_size: int,
    use_fp16: bool,
    device: str = "cuda",
) -> torch.Tensor:
    """Infer depth for a batch of frames.

    Returns: depth (B, H, W) float32 on CPU (so you can do min/max + smoothing cheaply).

    Why CPU return? Because your smoothing is sequential and you already do min/max in numpy.
    If you want even more speed, we can keep it on GPU and do reduction there.
    """
    assert device == "cuda"
    B = len(frames_bgr)
    H, W = frames_bgr[0].shape[:2]

    # Build batch tensor: (B,3,S,S)
    # We do resize on CPU (OpenCV) to keep code simple + predictable.
    resized_rgb = []
    for f in frames_bgr:
        if f.shape[:2] != (H, W):
            raise ValueError("All frames in batch must have same size")
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        r = cv2.resize(rgb, (input_size, input_size), interpolation=cv2.INTER_CUBIC)
        resized_rgb.append(r)

    x = np.stack(resized_rgb, axis=0)  # B,S,S,3 uint8
    x_t = torch.from_numpy(x).to(device=device, non_blocking=True)
    # Keep the input in float32; use autocast for fp16/bf16 safely.
    # Some DepthAnythingV2 checkpoints keep conv biases/weights in float32 and will error
    # if you manually pass a float16 input outside an autocast context.
    x_t = x_t.permute(0, 3, 1, 2).contiguous().to(dtype=torch.float32)
    x_t = x_t / 255.0

    mean = _IMAGENET_MEAN.to(device=device, dtype=torch.float32)
    std = _IMAGENET_STD.to(device=device, dtype=torch.float32)
    x_t = (x_t - mean) / std

    # Forward
    # DepthAnythingV2 forward typically returns (B,1,S,S) or (B,S,S); handle both.
    if use_fp16:
        with torch.cuda.amp.autocast(dtype=torch.float16):
            y = model(x_t)
    else:
        y = model(x_t)
    if y.dim() == 4 and y.shape[1] == 1:
        y = y[:, 0]

    # Upsample back to original H,W
    y = y.unsqueeze(1)  # B,1,S,S
    y = F.interpolate(y, size=(H, W), mode="bilinear", align_corners=False)
    y = y[:, 0]

    return y.float().cpu()  # B,H,W float32


# -------------------------
# Batch reprojection
# -------------------------


@torch.no_grad()
def sbs_batch_from_canvas_depth_torch(
    canvases_bgr_u8: np.ndarray,  # (B,H,W,3) uint8
    depths_01: np.ndarray,        # (B,H,W) float32
    max_shift_px: int,
    base_grid: torch.Tensor,      # (1,H,W,2)
    px_to_norm: float,
    use_fp16: bool,
) -> torch.Tensor:
    """GPU warp a batch into SBS frames.

    Returns: uint8 tensor on CPU with shape (B, H, 2W, 3).
    """
    dev = "cuda"
    B, H, W, _ = canvases_bgr_u8.shape
    assert base_grid.shape[1] == H and base_grid.shape[2] == W

    img_t = torch.from_numpy(canvases_bgr_u8).to(device=dev, non_blocking=True)
    img_t = img_t.permute(0, 3, 1, 2).contiguous()  # B,3,H,W
    img_t = img_t.to(dtype=torch.float16 if use_fp16 else torch.float32)

    depth_t = torch.from_numpy(depths_01).to(device=dev, non_blocking=True).to(dtype=torch.float32)  # B,H,W
    half_px = ((1.0 - depth_t).clamp(0.0, 1.0) * float(max_shift_px)) * 0.5
    dx = (half_px * px_to_norm).to(dtype=base_grid.dtype).unsqueeze(-1)  # B,H,W,1

    # Expand base grid to batch without allocating B copies of the full grid
    # (expand is a view). Then we still need actual tensors to write into, so clone once.
    grid = base_grid.expand(B, -1, -1, -1).clone()  # B,H,W,2
    grid_l = grid
    grid_r = grid.clone()

    grid_l[..., 0:1] = (grid_l[..., 0:1] + dx).clamp(-1.0, 1.0)
    grid_r[..., 0:1] = (grid_r[..., 0:1] - dx).clamp(-1.0, 1.0)

    left = F.grid_sample(img_t, grid_l, mode="bilinear", padding_mode="border", align_corners=True)
    right = F.grid_sample(img_t, grid_r, mode="bilinear", padding_mode="border", align_corners=True)
    sbs_t = torch.cat([left, right], dim=3)  # B,3,H,2W

    sbs_u8 = (
        sbs_t.clamp(0, 255)
        .to(dtype=torch.uint8)
        .permute(0, 2, 3, 1)
        .contiguous()
        .cpu()
    )
    return sbs_u8


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=str, help="Input mono video")
    ap.add_argument("output", type=str, help="Output SBS video (mp4)")
    ap.add_argument("--encoder", default="vits", choices=["vits", "vitb", "vitl", "vitg"],
                    help="Depth Anything encoder (vits recommended for 3GB VRAM)")
    ap.add_argument("--ckpt", default="", help="Checkpoint path (default: checkpoints/depth_anything_v2_<enc>.pth)")
    ap.add_argument("--input_size", type=int, default=DEFAULT_INPUT_SIZE)
    ap.add_argument("--max_shift", type=int, default=DEFAULT_MAX_SHIFT)
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)

    ap.add_argument("--sbs_w", type=int, default=SBS_W, help="Output SBS width (total, both eyes). Each eye is sbs_w/2.")
    ap.add_argument("--sbs_h", type=int, default=None, help="Output SBS height. Default: auto from input aspect ratio so each eye matches the input video.")

    ap.add_argument("--upscale", action="store_true", help=f"Upscale frames to {UPSCALE_W}x{UPSCALE_H} before depth")
    ap.add_argument("--up_w", type=int, default=UPSCALE_W)
    ap.add_argument("--up_h", type=int, default=UPSCALE_H)

    ap.add_argument("--fps", type=float, default=0.0, help="Force fps (0=auto)")

    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--preview_interval", type=float, default=3.0)
    ap.add_argument("--no_audio", action="store_true")

    ap.add_argument("--fp16", action="store_true", help="Use fp16 for depth+reprojection (recommended)")
    ap.add_argument("--batch", type=int, default=DEFAULT_BATCH, help="Batch size for depth+reprojection")

    ap.add_argument("--video_codec", type=str, default=DEFAULT_VIDEO_CODEC,
                    help="Video encoder: auto|nvenc|x264. auto picks nvenc if available, else libx264")
    ap.add_argument("--crf", type=int, default=DEFAULT_CRF, help="x264 CRF (used if libx264 is selected)")
    ap.add_argument("--x264_preset", type=str, default=DEFAULT_X264_PRESET, help="x264 preset (ultrafast..veryslow)")

    ap.add_argument("--cq", type=int, default=DEFAULT_CQ, help="NVENC constant quality (roughly like CRF)")
    ap.add_argument("--nv_preset", type=str, default=DEFAULT_NV_PRESET, help="NVENC preset p1..p7")

    args = ap.parse_args()

    inp = Path(args.input).resolve()
    out = Path(args.output).resolve()
    if not inp.exists():
        raise SystemExit(f"Input not found: {inp}")

    ckpt = Path(args.ckpt) if args.ckpt else Path(f"checkpoints/depth_anything_v2_{args.encoder}.pth")
    if not ckpt.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt} (put it under ./checkpoints/)")

    fps_in = args.fps if args.fps > 0 else ffprobe_fps(inp)
    fps_out = (1.0 / args.preview_interval) if args.preview else fps_in
    src_w, src_h = ffprobe_wh(inp)

    def _even(x: int) -> int:
        return x if (x % 2) == 0 else (x - 1)

    # Auto-pick sbs_h so that each eye matches the input aspect ratio (no forced "tall eye" frame).
    # If the user provides --sbs_h, we respect it.
    eye_w = args.sbs_w // 2
    if args.sbs_h is None:
        auto_eye_h = int(round(eye_w * (src_h / src_w)))
        auto_eye_h = max(2, auto_eye_h)
        eye_h = _even(auto_eye_h)
        args.sbs_h = eye_h
    else:
        eye_h = args.sbs_h

    # Enforce even dimensions (more compatible with H.264 / NVENC)
    args.sbs_w = _even(args.sbs_w)
    args.sbs_h = _even(args.sbs_h)
    eye_w = args.sbs_w // 2
    out_no_audio = out.with_suffix(".noaudio.mp4")

    print("[1] Load Depth Anything V2 model")
    model, device = load_model(args.encoder, ckpt)
    print(f"    device={device}")
    if device != "cuda":
        raise SystemExit("This script requires CUDA.")

    # Base grid cache
    base_dtype = torch.float16 if args.fp16 else torch.float32
    base_grid = build_base_grid(eye_h, eye_w, device="cuda", dtype=base_dtype)
    px_to_norm = 2.0 / float(max(eye_w - 1, 1))

    print("[2] Open ffmpeg reader (pipe)")
    reader = open_ffmpeg_reader(inp, fps=fps_in, preview=args.preview, preview_interval=args.preview_interval)
    assert reader.stdout is not None

    print("[3] Open ffmpeg NVENC writer (pipe)")
    video_encoder = pick_video_encoder(args.video_codec)
    if video_encoder == "h264_nvenc" and not ffmpeg_has_encoder("h264_nvenc"):
        print("[WARN] Requested NVENC but ffmpeg doesn't expose h264_nvenc; falling back to libx264", file=sys.stderr)
        video_encoder = "libx264"

    writer = open_ffmpeg_writer(
        out_no_audio,
        fps_out=fps_out,
        w=args.sbs_w,
        h=args.sbs_h,
        video_encoder=video_encoder,
        cq=args.cq,
        nv_preset=args.nv_preset,
        crf=args.crf,
        x264_preset=args.x264_preset,
    )
    assert writer.stdin is not None

    frame_bytes = src_w * src_h * 3
    prev_smooth: np.ndarray | None = None

    duration_s = ffprobe_duration(inp)
    total = estimate_total_frames(duration_s, fps_in=fps_in, preview=args.preview, preview_interval=args.preview_interval)
    pbar = tqdm(total=total, desc="Stream batch depth+SBS", unit="frame")

    batch = max(1, int(args.batch))
    try:
        while True:
            frames: list[np.ndarray] = []
            for _ in range(batch):
                raw = reader.stdout.read(frame_bytes)
                if raw is None or len(raw) < frame_bytes:
                    break
                f = np.frombuffer(raw, np.uint8).reshape((src_h, src_w, 3))
                if args.upscale and not args.preview:
                    f = cv2.resize(f, (args.up_w, args.up_h), interpolation=cv2.INTER_LANCZOS4)
                frames.append(f)

            if not frames:
                break

            # --- Depth in batch on GPU ---
            depths = infer_depth_batch(model, frames, input_size=args.input_size, use_fp16=args.fp16, device="cuda")  # B,H,W CPU float32

            # --- Normalize + temporal smoothing (still sequential) ---
            smooth_list: list[np.ndarray] = []
            for i in range(len(frames)):
                d = depths[i].numpy()
                dmin, dmax = float(d.min()), float(d.max())
                dn = (d - dmin) / (dmax - dmin + 1e-6)
                if prev_smooth is None:
                    smooth = dn.astype(np.float32)
                else:
                    smooth = (args.alpha * prev_smooth + (1.0 - args.alpha) * dn).astype(np.float32)
                prev_smooth = smooth
                smooth_list.append(smooth)

            # --- Build canvases + depths for reprojection ---
            canvases = []
            depths_c = []
            for f, sm in zip(frames, smooth_list):
                canvas, depth_c = make_eye_canvas_and_depth(f, sm, eye_w=eye_w, eye_h=eye_h)
                canvases.append(canvas)
                depths_c.append(depth_c)

            canvases_np = np.stack(canvases, axis=0)          # B,H,W,3
            depths_c_np = np.stack(depths_c, axis=0).astype(np.float32)  # B,H,W

            # --- Batch reprojection on GPU ---
            sbs_batch_u8 = sbs_batch_from_canvas_depth_torch(
                canvases_bgr_u8=canvases_np,
                depths_01=depths_c_np,
                max_shift_px=args.max_shift,
                base_grid=base_grid,
                px_to_norm=px_to_norm,
                use_fp16=args.fp16,
            )  # (B,H,2W,3) uint8 CPU

            # Write to ffmpeg
            for i in range(sbs_batch_u8.shape[0]):
                writer.stdin.write(sbs_batch_u8[i].numpy().tobytes())
                pbar.update(1)

    finally:
        pbar.close()
        try:
            if writer.stdin:
                writer.stdin.close()
        except Exception:
            pass
        try:
            if reader.stdout:
                reader.stdout.close()
        except Exception:
            pass

        reader.wait()
        writer.wait()

    if args.no_audio or args.preview:
        out_no_audio.replace(out)
        print(f"Done: {out}")
        return

    print("[4] Mux audio from source")
    mux_audio_from_source(inp, out_no_audio, out)
    out_no_audio.unlink(missing_ok=True)
    print(f"Done: {out}")


if __name__ == "__main__":
    main()
