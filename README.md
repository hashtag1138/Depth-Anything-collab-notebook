# Depth-Anything Colab Notebook — Mono → SBS (Pico 4)

This repo contains a **Google Colab notebook** plus a **Python script** to convert a **mono** video into **SBS (Side-By-Side)** by generating a depth map with **Depth-Anything-V2**, then reprojecting the image to create a left eye / right eye view.

> ⚠️ **Status: experimental.** The pipeline works, but it’s still evolving (performance, parameters, stability, compatibility).  
> Parts of the code and iterations were co-developed with **ChatGPT (GPT-5.2)**.

---

## Repo contents

- `Depth-Anything-collab.ipynb`: Colab notebook (video download, setup, execution).
- `mono_to_sbs_pico4_v2.py`: conversion script (ffmpeg streaming + DepthAnything inference + reprojection + encoding).  
  **Note:** the script’s internal header references a name like `...v4_stream_nvenc_batch...` (the file was renamed in the repo, but this is the one being used).

---

## How it works (quick overview)

1. **Video read** via `ffmpeg` (raw frames through a pipe).
2. **Depth inference** (Depth-Anything-V2) on GPU.
3. **Temporal smoothing** of depth (`alpha`).
4. **Reprojection** (warp) to create **left/right** with a maximum pixel shift (`max_shift`).
5. **Encoding** (H.264 NVENC) and optional **audio remux**.

---

## Using the Notebook (Google Colab)

### 1) Requirements
- A Colab runtime **with GPU** (ideally T4 / L4 / A10).  
- A source video:
  - either via **yt-dlp** (URL),
  - or via **Google Drive** (shared link / file).

### 2) Typical steps
1. Open `Depth-Anything-collab.ipynb` in Colab.
2. Run the **CONFIG (form)** cell and set:
   - source (yt-dlp / drive),
   - mode (**preview** or **full**),
   - SBS resolution,
   - `input_size`, `batch`, etc.
3. Run cells in order:
   - Drive mount (optional),
   - GPU check,
   - clone `Depth-Anything-V2`,
   - download the script,
   - download the checkpoint (vits/vitb/vitl),
   - run the conversion,
   - copy to Drive (optional).

### 3) Preview vs full mode
- **Preview**: converts only **1 frame every N frames** (useful to validate settings, framing, shift, artifacts).
- **Full**: converts the whole video → much longer.

---

## Running the script locally

### Dependencies
- Linux recommended
- `ffmpeg` + `ffprobe`
- Python 3.10+ (usually fine)
- CUDA + compatible PyTorch GPU setup
- `Depth-Anything-V2` repo available (the script assumes `depth_anything_v2/` is importable and checkpoints are in `./checkpoints/`)

### Example (from inside the `Depth-Anything-V2` folder)
```bash
python mono_to_sbs_pico4_v2.py input.mp4 output_sbs.mp4 \
  --encoder vits \
  --sbs_w 2560 --sbs_h 1440 \
  --max_shift 24 \
  --alpha 0.90 \
  --input_size 518 \
  --batch 8 \
  --fp16 \
  --cq 18 --nv_preset p6
```

### Preview (sampling)
```bash
python mono_to_sbs_pico4_v2.py input.mp4 preview_sbs.mp4 \
  --preview --preview_interval 3 \
  --no_audio \
  --fp16 --batch 16
```

---

## Things to double-check (important)

### 1) Notebook vs script: parameters may be out of sync

The notebook still exposes settings like `CRF` / `preset` (libx264), while the current script encodes with **NVENC** using `--cq` and `--nv_preset`.
➡️ Two options:

* **Update the notebook** to reflect the correct parameters,
* or keep the notebook as an “orchestrator” but **pass the actual args** expected by the script.

### 2) NVENC availability

On Colab or locally, check whether your ffmpeg supports NVENC:

```bash
ffmpeg -hide_banner -encoders | grep nvenc
```

### 3) Depth-Anything-V2 checkpoints

* The notebook downloads checkpoints from Hugging Face (vits/vitb/vitl).
* By default the script expects: `checkpoints/depth_anything_v2_<enc>.pth`.

### 4) VRAM / performance

* `--encoder vits` is recommended for low VRAM GPUs (e.g., GTX 1060 3GB).
* Reduce `--input_size` (e.g., 518 → 448 → 384) if you hit OOM.
* Tune `--batch` (too high = OOM, too low = underutilized GPU).

### 5) Visual quality

* `max_shift` too high → distortions, “ghosting”, stretched edges.
* `alpha` too high → depth “lags” in fast motion (but reduces flicker).
* Hard scene cuts can break smoothing (possible improvement: auto reset on scene cut).

---

## Improvement ideas (concrete)

### A) Notebook/script consistency

* Align the form with the actual arguments: `fp16`, `cq`, `nv_preset`, `batch`, `preview_interval`, etc.
* Print the exact command being executed (already partially done) + log versions (torch/cuda/ffmpeg).

### B) Robustness / UX

* Auto “scene cut” detection to reset depth smoothing.
* Save/resume (progress checkpoints) for long conversions.
* Better handling of `ffmpeg` errors (loggable stderr).

### C) Depth quality → reprojection quality

* More stable depth normalization (instead of per-frame min/max):

  * percentile clamp (p1/p99),
  * rolling stats (EMA) to avoid jumps.
* Dynamically adjust `max_shift` depending on the scene (avoid “over-3D”).

### D) Performance

* Keep more operations on the GPU (reduce CPU roundtrips).
* Double-buffer pipeline (ffmpeg read while running inference batches).
* Controlled “upscale” option + reproducible resolution presets (720/1080/1440/2160).

---

## Disclaimer & credits

* Pipeline based on **Depth-Anything-V2** (upstream repo cloned in the notebook).
* This repo provides Colab orchestration + an **experimental** conversion script.
* Iterative co-development with **ChatGPT (GPT-5.2)**: structuring, debugging, adding options (batching/streaming), and improving the notebook workflow.

---

