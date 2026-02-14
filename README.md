# Depth-Anything Colab Notebook — Mono → SBS (Pico 4 / VR)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hashtag1138/Depth-Anything-collab-notebook/blob/main/DepthAnything-Collab-Notebook.ipynb)

---

## 🇫🇷 FR

Ce repo fournit un **notebook Google Colab** + des **scripts Python** pour convertir une vidéo **mono** en vidéo **SBS (Side-By-Side)** destinée à la VR (ex: Pico 4), en s’appuyant sur **Depth Anything V2** pour estimer une carte de profondeur et reprojeter l’image (génération œil gauche / œil droit).

> ⚠️ Statut : expérimental / projet perso orienté “pipeline qui marche”.
> Les perfs et les réglages (shift, alpha, input_size, batch…) sont à ajuster selon tes vidéos et ton GPU.

### Ce que fait le pipeline

1. Lit la vidéo via **ffmpeg** en flux (pipe rawvideo)
2. Calcule la profondeur par frame (Depth Anything V2)
3. Reprojette l’image en stéréo (SBS) via `grid_sample`
4. Encode en H.264 (NVENC si dispo, sinon x264), puis remuxe l’audio (optionnel)

Résultat : une vidéo `*_sbs.mp4` lisible sur un casque VR (lecture via un player VR type DeoVR, etc.).

### Contenu du repo

- `DepthAnything-Collab-Notebook.ipynb` : notebook Colab (installation + exécution)
- `install_collab.py` : install “light” pour Colab (ffmpeg + clone Depth-Anything-V2 + checkpoints + dossiers)
- `install.py` : install local Linux (venv + deps + clone Depth-Anything-V2 + checkpoints)
- `mono_to_sbs_pico4_v2_autosize.py` : converter principal (mono → SBS)
- `new_job.py` : assistant interactif pour créer des jobs YAML
- `run_job.py` : runner qui exécute une file de jobs YAML (local ou ytdlp)
- `test_install.py` : smoke test avec progression et auto-détection des flags supportés
- `test_all.py` + `make_calibration_video_multi_res.py` : tests E2E (génère des vidéos de calibration puis lance des jobs)
- `calibration_pattern_3840x2160.png` : pattern de calibration

### Prérequis

**Local (Linux)**  
- Python 3.10+ (idéalement 3.11/3.12)
- `ffmpeg` + `ffprobe` + `git`
- GPU NVIDIA (optionnel mais recommandé), NVENC si tu veux encoder vite

**Colab**  
- Runtime GPU conseillé (T4/L4/A100 selon dispo)
- Le notebook s’occupe du reste

### Utilisation rapide (Google Colab)

1. Ouvre le notebook : `DepthAnything-Collab-Notebook.ipynb` (badge ci-dessus)
2. Installation / smoke-test :

```bash
!python install_collab.py --with-widgets --smoke-test
```

3. Conversion (exemple) :

```bash
!python mono_to_sbs_pico4_v2_autosize.py input.mp4 output_sbs.mp4   --encoder vitb   --sbs_w 3840 --sbs_h 2160   --max_shift 24 --alpha 0.90   --input_size 518   --batch 8   --fp16
```

> Astuce Colab : commence en `--preview` pour valider le rendu rapidement, puis relance en full.

### Utilisation (Local / Linux)

#### 1) Installer l’environnement (venv + deps + Depth-Anything-V2 + checkpoints)

```bash
git clone https://github.com/hashtag1138/Depth-Anything-collab-notebook
cd Depth-Anything-collab-notebook

python3 install.py --venv .venv --encoder vits,vitb --depth_repo ./Depth-Anything-V2
source .venv/bin/activate
```

> Torch (CUDA) : volontairement, `install.py` évite d’imposer une version CUDA (ça dépend du système).  
> Installe torch toi-même, ou utilise les options `--torch cpu` / `--torch cuXXX` selon ton setup.

#### 2) Convertir une vidéo (mode direct)

```bash
python mono_to_sbs_pico4_v2_autosize.py input.mp4 output_sbs.mp4   --encoder vitb   --sbs_w 3840 --sbs_h 2160   --max_shift 24 --alpha 0.90   --input_size 518   --batch 8   --video_codec auto
```

`--video_codec auto` choisit NVENC si disponible (`h264_nvenc`), sinon `libx264`.

---

## Workflow “Jobs” (file de conversions)

### 1) Créer un job YAML (wizard)

```bash
python new_job.py
```

### 2) Lancer tous les jobs en attente

```bash
python run_job.py
```

Le runner :

- scanne `./jobs/*.yaml`
- télécharge la source si besoin (yt-dlp)
- calcule automatiquement un nom `*_sbs.mp4` si configuré en auto
- lance le converter (logs + progression)
- déplace les jobs réussis dans `./job_done/` (les jobs en échec restent pour retry)

---

## Recettes de réglages (2K / 4K / Pico4 / Preview)

> Idée générale : **valider vite**, puis monter en qualité.  
> Les valeurs ci-dessous sont des “presets” pratiques (à adapter).

### A) Preview “ultra-rapide” (pour valider la 3D en 1–3 minutes)

Objectif : voir si le rendu “marche” (parallaxe, artefacts, confort) sans convertir toute la vidéo.

```bash
python mono_to_sbs_pico4_v2_autosize.py input.mp4 preview_sbs.mp4   --preview --preview_interval 2   --encoder vits   --input_size 384   --sbs_w 2560 --sbs_h 1440   --max_shift 16 --alpha 0.92   --batch 16   --video_codec auto
```

- `--preview_interval 2` : ~1 frame toutes les 2 secondes (à adapter)
- `vits` + `input_size 384` : beaucoup plus rapide
- `alpha` plus haut : profondeur plus stable en preview (limite le “flicker”)

### B) 2K SBS (bon compromis perf/qualité)

**SBS total 2560×1440** (par œil : 1280×1440)

```bash
python mono_to_sbs_pico4_v2_autosize.py input.mp4 output_2k_sbs.mp4   --encoder vitb   --sbs_w 2560 --sbs_h 1440   --max_shift 18 --alpha 0.90   --input_size 518   --batch 8   --video_codec auto
```

### C) 4K SBS (qualité max “grand écran”)

**SBS total 3840×2160** (par œil : 1920×2160)

```bash
python mono_to_sbs_pico4_v2_autosize.py input.mp4 output_4k_sbs.mp4   --encoder vitb   --sbs_w 3840 --sbs_h 2160   --max_shift 24 --alpha 0.90   --input_size 518   --batch 6   --fp16   --video_codec auto
```

> Si ça rame / OOM : baisse `--batch`, ou passe en `--sbs_w 2880 --sbs_h 1620` (entre-deux).

### D) “Confort VR” (moins agressif, souvent plus agréable)

```bash
python mono_to_sbs_pico4_v2_autosize.py input.mp4 output_comfort_sbs.mp4   --encoder vitb   --sbs_w 2560 --sbs_h 1440   --max_shift 12 --alpha 0.94   --input_size 518   --batch 8   --video_codec auto
```

- `max_shift` plus bas = moins de parallaxe -> moins fatigant, surtout scènes proches

### Encodage : NVENC vs x264 (qualité / taille)

- **NVENC (recommandé si dispo)** : utilise `--cq` et `--nv_preset p1..p7` (p7 = meilleure qualité/plus lent)  
- **x264** : utilise `--crf` et `--preset` (ex: `--crf 18 --preset slow`)

---

## Dépannage (les classiques)

- **`ModuleNotFoundError: depth_anything_v2...`**  
  Vérifie que `Depth-Anything-V2/` est bien cloné et que tu lances depuis le repo (ou que le runner injecte bien le PYTHONPATH).

- **Pas de NVENC / encodage lent**  
  `--video_codec auto` bascule en x264 si NVENC n’est pas dispo.  
  Sur Linux, vérifie drivers NVIDIA + ffmpeg compilé avec `h264_nvenc`.

- **Artefacts / profondeur instable**  
  Monte `--alpha` (ex: 0.92–0.96) pour lisser.  
  Baisse `--max_shift` si la 3D est trop agressive.  
  Monte `--input_size` (coûte cher) si tu veux des détails plus stables.

---

## Crédits

- Modèle de profondeur : **Depth Anything V2** (repo officiel)
- Ce repo : scripts + notebook d’orchestration pour conversion mono → SBS orientée VR (Pico 4)

---

## Fun fact (VR)

Le cerveau tolère assez mal une parallaxe “trop forte” (surtout sur des scènes proches) : baisser un peu `max_shift` donne souvent une 3D plus “pro” et moins fatigante… même si ça “fait moins wow” au premier regard.

---

## 🇬🇧 EN

This repo provides a **Google Colab notebook** + **Python scripts** to convert a **mono** video into a **SBS (Side‑By‑Side)** VR‑friendly video (e.g. Pico 4), using **Depth Anything V2** to estimate per‑frame depth and reproject the image (left eye / right eye).

> ⚠️ Status: experimental / personal “working pipeline”.
> Performance and parameters (shift, alpha, input_size, batch…) must be tuned to your videos and GPU.

### What the pipeline does

1. Streams frames via **ffmpeg** (rawvideo pipe)
2. Runs depth estimation per frame (Depth Anything V2)
3. Stereo reprojection (SBS) via `grid_sample`
4. H.264 encoding (NVENC if available, otherwise x264), then optional audio remux

Output: a `*_sbs.mp4` that you can play in VR players (e.g. DeoVR).

### Repository contents

Same files as listed in the French section (notebook, installers, converter, job workflow, tests, calibration pattern).

### Quick start (Google Colab)

1. Open `DepthAnything-Collab-Notebook.ipynb` (badge at the top)
2. Install / smoke test:

```bash
!python install_collab.py --with-widgets --smoke-test
```

3. Convert (example):

```bash
!python mono_to_sbs_pico4_v2_autosize.py input.mp4 output_sbs.mp4   --encoder vitb   --sbs_w 3840 --sbs_h 2160   --max_shift 24 --alpha 0.90   --input_size 518   --batch 8   --fp16
```

### “Jobs” workflow

- Create a job YAML: `python new_job.py`
- Run pending jobs: `python run_job.py`

### Tuning recipes (2K / 4K / Pico4 / Preview)

See the French “Recettes de réglages” section: the commands are the same and can be used as presets.

