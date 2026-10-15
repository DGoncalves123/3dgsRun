# Video → .ksplat 3D Gaussian Splatting Pipeline

Convert a video file into a `.ksplat` file for use with [`@mkkellogg/gaussian-splats-3d`](https://github.com/mkkellogg/GaussianSplats3D).

## Pipeline Overview

```
Video (.mp4)
    │
    ▼  Phase 1: Frame curation (scripts/extract_frames.py)
    │     sample → score sharpness → reject blur → dedupe viewpoints
    │     → CLAHE contrast enhance → write sharp, COLMAP-ready frames
    │
    ▼  Phase 2: COLMAP
    Camera Poses + Sparse Point Cloud (largest connected model)
    │
    ▼  Phase 3: 3D Gaussian Splatting
    point_cloud.ply
    │
    ▼  Phase 4: create-ksplat.js
    scene.ksplat  ← web-ready output
```

## Requirements

- **Linux** with NVIDIA GPU (10GB+ VRAM)
- CUDA toolkit (11.8 or 12.x)
- Python 3.8+
- Node.js 18+
- Git
- OpenCV (`opencv-python-headless`) for frame curation — installed by `setup.sh`

## Quick Start

```bash
# 1. One-time setup (clones repos, installs deps)
chmod +x setup.sh pipeline.sh
./setup.sh

# 2. Run the pipeline
./pipeline.sh /path/to/video.mp4 my_scene
```

The final `.ksplat` file will be at `data/my_scene/output/my_scene.ksplat`.

## Why frame curation matters (read this)

The single biggest reason dark, hand-held videos produce bad 3DGS models is
**blurry frames**. COLMAP relies on SIFT keypoints; blurry/dark frames have
almost none, so the reconstruction either fails or splits into several
disconnected pieces.

The naive "extract at N fps" approach keeps *every* frame — including all the
blur. This pipeline instead:

1. **Samples densely** (`EXTRACT_FPS`) so the occasional sharp frame between
   steps isn't missed.
2. **Rejects blurry frames** with an auto-tuned sharpness floor (variance of
   Laplacian). No absolute quality gate existed before, which is why blur
   leaked into COLMAP.
3. **De-duplicates by time**, keeping the sharpest frame in each ~0.5 s window.
   (Global-color / histogram de-dup is deliberately avoided: in a dark room,
   different viewpoints have nearly identical dark histograms and get wrongly
   merged.)
4. **Applies CLAHE contrast enhancement** to the survivors. This is critical in
   a dark room: it amplifies local contrast so SIFT can find features in
   otherwise flat, dark frames. A mild unsharp mask is also applied by default.

The per-frame scores and decisions are written to
`data/<scene>/frames/selection_report.json` so you can inspect/tune.

## Configuration

Edit `config.env` to tune parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `EXTRACT_FPS` | 6 | Sampling rate. Higher = more candidates for the curator. |
| `FRAME_MIN_SHARPNESS` | 0 | Absolute sharpness floor (0 = auto-tune). |
| `FRAME_SHARPNESS_RATIO` | 0.25 | Auto floor = ratio × 90th-pct sharpness. Lower keeps more (softer) frames; higher keeps only the sharpest. |
| `FRAME_MIN_GAP_SECONDS` | 0.5 | Min time between kept frames — the walkthrough de-duplicator. |
| `FRAME_MAX_FRAMES` | 0 | Hard cap on kept frames (0 = unlimited). |
| `FRAME_CLAHE` | 1 | Contrast enhancement (keep on for dark rooms). |
| `FRAME_DENOISE` | 0 | Optional light denoise (slower). |
| `FRAME_UNSHARP` | 1 | Optional unsharp mask (crispens soft edges). |
| `COLMAP_MAX_FEATURES` | 16384 | More SIFT features for weak/low-contrast scenes. |
| `COLMAP_GUIDED_MATCHING` | 1 | Recovers more matches on weak imagery. |
| `TRAIN_ITERATIONS` | 30000 | Training iterations (7000 for preview). |
| `TRAIN_RESOLUTION` | 2 | Image scale factor (2 = half res, good for 10GB VRAM). |
| `KSPLAT_COMPRESSION` | 1 | 0=none, 1=16-bit (recommended), 2=max. |

## Resuming a Failed Run

```bash
./pipeline.sh /path/to/video.mp4 my_scene --phase 3   # resume from phase 3
```

Each phase checks for existing output and skips if already completed.
To re-curate frames, delete `data/<scene>/frames/.selected` and the `*.jpg` in
`frames/`.

## Output Structure

```
data/my_scene/
├── input.mp4                  # Copy of source video
├── frames/                    # Curated, enhanced frames (COLMAP input)
│   ├── frame_XXXXX.jpg
│   └── selection_report.json  # per-frame sharpness + keep/discard decisions
├── colmap/
│   ├── database.db
│   ├── sparse/                # one or more models (largest is used)
│   └── undistorted/           # undistorted images for training
├── training/
│   └── point_cloud/iteration_30000/point_cloud.ply
└── output/
    └── my_scene.ksplat
```

## Tips for Good Results

### Video capture (the #1 lever)
- **Add light.** A dark room forces long exposures and high ISO → blur + noise.
  A cheap work light aimed at the column makes a night-and-day difference.
- **Walk slowly, keep the camera steady.** Stop briefly at each viewpoint if
  you can — sharp frames are what make the model.
- Maintain ~60–80% overlap between consecutive viewpoints.
- Cover the scene from multiple heights/angles.
- 30–90 seconds of usable footage is usually enough.
- Avoid screen rotation surprises: this pipeline autorotates phone footage
  (reads the video's rotation metadata).

### If COLMAP still fragments or registers < 40% of frames
- Lower `FRAME_SHARPNESS_RATIO` (e.g. 0.2) or raise `EXTRACT_FPS` to keep more
  sharp frames.
- Set `FRAME_DENOISE=1` for noisy footage.
- If the source is genuinely too blurry, **re-capture** — no script can invent
  sharpness.

### 10GB VRAM
- `TRAIN_RESOLUTION=2` (half resolution) works for most scenes.
- On OOM, try `TRAIN_RESOLUTION=4` or raise `TRAIN_DENSIFY_GRAD_THRESHOLD` to `0.0004`.

## Using the Output

Load the `.ksplat` with `@mkkellogg/gaussian-splats-3d` (see the companion
`3dsplat_html` viewer repo). Note the `cameraUp: [0, -1, 0]` — COLMAP's Y axis
points down, so using `[0, 1, 0]` renders the model upside-down.

```javascript
import * as GaussianSplats3D from '@mkkellogg/gaussian-splats-3d';

const viewer = new GaussianSplats3D.Viewer({
    cameraUp: [0, -1, 0],
    initialCameraPosition: [1, 0.5, 1],
    initialCameraLookAt: [0, 0, 0]
});

viewer.addSplatScene('my_scene.ksplat')
    .then(() => viewer.start());
```

## Tools Used

| Tool | Purpose |
|------|---------|
| [FFmpeg](https://ffmpeg.org/) | Video decode (via OpenCV) |
| [OpenCV](https://opencv.org/) | Frame scoring, CLAHE, enhancement |
| [COLMAP](https://colmap.github.io/) | Structure from Motion |
| [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) | 3DGS training |
| [GaussianSplats3D](https://github.com/mkkellogg/GaussianSplats3D) | PLY → .ksplat conversion |
