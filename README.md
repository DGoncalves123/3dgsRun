# Video → .ksplat 3D Gaussian Splatting Pipeline

Convert a video file into a `.ksplat` file for use with [`@mkkellogg/gaussian-splats-3d`](https://github.com/mkkellogg/GaussianSplats3D).

## Pipeline Overview

```
Video (.mp4)
    │
    ▼  Phase 1: FFmpeg
Frame Sequence (JPEGs)
    │
    ▼  Phase 2: COLMAP
Camera Poses + Sparse Point Cloud
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

## Quick Start

```bash
# 1. One-time setup (clones repos, installs deps)
chmod +x setup.sh pipeline.sh
./setup.sh

# 2. Run the pipeline
./pipeline.sh /path/to/video.mp4 my_scene
```

The final `.ksplat` file will be at `data/my_scene/output/my_scene.ksplat`.

## Configuration

Edit `config.env` to tune parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `EXTRACT_FPS` | 2 | Frames per second to extract |
| `COLMAP_MATCHER` | exhaustive | `exhaustive` (best) or `sequential` (faster) |
| `TRAIN_ITERATIONS` | 30000 | Training iterations (7000 for preview) |
| `TRAIN_RESOLUTION` | 2 | Image scale factor (2 = half res, good for 10GB VRAM) |
| `KSPLAT_COMPRESSION` | 1 | 0=none, 1=16-bit (recommended), 2=max |

## Resuming a Failed Run

If the pipeline fails at any phase, resume from that phase:

```bash
# Resume from phase 3 (training)
./pipeline.sh /path/to/video.mp4 my_scene --phase 3
```

Each phase checks for existing output and skips if already completed.

## Output Structure

```
data/my_scene/
├── input.mp4              # Copy of source video
├── frames/                # Extracted frames
├── colmap/
│   ├── database.db        # COLMAP feature database
│   ├── sparse/0/          # Camera poses + sparse points
│   └── undistorted/       # Undistorted images for training
├── training/
│   └── point_cloud/
│       └── iteration_30000/
│           └── point_cloud.ply
└── output/
    └── my_scene.ksplat    # Final output
```

## Tips for Good Results

### Video Capture
- Walk slowly around the scene
- Maintain ~60-80% overlap between consecutive viewpoints
- Avoid motion blur (good lighting helps)
- Cover the scene from multiple heights/angles
- 30-90 seconds of video is usually enough

### 10GB VRAM
- `TRAIN_RESOLUTION=2` (half resolution) is the default — works well for most scenes
- If you get OOM errors, try `TRAIN_RESOLUTION=4` or increase `TRAIN_DENSIFY_GRAD_THRESHOLD` to `0.0004`

### Too Many Frames
- If COLMAP is very slow, reduce `EXTRACT_FPS` to 1
- Or switch `COLMAP_MATCHER` to `sequential` (still good quality for video input)

## Using the Output

Load the `.ksplat` file with `@mkkellogg/gaussian-splats-3d`:

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

| Tool | Version | Purpose |
|------|---------|---------|
| [FFmpeg](https://ffmpeg.org/) | 5+ | Frame extraction |
| [COLMAP](https://colmap.github.io/) | 3.8+ | Structure from Motion |
| [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) | Latest | 3DGS training |
| [GaussianSplats3D](https://github.com/mkkellogg/GaussianSplats3D) | Latest | PLY → .ksplat conversion |
