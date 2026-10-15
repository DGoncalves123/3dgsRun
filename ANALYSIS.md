# Diagnosis: why your column scene didn't turn into a good 3DGS model

This documents what was measured on `statue.mp4` (the dark-room column video) and
what changed. It's written so you can see *why* the original approach failed,
not just *what* to run.

## 1. The video

| Property | Value |
|----------|-------|
| File | `statue.mp4` |
| Resolution | 3840×2160 (4K), rotated −90° (portrait) |
| Frames | 5198 @ ~27.8 fps |
| Duration | ~187 s (3:07) |

The video has a −90° rotation tag. The pipeline reads it through OpenCV, which
auto-applies the rotation, so extracted frames are upright.

## 2. What the original pipeline produced (and why it failed)

The original Phase 1 did two things:

1. `ffmpeg -vf fps=6` → 1122 candidate frames.
2. A "smart selector" that grouped frames by **HSV-histogram cosine similarity**
   and kept the "sharpest" member of each group.

Result: **455 frames kept, 667 discarded**. COLMAP then split those 455 frames
into **5 disconnected models**:

| Model | Registered images | Points |
|-------|-------------------|--------|
| sparse/0 | **2** | 610 |
| sparse/1 | 13 | 52 |
| sparse/2 | 93 | 8,477 |
| sparse/3 | 100 | 8,570 |
| sparse/4 | 32 | 3,417 |

Two separate problems here:

- **Only ~240 of 455 frames registered at all**, and they fell apart into 5
  pieces instead of one trajectory. Blurry frames break SIFT matching.
- **The pipeline used `sparse/0` for undistortion/training — a 2-image
  fragment.** It never looked for the largest model. Training therefore had
  almost nothing to work with (and the `training/` folder ended up empty).

### The root cause: no absolute sharpness gate

The original selector only compared frames *within* a histogram cluster. A
blurry frame that had no sharper "twin" was kept. Measured Laplacian variance
(on a fixed 480px scale) of the 455 "kept" frames:

| Statistic | Value |
|-----------|-------|
| p10 / p25 / p50 | 49.5 / 89.8 / 152.0 |
| p75 / p90 / max | 310.9 / 649.2 / 1822.9 |
| **% of kept frames below 150** | **49%** |
| % below 100 | 29% |
| % below 50 | 10% |

Half the frames it kept were still soft/blurry. (Amusingly, the *discarded*
frames were sharper on average than the kept ones — median 282 vs 152 — which
confirms the selector was not actually selecting for sharpness.)

### Secondary root cause: histogram de-dup is wrong for dark scenes

In a dark room, different viewpoints all have similar, darkness-dominated HSV
histograms, so histogram similarity cannot tell viewpoints apart. The new
pipeline de-duplicates by **time** instead (keep the sharpest frame per window).

## 3. The fix, and why it works

The new `scripts/extract_frames.py`:

1. **Samples densely** (6 fps → ~1040 candidates) so the occasional sharp frame
   between steps isn't missed.
2. **Rejects blurry frames** with an auto-tuned absolute floor
   (`FRAME_SHARPNESS_RATIO × 90th-percentile sharpness`).
3. **De-duplicates by time** (`FRAME_MIN_GAP_SECONDS`), keeping the sharpest
   frame per window.
4. **Applies CLAHE contrast enhancement** (and a mild unsharp mask) to the
   survivors before COLMAP.

The CLAHE step is the big one for a dark room. Measured on your existing frames:

> **SIFT keypoints per frame: 797 (raw) → 1553 (CLAHE), +95%.**

Even the blurrier frames roughly double their keypoints (e.g. 241→548, 270→441,
430→705). More SIFT keypoints = more matches = COLMAP can actually register the
frames into one coherent model.

Combined end-to-end comparison (what COLMAP actually receives):

| Set | Frames | Mean SIFT keypoints/frame | Median | Min |
|-----|--------|---------------------------|--------|-----|
| Old "kept" set (raw) | 455 (≈49% blurry) | 922 | 459 | 79 |
| New curated set (enhanced) | **124** (all sharp) | **4715** | 4008 | 1391 |

**+411% keypoints per frame**, and no blurry frames left to poison matching.

### Sharpness distribution of this video

| Percentile | p25 | p50 | p75 | p90 | p99 | max |
|-----------|-----|-----|-----|-----|-----|-----|
| Laplacian var (480px) | 118 | 215 | 500 | 1117 | 2092 | 3830 |

The sharp frames are **bursty**: the video only contains ~120–140 genuinely
sharp, well-spaced viewpoints. That's enough for a single column, but it's the
hard ceiling of what this footage can give you.

## 4. Recommended settings for this video

In `config.env`:

```ini
EXTRACT_FPS=6
FRAME_SHARPNESS_RATIO=0.25     # floor ≈ 279 on this video
FRAME_MIN_GAP_SECONDS=0.5
FRAME_CLAHE=1                  # keep on — the single biggest win here
FRAME_UNSHARP=1
FRAME_DENOISE=0
```

This yields **124 curated frames** for COLMAP — sharp, evenly spaced, and
contrast-enhanced (down from 455 that were about half blurry).

### If the result is still poor

- Lower `FRAME_SHARPNESS_RATIO` to `0.15` and/or `FRAME_MIN_GAP_SECONDS` to
  `0.3` to keep more (softer) frames.
- Set `FRAME_DENOISE=1` if you see high-ISO noise.
- **Re-capture** if possible: add light aimed at the column, walk slower, and
  pause briefly at each viewpoint. No script can invent sharpness — light is
  what turns a dark blur into a usable reconstruction.

## 5. Two other things fixed along the way

- **COLMAP largest-model selection.** The pipeline now picks the largest sparse
  model instead of blindly using `sparse/0`, and warns when the reconstruction
  fragments or registers < 40% of frames.
- **numpy 2.x breaks 3DGS training.** The `tools/gaussian-splatting` venv had
  numpy 2.4.3, which crashes torch/CUDA kernels compiled against numpy 1.x.
  `setup.sh` now pins `numpy<2`.

> Note: the machine this analysis ran on has no working NVIDIA driver
> (`nvidia-smi` fails; COLMAP is built without CUDA), so 3DGS training must be
> run on your GPU machine. The frame-curation and COLMAP improvements above are
> CPU-friendly and apply everywhere.
