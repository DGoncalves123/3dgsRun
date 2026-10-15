#!/usr/bin/env python3
"""
Video -> curated, COLMAP-ready frames.

Purpose
-------
The stock "extract at N fps" approach is what kills 3D Gaussian Splatting on
dark, blurry, hand-held walkthrough video: it keeps every blurry frame, and
COLMAP (SIFT) finds almost no reliable keypoints in those frames, so the
reconstruction is sparse / wrong and the final .ksplat is garbage.

This script instead:

  1. Samples the video densely (configurable fps).
  2. Scores every candidate for sharpness (variance of Laplacian) and
     brightness on a fast downscaled copy.
  3. Rejects frames below an auto-tuned sharpness floor (or an explicit one).
  4. Enforces a minimum TIME gap between kept frames (a walkthrough's natural
     de-duplicator) while keeping the sharpest frame in each window. HSV /
     global-color de-dup is deliberately avoided: in a dark room different
     viewpoints have nearly identical (dark) histograms and get wrongly merged.
  5. Applies CLAHE contrast enhancement (huge win for SIFT in dark scenes)
     plus optional mild denoise / unsharp, then writes the curated set.

Outputs
-------
  <out>/frame_%05d.jpg        curated, enhanced frames (feed these to COLMAP)
  <out>/selection_report.json per-frame scores + decision (for tuning)
  (discarded frames are only reported, not written, to save disk)

Requires: opencv-python (cv2) and numpy.
"""

import argparse
import json
import os
import sys


def log(msg):
    print(f"[frames] {msg}", flush=True)


def load_gray(img):
    """Downscale and convert to grayscale for cheap, scale-consistent scoring."""
    h, w = img.shape[:2]
    scale = MAX_DIM / float(max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def sharpness(gray):
    """Variance of Laplacian. High = sharp; low = blurry. Scale-independent-ish
    because it is computed on a downscaled copy of fixed max dimension."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def brightness(gray):
    return float(gray.mean())


def enhance(img, args):
    """CLAHE on the L channel of LAB, then optional denoise and unsharp mask."""
    if args.denoise:
        img = cv2.fastNlMeansDenoisingColored(img, None, 3.0, 3.0, 7, 21)

    if args.clahe:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=args.clahe_clip,
                                tileGridSize=(args.clahe_grid, args.clahe_grid))
        l = clahe.apply(l)
        img = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    if args.unsharp:
        blur = cv2.GaussianBlur(img, (0, 0), args.unsharp_sigma)
        img = cv2.addWeighted(img, 1.0 + args.unsharp_amount, blur,
                              -args.unsharp_amount, 0)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("out_dir")
    ap.add_argument("--fps", type=float, default=6.0)
    ap.add_argument("--min-sharpness", type=float, default=0.0,
                    help="Absolute sharpness floor. 0 = auto-tune.")
    ap.add_argument("--sharpness-ratio", type=float, default=0.25,
                    help="Auto floor = ratio * (90th-percentile sharpness).")
    ap.add_argument("--min-brightness", type=float, default=0.0,
                    help="Reject frames darker than this (0-255). 0 = off.")
    ap.add_argument("--min-gap-seconds", type=float, default=0.5,
                    help="Minimum time between kept frames (walkthrough dedup).")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="Hard cap on kept frames (0 = unlimited).")
    ap.add_argument("--clahe", type=int, default=1)
    ap.add_argument("--clahe-clip", type=float, default=2.0)
    ap.add_argument("--clahe-grid", type=int, default=8)
    ap.add_argument("--denoise", type=int, default=0)
    ap.add_argument("--unsharp", type=int, default=1)
    ap.add_argument("--unsharp-amount", type=float, default=0.5)
    ap.add_argument("--unsharp-sigma", type=float, default=1.5)
    ap.add_argument("--jpeg-quality", type=int, default=95)
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    global MAX_DIM
    MAX_DIM = 480  # scoring resolution

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: could not open video: {args.video}")
        sys.exit(1)

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    interval = max(1, int(round(native_fps / max(args.fps, 1e-3))))
    log(f"Video: {os.path.basename(args.video)} ({native_fps:.2f} fps native, "
        f"{total_frames} frames) -> sampling every {interval} frame(s) (~{args.fps} fps)")

    # ── Pass 1: score every sampled frame ────────────────────────────────
    candidates = []  # dicts: idx, sharp, bright
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % interval == 0:
            gray = load_gray(frame)
            candidates.append({
                "idx": idx,
                "sharp": sharpness(gray),
                "bright": brightness(gray),
            })
        idx += 1
    cap.release()

    if not candidates:
        print("ERROR: no frames sampled from video.")
        sys.exit(1)

    sampled = len(candidates)
    sharps = [c["sharp"] for c in candidates]

    # Auto sharpness floor: ratio of the 90th-percentile sharpness (robust to
    # a few freak sharp outliers that a max/99th-pct would chase).
    if args.min_sharpness > 0:
        floor = args.min_sharpness
        mode = "manual"
    else:
        ref = float(np.percentile(sharps, 90.0))
        floor = args.sharpness_ratio * ref
        mode = "auto"
        if ref < 100.0:
            log("WARNING: even the sharper frames score very low "
                f"(90th-pct Laplacian var = {ref:.1f}). The source may simply be "
                "too blurry for a good 3DGS model; consider re-capturing with "
                "more light / a tripod / slower movement.")

    p25, p50, p75 = np.percentile(sharps, [25, 50, 75])
    log(f"Sharpness  p25={p25:.1f}  p50={p50:.1f}  p75={p75:.1f}  "
        f"p90={np.percentile(sharps, 90):.1f}  |  floor={floor:.1f} ({mode})")

    # ── Sharpness / brightness gate ──────────────────────────────────────
    for c in candidates:
        if c["sharp"] < floor:
            c["keep"] = False
            c["reason"] = "blur"
        elif args.min_brightness > 0 and c["bright"] < args.min_brightness:
            c["keep"] = False
            c["reason"] = "dark"
        else:
            c["keep"] = True
            c["reason"] = ""

    survivors = [c for c in candidates if c["keep"]]
    n_blur = sum(1 for c in candidates if c["reason"] == "blur")
    n_dark = sum(1 for c in candidates if c["reason"] == "dark")

    # ── Temporal de-duplication ──────────────────────────────────────────
    # Keep the sharpest frame in each sliding time window (a walkthrough's
    # natural dedup). Robust in dark scenes where global-color de-dup fails.
    min_gap_frames = max(1, int(round(args.min_gap_seconds * native_fps)))
    selected = []  # list of [idx, sharp]
    for c in survivors:
        if not selected or (c["idx"] - selected[-1][0]) >= min_gap_frames:
            selected.append([c["idx"], c["sharp"]])
        elif c["sharp"] > selected[-1][1]:
            selected[-1] = [c["idx"], c["sharp"]]

    keep_idx = {i for i, _ in selected}
    n_spacing = sum(1 for c in survivors if c["idx"] not in keep_idx)
    for c in survivors:
        if c["idx"] not in keep_idx:
            c["keep"] = False
            c["reason"] = "spacing"

    # Optional hard cap: keep the sharpest N.
    n_cap = 0
    if args.max_frames > 0 and len(keep_idx) > args.max_frames:
        top = sorted(keep_idx, key=lambda i: -next(c["sharp"] for c in candidates if c["idx"] == i))[:args.max_frames]
        for i in keep_idx:
            if i not in top:
                c = next(c for c in candidates if c["idx"] == i)
                c["keep"] = False
                c["reason"] = "cap"
                n_cap += 1
        keep_idx = set(top)

    kept = sorted(keep_idx)

    # ── Pass 2: write enhanced frames for the kept indices ───────────────
    cap = cv2.VideoCapture(args.video)
    written = 0
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in keep_idx:
            frame = enhance(frame, args)
            out_path = os.path.join(args.out_dir, f"frame_{idx:05d}.jpg")
            cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
            written += 1
        idx += 1
    cap.release()

    # ── Report ───────────────────────────────────────────────────────────
    report_frames = [
        {"idx": c["idx"], "sharp": round(c["sharp"], 1),
         "bright": round(c["bright"], 1),
         "keep": c["keep"], "reason": c["reason"]}
        for c in candidates
    ]
    report = {
        "video": os.path.basename(args.video),
        "native_fps": native_fps,
        "sample_interval": interval,
        "sampled": sampled,
        "sharpness_floor": round(floor, 1),
        "kept": len(kept),
        "discarded": sampled - len(kept),
        "discard_blur": n_blur,
        "discard_dark": n_dark,
        "discard_spacing": n_spacing,
        "discard_cap": n_cap,
        "frames": report_frames,
    }
    if args.report:
        with open(args.report, "w") as f:
            json.dump(report, f, indent=2)

    log(f"Kept {written} frames (of {sampled} sampled).")
    log(f"  rejected: {n_blur} blurry, {n_dark} too dark, "
        f"{n_spacing} too close in time, {n_cap} over cap")
    if args.report:
        log(f"Report: {args.report}")
    if written == 0:
        print("ERROR: zero frames survived curation.")
        print("  - Lower FRAME_MIN_SHARPNESS / FRAME_SHARPNESS_RATIO, or")
        print("  - raise EXTRACT_FPS, or")
        print("  - the video is genuinely too blurry -> re-capture with better light.")
        sys.exit(1)


if __name__ == "__main__":
    try:
        import cv2
        import numpy as np
    except ImportError:
        print("ERROR: opencv-python (cv2) and numpy are required.")
        print("  Install:  pip install opencv-python-headless numpy")
        sys.exit(1)
    main()
