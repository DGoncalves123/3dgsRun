#!/usr/bin/env bash
# ============================================================
# Video → .ksplat Pipeline
#
# Usage: ./pipeline.sh <video_file> <project_name> [--phase N]
#
# Phases:
#   1  Frame extraction (FFmpeg)
#   2  Camera pose estimation (COLMAP)
#   3  3DGS training
#   4  PLY → .ksplat conversion
#
# Example:
#   ./pipeline.sh /path/to/video.mp4 my_scene
#   ./pipeline.sh /path/to/video.mp4 my_scene --phase 3   # resume from phase 3
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.env"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

log()   { echo -e "${GREEN}[PIPELINE]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }
phase() { echo -e "\n${CYAN}══════════════════════════════════════════${NC}"; echo -e "${CYAN}  Phase $1: $2${NC}"; echo -e "${CYAN}══════════════════════════════════════════${NC}\n"; }

timer_start() { PHASE_START=$(date +%s); }
timer_end()   { local elapsed=$(( $(date +%s) - PHASE_START )); log "Phase completed in $((elapsed/60))m $((elapsed%60))s"; }

# ── Argument parsing ──────────────────────────────────────

VIDEO_FILE=""
PROJECT_NAME=""
START_PHASE=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --phase) START_PHASE="$2"; shift 2 ;;
        *)
            if [[ -z "$VIDEO_FILE" ]]; then
                VIDEO_FILE="$1"
            elif [[ -z "$PROJECT_NAME" ]]; then
                PROJECT_NAME="$1"
            fi
            shift
            ;;
    esac
done

if [[ -z "$VIDEO_FILE" || -z "$PROJECT_NAME" ]]; then
    echo "Usage: ./pipeline.sh <video_file> <project_name> [--phase N]"
    exit 1
fi

# Resolve to absolute path
VIDEO_FILE="$(realpath "$VIDEO_FILE")"

if [[ ! -f "$VIDEO_FILE" ]]; then
    error "Video file not found: $VIDEO_FILE"
fi

# ── Project directory setup ───────────────────────────────

PROJECT_DIR="$DATA_DIR/$PROJECT_NAME"
FRAMES_DIR="$PROJECT_DIR/frames"
COLMAP_DIR="$PROJECT_DIR/colmap"
COLMAP_DB="$COLMAP_DIR/database.db"
COLMAP_SPARSE="$COLMAP_DIR/sparse"
COLMAP_UNDISTORTED="$COLMAP_DIR/undistorted"
TRAIN_DIR="$PROJECT_DIR/training"
OUTPUT_DIR="$PROJECT_DIR/output"

mkdir -p "$FRAMES_DIR" "$COLMAP_DIR" "$COLMAP_SPARSE" "$TRAIN_DIR" "$OUTPUT_DIR"

# Copy video into project directory for reference
if [[ ! -f "$PROJECT_DIR/input$(basename "$VIDEO_FILE" | sed 's/.*\./\./')" ]]; then
    cp "$VIDEO_FILE" "$PROJECT_DIR/input$(basename "$VIDEO_FILE" | sed 's/.*\./\./')"
fi

log "Project directory: $PROJECT_DIR"
log "Starting from phase: $START_PHASE"

PIPELINE_START=$(date +%s)

# ══════════════════════════════════════════════════════════
# PHASE 1: Frame Extraction
# ══════════════════════════════════════════════════════════

if [[ "$START_PHASE" -le 1 ]]; then
    phase 1 "Frame Curation (extract + deblur-gate + CLAHE + dedupe)"
    timer_start

    SELECTED_SENTINEL="$FRAMES_DIR/.selected"
    if [[ -f "$SELECTED_SENTINEL" ]]; then
        FRAME_COUNT=$(find "$FRAMES_DIR" -name "*.jpg" 2>/dev/null | wc -l)
        log "Frame curation already done ($FRAME_COUNT frames kept)."
        warn "Delete $FRAMES_DIR/.selected and $FRAMES_DIR/*.jpg to re-curate."
    else
        PY_BIN="$GAUSSIAN_SPLATTING_DIR/venv/bin/python"
        if [[ ! -x "$PY_BIN" ]]; then
            error "Python venv not found at $PY_BIN. Run ./setup.sh first."
        fi
        if ! "$PY_BIN" -c "import cv2, numpy" 2>/dev/null; then
            error "opencv-python / numpy missing from the venv. Run ./setup.sh first."
        fi

        log "Curating frames (fps=${EXTRACT_FPS}, auto sharpness floor, CLAHE=${FRAME_CLAHE})..."

        "$PY_BIN" "$SCRIPT_DIR/scripts/extract_frames.py" "$VIDEO_FILE" "$FRAMES_DIR" \
            --fps "$EXTRACT_FPS" \
            --min-gap-seconds "$FRAME_MIN_GAP_SECONDS" \
            --max-frames "$FRAME_MAX_FRAMES" \
            --min-sharpness "$FRAME_MIN_SHARPNESS" \
            --sharpness-ratio "$FRAME_SHARPNESS_RATIO" \
            --min-brightness "$FRAME_MIN_BRIGHTNESS" \
            --clahe "$FRAME_CLAHE" \
            --clahe-clip "$FRAME_CLAHE_CLIP" \
            --clahe-grid "$FRAME_CLAHE_GRID" \
            --denoise "$FRAME_DENOISE" \
            --unsharp "$FRAME_UNSHARP" \
            --unsharp-amount "$FRAME_UNSHARP_AMOUNT" \
            --jpeg-quality "$FRAME_JPEG_QUALITY" \
            --report "$PROJECT_DIR/selection_report.json"

        touch "$SELECTED_SENTINEL"
        FRAME_COUNT=$(find "$FRAMES_DIR" -name "*.jpg" | wc -l)
        log "Frame curation complete: $FRAME_COUNT curated frames."

        if [[ "$FRAME_COUNT" -lt 20 ]]; then
            warn "Only $FRAME_COUNT frames survived curation."
            warn "  - Lower FRAME_MIN_SHARPNESS / FRAME_SHARPNESS_RATIO, or raise EXTRACT_FPS,"
            warn "  - or the video is genuinely too blurry -> re-capture with more light."
        fi
        if [[ "$FRAME_COUNT" -gt 500 ]]; then
            warn "$FRAME_COUNT frames. Consider raising FRAME_SHARPNESS_RATIO / FRAME_MIN_GAP_SECONDS or setting FRAME_MAX_FRAMES."
        fi
    fi

    timer_end
fi

# ══════════════════════════════════════════════════════════
# PHASE 2: Camera Pose Estimation (COLMAP)
# ══════════════════════════════════════════════════════════

if [[ "$START_PHASE" -le 2 ]]; then
    phase 2 "Camera Pose Estimation (COLMAP)"
    timer_start

    if [[ -d "$COLMAP_UNDISTORTED/sparse" ]] && [[ -n "$(ls -A "$COLMAP_UNDISTORTED/sparse" 2>/dev/null)" ]]; then
        warn "COLMAP undistorted output already exists. Skipping."
        warn "Delete $COLMAP_DIR to re-run COLMAP."
    else
        # Step 2a: Feature Extraction
        log "Step 2a: Feature extraction..."
        colmap feature_extractor \
            --database_path "$COLMAP_DB" \
            --image_path "$FRAMES_DIR" \
            --ImageReader.camera_model "$COLMAP_CAMERA_MODEL" \
            --ImageReader.single_camera "$COLMAP_SINGLE_CAMERA" \
            --SiftExtraction.use_gpu "$COLMAP_USE_GPU" \
            --SiftExtraction.gpu_index "$COLMAP_GPU_INDEX" \
            --SiftExtraction.max_num_features "$COLMAP_MAX_FEATURES" \
            --SiftExtraction.estimate_affine_shape 1 \
            --SiftExtraction.domain_size_pooling 1

        # Step 2b: Feature Matching
        log "Step 2b: Feature matching (${COLMAP_MATCHER})..."
        if [[ "$COLMAP_MATCHER" == "exhaustive" ]]; then
            colmap exhaustive_matcher \
                --database_path "$COLMAP_DB" \
                --SiftMatching.use_gpu "$COLMAP_USE_GPU" \
                --SiftMatching.gpu_index "$COLMAP_GPU_INDEX" \
                --SiftMatching.guided_matching "$COLMAP_GUIDED_MATCHING"
        elif [[ "$COLMAP_MATCHER" == "sequential" ]]; then
            colmap sequential_matcher \
                --database_path "$COLMAP_DB" \
                --SiftMatching.use_gpu "$COLMAP_USE_GPU" \
                --SiftMatching.gpu_index "$COLMAP_GPU_INDEX" \
                --SiftMatching.guided_matching "$COLMAP_GUIDED_MATCHING" \
                --SequentialMatching.loop_detection 1
        else
            error "Unknown matcher type: $COLMAP_MATCHER. Use 'exhaustive' or 'sequential'."
        fi

        # Step 2c: Sparse Reconstruction (Mapping)
        log "Step 2c: Sparse reconstruction..."
        colmap mapper \
            --database_path "$COLMAP_DB" \
            --image_path "$FRAMES_DIR" \
            --output_path "$COLMAP_SPARSE"

        # COLMAP can split the scene into several disconnected models when
        # frames are blurry / feature-poor. Pick the LARGEST one instead of
        # blindly using sparse/0 (which is often a tiny degenerate fragment).
        BEST_SPARSE=""
        BEST_COUNT=0
        MODEL_COUNT=0
        for model_dir in "$COLMAP_SPARSE"/*/; do
            [[ -f "$model_dir/images.bin" ]] || continue
            MODEL_COUNT=$((MODEL_COUNT + 1))
            CNT=$(colmap model_analyzer --path "$model_dir" 2>/dev/null \
                  | grep -iE "registered images" | grep -oE '[0-9]+' | head -1 || true)
            CNT=${CNT:-0}
            log "  model $model_dir: $CNT registered images"
            if [[ "$CNT" -gt "$BEST_COUNT" ]]; then
                BEST_COUNT=$CNT
                BEST_SPARSE="$model_dir"
            fi
        done

        if [[ -z "$BEST_SPARSE" ]]; then
            error "COLMAP reconstruction failed. No sparse model with images.bin found."
        fi

        TOTAL_FRAMES=$(find "$FRAMES_DIR" -name "*.jpg" | wc -l)
        log "Reconstruction complete. Using largest model: $BEST_SPARSE ($BEST_COUNT/$TOTAL_FRAMES images registered)"

        if [[ "$MODEL_COUNT" -gt 1 ]]; then
            warn "COLMAP split the scene into $MODEL_COUNT disconnected models."
            warn "This usually means blurry/feature-poor frames. Try:"
            warn "  - lower FRAME_SHARPNESS_RATIO / raise EXTRACT_FPS to keep more sharp frames"
            warn "  - increase FRAME_CLAHE (already on) or add FRAME_DENOISE=1"
            warn "  - re-capture the video with more light and slower movement"
        fi

        PCT=$(awk "BEGIN{printf \"%.0f\", ($TOTAL_FRAMES>0 ? $BEST_COUNT*100/$TOTAL_FRAMES : 0)}")
        if [[ "$BEST_COUNT" -lt 20 ]] || [[ "$PCT" -lt 40 ]]; then
            warn "Only $PCT% of frames registered. The reconstruction will be poor."
        fi

        # Step 2d: Image Undistortion
        log "Step 2d: Image undistortion..."
        colmap image_undistorter \
            --image_path "$FRAMES_DIR" \
            --input_path "$BEST_SPARSE" \
            --output_path "$COLMAP_UNDISTORTED" \
            --output_type COLMAP

        log "COLMAP processing complete."
        log "Undistorted output: $COLMAP_UNDISTORTED"
    fi

    timer_end
fi

# ══════════════════════════════════════════════════════════
# PHASE 3: 3DGS Training
# ══════════════════════════════════════════════════════════

if [[ "$START_PHASE" -le 3 ]]; then
    phase 3 "3D Gaussian Splatting Training"
    timer_start

    PLY_OUTPUT="$TRAIN_DIR/point_cloud/iteration_${TRAIN_ITERATIONS}/point_cloud.ply"

    if [[ -f "$PLY_OUTPUT" ]]; then
        warn "Training output already exists: $PLY_OUTPUT"
        warn "Delete $TRAIN_DIR to retrain."
    else
        GS_DIR="$(realpath "$GAUSSIAN_SPLATTING_DIR")"
        INPUT_DIR="$(realpath "$COLMAP_UNDISTORTED")"
        ABS_TRAIN_DIR="$(realpath "$TRAIN_DIR")"

        if [[ ! -d "$GS_DIR/venv" ]]; then
            error "3DGS virtual environment not found. Run ./setup.sh first."
        fi

        log "Training with:"
        log "  Input:      $INPUT_DIR"
        log "  Output:     $ABS_TRAIN_DIR"
        log "  Iterations: $TRAIN_ITERATIONS"
        log "  Resolution: 1/${TRAIN_RESOLUTION}"
        log "  Densify threshold: $TRAIN_DENSIFY_GRAD_THRESHOLD"

        (
            cd "$GS_DIR"
            source venv/bin/activate

            python train.py \
                -s "$INPUT_DIR" \
                -m "$ABS_TRAIN_DIR" \
                --iterations "$TRAIN_ITERATIONS" \
                --resolution "$TRAIN_RESOLUTION" \
                --densify_grad_threshold "$TRAIN_DENSIFY_GRAD_THRESHOLD" \
                --test_iterations "$TRAIN_ITERATIONS" \
                --save_iterations "$TRAIN_ITERATIONS"

            deactivate
        )

        if [[ ! -f "$PLY_OUTPUT" ]]; then
            # Check if saved at a different iteration count
            LATEST_PLY=$(find "$TRAIN_DIR/point_cloud" -name "point_cloud.ply" 2>/dev/null | sort | tail -1)
            if [[ -n "$LATEST_PLY" ]]; then
                PLY_OUTPUT="$LATEST_PLY"
                warn "PLY found at: $PLY_OUTPUT (different iteration than expected)"
            else
                error "Training completed but no PLY file found in $TRAIN_DIR"
            fi
        fi

        PLY_SIZE=$(du -h "$PLY_OUTPUT" | cut -f1)
        log "Training complete. PLY output: $PLY_OUTPUT ($PLY_SIZE)"
    fi

    timer_end
fi

# ══════════════════════════════════════════════════════════
# PHASE 4: PLY → .ksplat Conversion
# ══════════════════════════════════════════════════════════

if [[ "$START_PHASE" -le 4 ]]; then
    phase 4 "PLY → .ksplat Conversion"
    timer_start

    KSPLAT_OUTPUT="$OUTPUT_DIR/${PROJECT_NAME}.ksplat"

    # Find the PLY file
    if [[ -z "${PLY_OUTPUT:-}" ]] || [[ ! -f "${PLY_OUTPUT:-}" ]]; then
        PLY_OUTPUT=$(find "$TRAIN_DIR/point_cloud" -name "point_cloud.ply" 2>/dev/null | sort | tail -1)
        if [[ -z "$PLY_OUTPUT" ]]; then
            error "No PLY file found. Run phase 3 first."
        fi
    fi

    if [[ -f "$KSPLAT_OUTPUT" ]]; then
        warn "KSplat output already exists: $KSPLAT_OUTPUT"
        warn "Delete it to re-convert."
    else
        GS3D_DIR="$(realpath "$GAUSSIAN_SPLATS_3D_DIR")"
        ABS_PLY="$(realpath "$PLY_OUTPUT")"
        ABS_KSPLAT="$(realpath "$OUTPUT_DIR")/${PROJECT_NAME}.ksplat"

        if [[ ! -f "$GS3D_DIR/util/create-ksplat.js" ]]; then
            error "create-ksplat.js not found. Run ./setup.sh first."
        fi

        log "Converting PLY → .ksplat (compression level $KSPLAT_COMPRESSION)..."
        log "  Input:  $ABS_PLY"
        log "  Output: $ABS_KSPLAT"

        (
            cd "$GS3D_DIR"
            node util/create-ksplat.js "$ABS_PLY" "$ABS_KSPLAT" "$KSPLAT_COMPRESSION"
        )

        if [[ ! -f "$ABS_KSPLAT" ]]; then
            error "Conversion failed. No .ksplat file generated."
        fi

        KSPLAT_SIZE=$(du -h "$ABS_KSPLAT" | cut -f1)
        PLY_SIZE=$(du -h "$ABS_PLY" | cut -f1)
        log "Conversion complete: $KSPLAT_SIZE (from $PLY_SIZE PLY)"
    fi

    timer_end
fi

# ══════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════

TOTAL_ELAPSED=$(( $(date +%s) - PIPELINE_START ))

echo ""
log "=========================================="
log "  Pipeline Complete!"
log "=========================================="
echo ""
echo "  Total time: $((TOTAL_ELAPSED/60))m $((TOTAL_ELAPSED%60))s"
echo ""
echo "  Project:  $PROJECT_DIR"
echo "  Frames:   $(find "$FRAMES_DIR" -name "*.jpg" 2>/dev/null | wc -l) images"

if [[ -f "${PLY_OUTPUT:-/dev/null}" ]]; then
    echo "  PLY:      $(du -h "$PLY_OUTPUT" | cut -f1) → $PLY_OUTPUT"
fi

KSPLAT_FILE="$OUTPUT_DIR/${PROJECT_NAME}.ksplat"
if [[ -f "$KSPLAT_FILE" ]]; then
    echo "  KSplat:   $(du -h "$KSPLAT_FILE" | cut -f1) → $KSPLAT_FILE"
fi

echo ""
echo "  Your .ksplat file is ready for use with @mkkellogg/gaussian-splats-3d"
echo ""
