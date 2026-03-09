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
    phase 1 "Frame Extraction (FFmpeg)"
    timer_start

    FRAME_COUNT=$(find "$FRAMES_DIR" -name "*.jpg" 2>/dev/null | wc -l)

    if [[ "$FRAME_COUNT" -gt 0 ]]; then
        warn "Frames directory already has $FRAME_COUNT images. Skipping extraction."
        warn "Delete $FRAMES_DIR/*.jpg to re-extract."
    else
        log "Extracting frames at ${EXTRACT_FPS} fps with JPEG quality ${JPEG_QUALITY}..."

        ffmpeg -i "$VIDEO_FILE" \
            -vf "fps=${EXTRACT_FPS}" \
            -q:v "$JPEG_QUALITY" \
            "$FRAMES_DIR/frame_%04d.jpg"

        FRAME_COUNT=$(find "$FRAMES_DIR" -name "*.jpg" | wc -l)
        log "Extracted $FRAME_COUNT frames."

        if [[ "$FRAME_COUNT" -lt 20 ]]; then
            warn "Only $FRAME_COUNT frames extracted. Consider increasing EXTRACT_FPS in config.env."
        fi

        if [[ "$FRAME_COUNT" -gt 500 ]]; then
            warn "$FRAME_COUNT frames extracted. This may slow down COLMAP significantly."
            warn "Consider reducing EXTRACT_FPS or switching COLMAP_MATCHER to 'sequential' in config.env."
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
            --SiftExtraction.use_gpu 1 \
            --SiftExtraction.gpu_index "$COLMAP_GPU_INDEX"

        # Step 2b: Feature Matching
        log "Step 2b: Feature matching (${COLMAP_MATCHER})..."
        if [[ "$COLMAP_MATCHER" == "exhaustive" ]]; then
            colmap exhaustive_matcher \
                --database_path "$COLMAP_DB" \
                --SiftMatching.use_gpu 1 \
                --SiftMatching.gpu_index "$COLMAP_GPU_INDEX"
        elif [[ "$COLMAP_MATCHER" == "sequential" ]]; then
            colmap sequential_matcher \
                --database_path "$COLMAP_DB" \
                --SiftMatching.use_gpu 1 \
                --SiftMatching.gpu_index "$COLMAP_GPU_INDEX" \
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

        # Verify reconstruction
        if [[ ! -d "$COLMAP_SPARSE/0" ]]; then
            error "COLMAP reconstruction failed. No sparse/0 directory found."
        fi

        # Count registered images
        log "Reconstruction complete. Checking registration..."
        TOTAL_FRAMES=$(find "$FRAMES_DIR" -name "*.jpg" | wc -l)
        # Use COLMAP model analyzer to check stats
        colmap model_analyzer --path "$COLMAP_SPARSE/0" 2>&1 | head -20 || true

        # Step 2d: Image Undistortion
        log "Step 2d: Image undistortion..."
        colmap image_undistorter \
            --image_path "$FRAMES_DIR" \
            --input_path "$COLMAP_SPARSE/0" \
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
