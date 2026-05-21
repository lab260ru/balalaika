#!/bin/bash
# =============================================================================
# Balalaika — pipeline orchestrator.
#
# Usage:
#   bash base.sh [--config_path PATH] [--stage N] [--stop_stage N]
#
# Stages (positive integers; ``--stage 0`` runs from the beginning):
#   0  Download                     (Yandex Music; opt-in, see configs/config.yaml -> download)
#   1  Preprocess: chunking         (src.preprocess.preprocess)
#   2  Preprocess: crest filter     (src.preprocess.crest_factor_remover)
#   3  Preprocess: loudness         (src.preprocess.preprocess_audio)
#   4  Separation: music detection  (src.separation.music_detect)
#   5  Separation: DistillMOS       (src.separation.distillmos_process)
#   5.5 DistillMOS filter            (src.separation.distillmos_filter)
#   6  Transcription                (src.transcription.transcription)
#   7  Punctuation                  (src.punctuation.punctuation)
#   8  Accents                      (src.accents.accents)
#   9  Phonemizer                   (src.phonemizer.phonemizer)
#   10 Denoising / enhancement      (src.denoising.denoising)
#   11 Collate -> parquet           (src.collate)
#   12 Export -> WebDataset         (src.to_webdataset)
#   13 Filter report                (src.report)
#
# Run a single stage:    bash base.sh --stage 6 --stop_stage 6
# Run from a checkpoint: bash base.sh --stage 4
# =============================================================================
set -euo pipefail

# ---- defaults ---------------------------------------------------------------
config_path="configs/config.yaml"
stage=1
stop_stage=9
strict_mode=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config_path|--config)
            config_path="$2"; shift 2 ;;
        --stage)
            stage="$2"; shift 2 ;;
        --stop_stage)
            stop_stage="$2"; shift 2 ;;
        --strict)
            strict_mode=1; shift ;;
        --help|-h)
            sed -n '2,28p' "$0"
            exit 0 ;;
        *)
            # Backwards-compat: accept positional config path as before.
            if [[ "$1" != --* ]]; then
                config_path="$1"; shift
            else
                echo "Unknown option: $1"; exit 1
            fi ;;
    esac
done

if [ ! -f "$config_path" ]; then
    echo "Error: config file not found: $config_path" >&2
    exit 1
fi

config_path="$(realpath "$config_path")"
echo "Using config: $config_path"

# ---- runtime env from configs/config.yaml ----------------------------------
# Exports BALALAIKA_VENV / BALALAIKA_CPU_AFFINITY / BALALAIKA_LOG_DIR / ...
if ! eval "$(python3 -m src.utils.runtime_env --config_path "$config_path")"; then
    echo "Failed to read runtime env from $config_path" >&2
    exit 1
fi

# ---- venv activation --------------------------------------------------------
activate_venv() {
    local venv_path=$1
    if [ ! -f "$venv_path/bin/activate" ]; then
        echo "Error: Virtual environment not found at $venv_path" >&2
        exit 1
    fi
    # shellcheck disable=SC1090
    source "$venv_path/bin/activate"
    echo "Activated: $(which python3)"

    local python_version
    python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    local nvidia_base="$venv_path/lib/python$python_version/site-packages/nvidia"
    if [ -d "$nvidia_base" ]; then
        export LD_LIBRARY_PATH="${nvidia_base}/cublas/lib:${nvidia_base}/cudnn/lib:${nvidia_base}/cuda_runtime/lib:${nvidia_base}/cuda_nvrtc/lib:${nvidia_base}/cufft/lib:${nvidia_base}/nvjitlink/lib:${nvidia_base}/cusolver/lib:${nvidia_base}/cusparse/lib:${LD_LIBRARY_PATH:-}"
    fi
    local trt_libs="$venv_path/lib/python$python_version/site-packages/tensorrt_libs"
    if [ -d "$trt_libs" ]; then
        export LD_LIBRARY_PATH="${trt_libs}:${LD_LIBRARY_PATH:-}"
    fi
}

activate_venv "${BALALAIKA_VENV:-.dev_venv}"

mkdir -p "${BALALAIKA_LOG_DIR:-./logs}"

# ---- helpers ----------------------------------------------------------------
run_python() {
    # Run a Python module with optional CPU pinning + the configured log dir.
    local module="$1"; shift
    local extra_args=("$@")
    local cmd=(python3 -m "$module" --config_path "$config_path" --log_dir "$BALALAIKA_LOG_DIR" "${extra_args[@]}")

    if [[ -n "${BALALAIKA_CPU_AFFINITY:-}" ]] && command -v taskset >/dev/null 2>&1; then
        cmd=(taskset -c "$BALALAIKA_CPU_AFFINITY" "${cmd[@]}")
    fi

    echo -e "\n\033[1;34m=== [$module] ===\033[0m"
    "${cmd[@]}"
}

stage_active() {
    # Returns 0 (true) when the requested stage falls into [stage, stop_stage].
    local s="$1"
    local stage_i stop_stage_i s_i
    stage_i=$(stage_to_units "$stage")
    stop_stage_i=$(stage_to_units "$stop_stage")
    s_i=$(stage_to_units "$s")
    (( stage_i <= s_i && stop_stage_i >= s_i ))
}

stage_to_units() {
    # Convert stage numbers to integer hundredths so 5.5 works without bc.
    local value="$1"
    if [[ ! "$value" =~ ^([0-9]+)(\.([0-9]+))?$ ]]; then
        echo "Invalid stage value: $value" >&2
        return 1
    fi
    local whole="${BASH_REMATCH[1]}"
    local frac="${BASH_REMATCH[3]:-}"
    frac="${frac}00"
    frac="${frac:0:2}"
    echo $((10#$whole * 100 + 10#$frac))
}

check_stage_status() {
    local s="$1"
    local status_file="${BALALAIKA_LOG_DIR:-./logs}/stage_${s}_status.json"

    if [[ "${strict_mode:-0}" != "1" ]]; then
        return 0
    fi

    if [[ ! -f "$status_file" ]]; then
        echo -e "\033[1;31m[FAIL] Stage $s: status file not found (stage did not complete)\033[0m" >&2
        exit 1
    fi

    local errors
    errors=$(python3 -c '
import json, sys
with open(sys.argv[1]) as f:
    data = json.load(f)
print(data.get("errors", 0))
' "$status_file")
    if [[ "$errors" -gt 0 ]]; then
        echo -e "\033[1;31m[FAIL] Stage $s: $errors error(s). Pipeline aborted.\033[0m" >&2
        echo "See: $status_file" >&2
        exit 1
    fi
}

# ---- pipeline ---------------------------------------------------------------
if stage_active 0; then
    echo "Stage 0: Download (Yandex Music)"
    run_python src.download.download
    check_stage_status 0
fi

if stage_active 1; then
    echo "Stage 1: Preprocess — Sortformer chunking + Smart Turn refinement"
    run_python src.preprocess.preprocess
    check_stage_status 1
fi

if stage_active 2; then
    echo "Stage 2: Preprocess — crest factor filter"
    run_python src.preprocess.crest_factor_remover
    check_stage_status 2
fi

if stage_active 3; then
    echo "Stage 3: Preprocess — loudness normalization (BS.1770-4)"
    run_python src.preprocess.preprocess_audio
    check_stage_status 3
fi

if stage_active 4; then
    echo "Stage 4: Separation — music detection"
    run_python src.separation.music_detect
    check_stage_status 4
fi

if stage_active 5; then
    echo "Stage 5: Separation — DistillMOS scoring"
    run_python src.separation.distillmos_process
    check_stage_status 5
fi

if stage_active 5.5; then
    echo "Stage 5.5: DistillMOS filter — quality-based deletion"
    run_python src.separation.distillmos_filter
    check_stage_status 5.5
fi

if stage_active 6; then
    echo "Stage 6: Transcription — onnx-asr + ROVER"
    run_python src.transcription.transcription
    check_stage_status 6
fi

if stage_active 7; then
    echo "Stage 7: Punctuation — RUPunct"
    run_python src.punctuation.punctuation
    check_stage_status 7
fi

if stage_active 8; then
    echo "Stage 8: Accents — ruAccent"
    run_python src.accents.accents
    check_stage_status 8
fi

if stage_active 9; then
    echo "Stage 9: Phonemizer — TryIParu G2P"
    run_python src.phonemizer.phonemizer
    check_stage_status 9
fi

if stage_active 10; then
    echo "Stage 10: Denoising — ClearVoice MossFormer2_SE_48K"
    run_python src.denoising.denoising
    check_stage_status 10
fi

if stage_active 11; then
    echo "Stage 11: Collate — balalaika.parquet"
    run_python src.collate
    check_stage_status 11
fi

if stage_active 12; then
    echo "Stage 12: Export — WebDataset shards"
    run_python src.to_webdataset
    check_stage_status 12
fi

if stage_active 13; then
    echo "Stage 13: Filter report — filter_report.md"
    run_python src.report --quiet
    check_stage_status 13
fi

echo -e "\n\033[1;32mPipeline finished (stages ${stage}..${stop_stage})\033[0m"
