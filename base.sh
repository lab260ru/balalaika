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
#   6  Anti-spoofing scoring        (src.separation.antispoofing)
#   6.5 Anti-spoofing filter         (src.separation.antispoofing_filter)
#   7  Transcription                (src.transcription.transcription)
#   8  Punctuation                  (src.punctuation.punctuation)
#   9  Accents                      (src.accents.accents)
#   10 Phonemizer                   (src.phonemizer.phonemizer)
#   11 Denoising / enhancement      (src.denoising.denoising)
#   12 Collate -> parquet           (src.collate)
#   13 Export -> WebDataset         (src.to_webdataset)
#   14 Filter report                (src.report)
#
# Run a single stage:    bash base.sh --stage 7 --stop_stage 7
# Run from a checkpoint: bash base.sh --stage 4
# =============================================================================
set -euo pipefail

# ---- defaults ---------------------------------------------------------------
config_path="configs/config.yaml"
stage=11
stop_stage=14
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
runtime_exports="$(python3 -m src.utils.runtime_env --config_path "$config_path")" || {
    echo "Failed to read runtime env from $config_path" >&2
    exit 1
}
eval "$runtime_exports"

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

# ---- cudf.pandas accelerator (OPT-IN) ---------------------------------------
# cudf.pandas transparently routes pandas calls to the GPU. It is now OPT-IN
# (default OFF) for two concrete reasons:
#
#  1. csv_manager's hot path is the pyarrow CSV engine (report.md §4.1:
#     3.4x read / 4.1x write). csv_manager deliberately DISABLES both pyarrow
#     fast paths when it detects the cudf proxy, so enabling cudf.pandas
#     silently forfeits those measured wins and routes the CSV-heavy path back
#     through cuDF interception (fcntl-locked read/merge/write cycles full of
#     ops cudf must D2H-fallback for).
#  2. The blanket prefix injected cudf.pandas.install() into EVERY stage,
#     including the GPU multiprocessing stages (ASR, denoising, DistillMOS,
#     music_detect, antispoofing) whose parents do little parent-side pandas.
#     Each such process then holds a cuDF CUDA context (hundreds of MB of VRAM)
#     that contends with ORT/torch — an OOM risk on GPUs shared with a
#     training job, not merely overhead.
#
# Opt back in on a node that genuinely benefits (CSV-bound stages, idle GPU)
# with BALALAIKA_ENABLE_CUDF=1. BALALAIKA_DISABLE_CUDF=1 still forces it off
# and wins if both are set.
cudf_prefix=()
if [[ "${BALALAIKA_DISABLE_CUDF:-0}" == "1" ]]; then
    echo "cudf.pandas: disabled via BALALAIKA_DISABLE_CUDF=1"
elif [[ "${BALALAIKA_ENABLE_CUDF:-0}" != "1" ]]; then
    echo "cudf.pandas: off by default (set BALALAIKA_ENABLE_CUDF=1 to opt in; keeps pyarrow CSV fast paths)"
elif python3 -c "import cudf.pandas" >/dev/null 2>&1; then
    cudf_prefix=(python3 -m cudf.pandas)
    echo "cudf.pandas: enabled via BALALAIKA_ENABLE_CUDF=1 (Pandas Accelerator Mode)"
else
    echo "cudf.pandas: requested via BALALAIKA_ENABLE_CUDF=1 but not importable; falling back to vanilla pandas"
fi

# ---- thread-pool hygiene (OPT-IN) -------------------------------------------
# This box has 48 logical cores shared with the user's training job. Left
# uncapped, OpenMP/OpenBLAS/MKL each lazily spawn a full per-process thread
# team (one per visible core) in every stage process AND every forked child
# (loader workers, probe pools). With several workers that is hundreds of
# spinning threads fighting the GPU EPs and the co-resident training job.
#
# runtime.threads_per_worker (BALALAIKA_THREADS_PER_WORKER) caps those teams
# for all stage processes and the children they fork. Empty (the default)
# exports nothing, so single-worker latency is unchanged (library defaults).
if [[ -n "${BALALAIKA_THREADS_PER_WORKER:-}" ]]; then
    export OMP_NUM_THREADS="$BALALAIKA_THREADS_PER_WORKER"
    export OPENBLAS_NUM_THREADS="$BALALAIKA_THREADS_PER_WORKER"
    export MKL_NUM_THREADS="$BALALAIKA_THREADS_PER_WORKER"
    export NUMEXPR_NUM_THREADS="$BALALAIKA_THREADS_PER_WORKER"
    echo "thread caps: OMP/OPENBLAS/MKL/NUMEXPR = $BALALAIKA_THREADS_PER_WORKER per worker"
fi

# ---- helpers ----------------------------------------------------------------
run_python() {
    # Run a Python module with optional CPU pinning + the configured log dir.
    local module="$1"; shift
    local extra_args=("$@")
    local cmd

    if (( ${#cudf_prefix[@]} > 0 )); then
        # Do NOT use:
        #   python3 -m cudf.pandas -m "$module" --config_path ...
        #
        # Because cudf.pandas CLI parses arguments itself and can fail on
        # project-specific args like --config_path / --log_dir.
        #
        # Instead, install cudf.pandas hook inside Python before importing
        # the target module, then run the target module with corrected sys.argv.
        cmd=(python3 -c '
import runpy
import sys

import cudf.pandas
cudf.pandas.install()

module = sys.argv[1]
sys.argv = [module] + sys.argv[2:]

runpy.run_module(module, run_name="__main__", alter_sys=True)
' "$module" --config_path "$config_path" --log_dir "$BALALAIKA_LOG_DIR" "${extra_args[@]}")
    else
        cmd=(python3 -m "$module" --config_path "$config_path" --log_dir "$BALALAIKA_LOG_DIR" "${extra_args[@]}")
    fi

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
    echo "Stage 6: Anti-spoofing — raw Spectra-0 scoring"
    run_python src.separation.antispoofing
    check_stage_status 6
fi

if stage_active 6.5; then
    echo "Stage 6.5: Anti-spoofing filter — spoof-margin deletion"
    run_python src.separation.antispoofing_filter
    check_stage_status 6.5
fi

if stage_active 7; then
    echo "Stage 7: Transcription — onnx-asr + ROVER"
    run_python src.transcription.transcription
    check_stage_status 7
fi

if stage_active 8; then
    echo "Stage 8: Punctuation — RUPunct"
    run_python src.punctuation.punctuation
    check_stage_status 8
fi

if stage_active 9; then
    echo "Stage 9: Accents — ruAccent"
    run_python src.accents.accents
    check_stage_status 9
fi

if stage_active 10; then
    echo "Stage 10: Phonemizer — TryIParu G2P"
    run_python src.phonemizer.phonemizer
    check_stage_status 10
fi

if stage_active 11; then
    echo "Stage 11: Denoising — ClearVoice MossFormer2_SE_48K"
    run_python src.denoising.denoising
    check_stage_status 11
fi

if stage_active 12; then
    echo "Stage 12: Collate — balalaika.parquet"
    run_python src.collate
    check_stage_status 12
fi

if stage_active 13; then
    echo "Stage 13: Export — WebDataset shards"
    run_python src.to_webdataset
    check_stage_status 13
fi

if stage_active 14; then
    echo "Stage 14: Filter report — filter_report.md"
    run_python src.report --quiet
    check_stage_status 14
fi

echo -e "\n\033[1;32mPipeline finished (stages ${stage}..${stop_stage})\033[0m"
