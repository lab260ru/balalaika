#!/bin/bash
set -euo pipefail

activate_venv() {
    local venv_path=$1
    if [ ! -f "$venv_path/bin/activate" ]; then
        echo "Error: Virtual environment not found at $venv_path"
        exit 1
    fi
    source "$venv_path/bin/activate"
    echo "Activated: $(which python)"
    
    local python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    local nvidia_base="$venv_path/lib/python$python_version/site-packages/nvidia"
    
    if [ -d "$nvidia_base" ]; then
        export LD_LIBRARY_PATH="${nvidia_base}/cublas/lib:${nvidia_base}/cudnn/lib:${nvidia_base}/cuda_runtime/lib:${nvidia_base}/cuda_nvrtc/lib:${nvidia_base}/cufft/lib:${nvidia_base}/nvjitlink/lib:${nvidia_base}/cusolver/lib:${nvidia_base}/cusparse/lib:${LD_LIBRARY_PATH:-}"
    fi
    
    local trt_libs="$venv_path/lib/python$python_version/site-packages/tensorrt_libs"
    if [ -d "$trt_libs" ]; then
        export LD_LIBRARY_PATH="${trt_libs}:${LD_LIBRARY_PATH:-}"
    fi
}

if [ -z "${1:-}" ]; then
    echo "Usage: $0 <config_path>"
    exit 1
fi

CONFIG_PATH=$(realpath "$1")
echo $CONFIG_PATH "--src"
SCRIPTS=(
    # "./src/download/download_yaml.sh"
    "./src/preprocess/preprocess_yaml.sh"
    "./src/separation/separation_yaml.sh"
    "./src/transcription/transcription_yaml.sh"
    "./src/punctuation/punctuation_yaml.sh"
    "./src/accents/accents_yaml.sh"
    "./src/phonemizer/phonemizer_yaml.sh"
    "./src/collate_yamls.sh"
)

activate_venv ".dev_venv"

for script in "${SCRIPTS[@]}"; do
    echo -e "\n\033[1;34m=== Executing $script ===\033[0m"

    if [ ! -f "$script" ]; then
        echo -e "\033[1;31mError: Script $script not found\033[0m"
        exit 1
    fi

    bash "$script" "$CONFIG_PATH" || {
        echo -e "\033[1;31mError in $script\033[0m"
        exit 1
    }
done

echo -e "\n\033[1;32mAll scripts executed successfully!\033[0m"
