#!/bin/bash
set -euo pipefail

create_venv_env() {
    local env_name=$1
    local requirements_file=$2
    
    if [ ! -d "$env_name" ]; then
        echo "Creating $env_name environment..."
        uv venv "$env_name" --python 3.12
    
        if [ -f "$env_name/Scripts/activate" ]; then
            source "$env_name/Scripts/activate"
        elif [ -f "$env_name/bin/activate" ]; then
            source "$env_name/bin/activate"
        else
            echo "Error: Could not find activate script in $env_name"
            exit 1
        fi
        
        uv pip install -r "$requirements_file"

        # ONNX Runtime GPU: stable CUDA-12 build. The CUDA-13 nightly feed
        # needs a CUDA 13 driver (>= 580); nodes on CUDA 12.x drivers cannot
        # load it at all. TensorRT must stay on the 10.x line: onnxruntime-gpu
        # links libnvinfer.so.10, while a plain `pip install tensorrt-cu12`
        # now resolves to TensorRT 11 (libnvinfer.so.11) and the TRT provider
        # silently falls back to CPU.
        echo "Installing ONNX Runtime GPU (CUDA 12 stable) + TensorRT 10..."
        uv pip install coloredlogs flatbuffers numpy packaging protobuf sympy
        uv pip install onnxruntime-gpu "tensorrt-cu12==10.*"
        uv pip install onnx-asr[gpu,hub]
        
        deactivate
    else
        echo "Environment $env_name already exists"
        
        if [ -f "$env_name/Scripts/activate" ]; then
            source "$env_name/Scripts/activate"
        elif [ -f "$env_name/bin/activate" ]; then
            source "$env_name/bin/activate"
        else
            echo "Error: Could not find activate script in $env_name"
            exit 1
        fi
        
        if ! pip check; then
            echo "Some dependencies are missing or incompatible in $env_name, reinstalling..."
            pip install -r "$requirements_file" --force-reinstall
        fi
        deactivate
    fi
}

create_venv_env ".dev_venv" "requirements_dev.txt"