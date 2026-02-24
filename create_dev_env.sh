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
        
        echo "Installing ONNX Runtime GPU (CUDA 13 nightly)..."
        uv pip install coloredlogs flatbuffers numpy packaging protobuf sympy
        uv pip install --pre --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ort-cuda-13-nightly/pypi/simple/ onnxruntime-gpu
        uv pip install tensorrt-cu13
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