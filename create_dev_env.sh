#!/bin/bash
set -euo pipefail

create_venv_env() {
    local env_name=$1
    local requirements_file=$2
    
    if [ ! -d "$env_name" ]; then
        echo "Creating $env_name environment..."
        uv venv "$env_name"
    
        if [ -f "$env_name/Scripts/activate" ]; then
            source "$env_name/Scripts/activate"
        elif [ -f "$env_name/bin/activate" ]; then
            source "$env_name/bin/activate"
        else
            echo "Error: Could not find activate script in $env_name"
            exit 1
        fi
        
        uv pip install -r "$requirements_file"
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