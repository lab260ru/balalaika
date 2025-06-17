set -euo pipefail

MAIN_VENV=".main_venv"
SUPPORT_VENV=".support_venv"

activate_venv() {
    local venv_path=$1
    if [ ! -f "$venv_path/bin/activate" ]; then
        echo "Error: Virtual environment not found at $venv_path"
        exit 1
    fi
    source "$venv_path/bin/activate"
    echo "Activated: $(which python)"
}

[ ! -d "$MAIN_VENV" ] && { echo "Main venv not found at $MAIN_VENV"; exit 1; }
[ ! -d "$SUPPORT_VENV" ] && { echo "Support venv not found at $SUPPORT_VENV"; exit 1; }

SCRIPTS=(
    "./src/download/download_yaml.sh"
    "./src/preprocess/preprocess_yaml.sh"
    "./src/separation/separation_yaml.sh"
    "./src/transcription/transcription_yaml.sh"
    "./src/punctuation/punctuation_yaml.sh"
    "./src/accents/accents_yaml.sh"
    "./src/yofication/yofication_yaml.sh"
    "./src/phonemizer/phonemizer_yaml.sh"
    "./src/classificaton/classification_yaml.sh"
)


activate_venv "$MAIN_VENV"

for script in "${SCRIPTS[@]}"; do
    echo -e "\n\033[1;34m=== Executing $script ===\033[0m"
    
    if [[ "$script" == *"separation_yaml.sh"* || "$script" == *"accents_yaml.sh"* || "$script" == *"classification_yaml.sh"* || "$script" == *"phonemizer_yaml.sh"* ]]; then
        echo "Switching to support virtual environment..."
        deactivate
        activate_venv "$SUPPORT_VENV"
    fi

    if [ ! -f "$script" ]; then
        echo -e "\033[1;31mError: Script $script not found\033[0m"
        exit 1
    fi

    bash "$script" || {
        echo -e "\033[1;31mError in $script\033[0m"
        exit 1
    }
    
    if [[ "$script" == *"separation_yaml.sh"* || "$script" == *"accents_yaml.sh"* ]]; then
        echo "Switching back to main virtual environment..."
        deactivate
        activate_venv "$MAIN_VENV"
    fi
done

echo -e  "\n\033[1;32mAll scripts executed successfully!\033[0m"