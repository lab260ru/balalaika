#!/bin/bash
# Run the preprocess sub-stages (chunking + crest filter + loudness norm).
SCRIPT_DIR=$(dirname "$(realpath "$0")")
. "$SCRIPT_DIR/../stage_runner.sh"
stage_init "$@"

stage_run src.preprocess.preprocess
stage_run src.preprocess.crest_factor_remover
stage_run src.preprocess.preprocess_audio
