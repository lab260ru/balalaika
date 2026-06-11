#!/bin/bash
SCRIPT_DIR=$(dirname "$(realpath "$0")")
. "$SCRIPT_DIR/../stage_runner.sh"
stage_init "$@"

stage_run src.transcription.transcription
