#!/bin/bash
SCRIPT_DIR=$(dirname "$(realpath "$0")")
. "$SCRIPT_DIR/../stage_runner.sh"
stage_init "$@"

stage_run src.separation.music_detect
stage_run src.separation.distillmos_process
