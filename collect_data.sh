#!/bin/bash

task_name=${1}
task_config=${2}
gpu_id=${3}
run_id=${4:-}

./script/.update_path.sh > /dev/null 2>&1

export CUDA_VISIBLE_DEVICES=${gpu_id}

config_path="task_config/${task_config}.yml"
collector=$(python -c '
import sys
import yaml

with open(sys.argv[1], "r", encoding="utf-8") as config_file:
    config = yaml.safe_load(config_file) or {}

print(
    "script/collect_controlled_failures.py"
    if "intervention" in config
    else "script/collect_data.py"
)
' "${config_path}") || exit 1

if [ -n "${run_id}" ]; then
    PYTHONWARNINGS=ignore::UserWarning \
    python "${collector}" "${task_name}" "${task_config}" --run-id "${run_id}"
else
    PYTHONWARNINGS=ignore::UserWarning \
    python "${collector}" "${task_name}" "${task_config}"
fi
