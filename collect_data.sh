#!/bin/bash

task_name=${1}
task_config=${2}
gpu_id=${3}
run_id=${4:-}

./script/.update_path.sh > /dev/null 2>&1

export CUDA_VISIBLE_DEVICES=${gpu_id}

if [ -n "${run_id}" ]; then
    PYTHONWARNINGS=ignore::UserWarning \
    python script/collect_data.py "${task_name}" "${task_config}" --run-id "${run_id}"
else
    PYTHONWARNINGS=ignore::UserWarning \
    python script/collect_data.py "${task_name}" "${task_config}"
fi
