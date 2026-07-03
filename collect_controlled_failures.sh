#!/bin/bash
set -e

task_name=${1}
task_config=${2}
gpu_id=${3}
run_id=${4:-}

if [ -z "${task_name}" ] || [ -z "${task_config}" ] || [ -z "${gpu_id}" ]; then
    echo "Usage: $0 <task_name> <task_config> <gpu_id> [run_id]"
    echo "Example: $0 adjust_bottle_controlled multiple_interventions 0"
    exit 1
fi

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${repo_dir}"

if [ -f "./script/.update_path.sh" ]; then
    ./script/.update_path.sh > /dev/null 2>&1
fi

export PYTHONPATH="${repo_dir}:${repo_dir}/policy:${repo_dir}/description/utils${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES=${gpu_id}
export PYTHONWARNINGS=ignore::UserWarning

if [ -n "${run_id}" ]; then
    python script/collect_controlled_failures.py "${task_name}" "${task_config}" --run-id "${run_id}"
else
    python script/collect_controlled_failures.py "${task_name}" "${task_config}"
fi
