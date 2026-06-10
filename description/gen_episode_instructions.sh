task_name=${1}
setting=${2}
max_num=${3}
data_dir=${4:-}

if [ -n "${data_dir}" ]; then
    python utils/generate_episode_instructions.py "${task_name}" "${setting}" "${max_num}" --data-dir "${data_dir}"
else
    python utils/generate_episode_instructions.py "${task_name}" "${setting}" "${max_num}"
fi
