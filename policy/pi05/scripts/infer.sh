#!/bin/bash

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 # ensure GPU < 24G

policy_name=pi05
task_name=${1}
task_config=${2}
train_config_name=pi05_base_aloha_lora
gpu_id=${3}
seed=1
checkpoint_dir=/data1/common_data/WorldArena/10data

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

# source .venv/bin/activate
cd ../.. # move to root

PYTHONWARNINGS=ignore::UserWarning \
python /data1/fangxuebin/RoboTwin/policy/pi05/scripts/inference.py --config policy/$policy_name/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --train_config_name ${train_config_name} \
    --checkpoint_dir ${checkpoint_dir} \
    --policy_name ${policy_name} \
    --seed ${seed} 
