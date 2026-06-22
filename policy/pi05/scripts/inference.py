import importlib
import argparse
import os
import sys
import yaml
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
os.chdir(REPO_ROOT)
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "policy"))
sys.path.insert(0, os.path.join(REPO_ROOT, "description", "utils"))

from envs import CONFIGS_PATH
from generate_episode_instructions import generate_episode_descriptions
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config



def parse_args_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(REPO_ROOT, config_path)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    def parse_override_pairs(pairs):
        override_dict = {}
        for i in range(0, len(pairs), 2):
            key = pairs[i].lstrip("--")
            value = pairs[i + 1]
            try:
                value = eval(value)
            except:
                pass
            override_dict[key] = value
        return override_dict

    if args.overrides:
        overrides = parse_override_pairs(args.overrides)
        config.update(overrides)

    return config

def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No Task")
    return env_instance


def get_embodiment_config(robot_file):
    with open(os.path.join(robot_file, "config.yml"), "r", encoding="utf-8") as f:
        return yaml.load(f.read(), Loader=yaml.FullLoader)


def build_task_args(usr_args):
    task_config = usr_args["task_config"]
    if isinstance(task_config, dict):
        args = dict(task_config)
    else:
        task_config_path = os.path.join(REPO_ROOT, "task_config", f"{task_config}.yml")
        with open(task_config_path, "r", encoding="utf-8") as f:
            args = yaml.load(f.read(), Loader=yaml.FullLoader)

    runtime_keys = {
        "checkpoint_dir",
        "checkpoint_id",
        "ckpt_setting",
        "instruction_type",
        "model_name",
        "policy_name",
        "prompt",
        "seed",
        "task_config",
        "train_config_name",
    }
    
    args.update({k: v for k, v in usr_args.items() if v is not None and k not in runtime_keys})
    args["task_name"] = usr_args["task_name"]

    with open(os.path.join(CONFIGS_PATH, "_embodiment_config.yml"), "r", encoding="utf-8") as f:
        embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise ValueError(f"No embodiment file configured for {embodiment_type}")
        return robot_file

    embodiment_type = args["embodiment"]
    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise ValueError("embodiment items should be 1 or 3")

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])
    return args


def encode_obs(observation):
    input_rgb_arr = [
        observation["observation"]["head_camera"]["rgb"],
        observation["observation"]["right_camera"]["rgb"],
        observation["observation"]["left_camera"]["rgb"],
    ]
    input_state = observation["joint_action"]["vector"]

    return input_rgb_arr, input_state


def get_scene_prompt(task_name, task_args, seed, instruction_type):
    prompt_env = class_decorator(task_name)
    try:
        prompt_env.setup_demo(now_ep_num=0, seed=seed, is_test=True, **task_args)
        episode_info = prompt_env.play_once()
        results = generate_episode_descriptions(task_name, [episode_info["info"]], 100)
        prompts = results[0].get(instruction_type) or results[0].get("seen") or results[0].get("unseen")
        if not prompts:
            raise ValueError(f"No {instruction_type} instructions generated for {task_name}")
        return np.random.choice(prompts)
    finally:
        prompt_env.close_env()


def main(usr_args):
    task_args = build_task_args(usr_args)
    prompt = usr_args.get("prompt")
    if prompt is None:
        prompt = get_scene_prompt(
            usr_args["task_name"],
            task_args,
            usr_args["seed"],
            usr_args.get("instruction_type", "unseen"),
        )

    TASK_ENV = class_decorator(usr_args["task_name"])
    TASK_ENV.setup_demo(now_ep_num=0, seed=usr_args["seed"], is_test=True, **task_args)
    TASK_ENV.set_instruction(prompt)
    
    config = _config.get_config(usr_args["train_config_name"])
    checkpoint_dir = usr_args.get("checkpoint_dir")
    if checkpoint_dir is None:
        checkpoint_dir = (
            f"{REPO_ROOT}/policy/pi05/checkpoints/{usr_args['train_config_name']}/"
            f"{usr_args['model_name']}/{usr_args.get('checkpoint_id', 30000)}"
        )
    elif not os.path.isabs(checkpoint_dir):
        checkpoint_dir = os.path.join(REPO_ROOT, checkpoint_dir)

    assets_dir = os.path.join(checkpoint_dir, "assets")
    robotwin_repo_id = None
    if os.path.isdir(assets_dir):
        asset_entries = [
            entry
            for entry in os.listdir(assets_dir)
            if os.path.isdir(os.path.join(assets_dir, entry))
        ]
        if asset_entries:
            robotwin_repo_id = asset_entries[0]

    policy = _policy_config.create_trained_policy(
        config,
        checkpoint_dir,
        robotwin_repo_id=robotwin_repo_id,
    )

    obs = TASK_ENV.get_obs()
    rgb_arr, input_state = encode_obs(obs)
    parsed_obs = {
        "state": input_state,
        "images": {
            "cam_high": rgb_arr[0].transpose(2, 0, 1),
            "cam_right_wrist": rgb_arr[1].transpose(2, 0, 1),
            "cam_left_wrist": rgb_arr[2].transpose(2, 0, 1),
        },
        "prompt": prompt,
    }
    result = policy.infer(parsed_obs)
    print("Action shape: ", result["actions"].shape)



if __name__ == "__main__":

    usr_args = parse_args_and_config()

    main(usr_args)
