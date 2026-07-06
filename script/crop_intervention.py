import argparse
import json
import os
import pickle
import shutil
import subprocess

import h5py
import numpy as np
import yaml


PLANNED_INTERVENTION_PATH_COUNTS = {
    "trajectory_perturbation": 1,
    "move_waypoint": 1,
    # grasp_actor normally plans pre-grasp and grasp moves before closing gripper.
    "grasp_pose_perturbation": 2,
}


def resolve_default_run_dir(task_name, task_config):
    base_dir = os.path.join("data", task_name, task_config)
    latest_path = os.path.join(base_dir, "latest_run.txt")
    if os.path.exists(latest_path):
        with open(latest_path, "r", encoding="utf-8") as f:
            latest = f.read().strip()
        if latest:
            return os.path.join(base_dir, latest)

    runs = [
        name
        for name in os.listdir(base_dir)
        if name.startswith("run_") and os.path.isdir(os.path.join(base_dir, name))
    ]
    if not runs:
        raise FileNotFoundError(f"No run directory found under {base_dir}")
    return os.path.join(base_dir, sorted(runs)[-1])


def load_config(task_config):
    config_path = os.path.join("task_config", f"{task_config}.yml")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.load(f.read(), Loader=yaml.FullLoader)


def group_consecutive(indices):
    if len(indices) == 0:
        return []
    groups = []
    start = prev = int(indices[0])
    for idx in indices[1:]:
        idx = int(idx)
        if idx == prev + 1:
            prev = idx
        else:
            groups.append((start, prev))
            start = prev = idx
    groups.append((start, prev))
    return groups


def intervention_specs(config):
    specs = []
    for idx in range(config.get("num_of_interventions", 0)):
        spec = config.get(f"intervention {idx}", {})
        specs.append({"index": idx, "type": spec.get("type", "unknown")})
    return specs


def find_active_intervention_spans(hdf5_path, eps):
    with h5py.File(hdf5_path, "r") as f:
        vector = f["joint_action/vector"][()]
    moving = np.where(np.linalg.norm(np.diff(vector, axis=0), axis=1) > eps)[0]
    # diff i means frame i -> i + 1 changes, so the active frames start at i + 1.
    return [(start + 1, end + 2) for start, end in group_consecutive(moving)]


def ranges_from_spans(num_frames, spans, specs):
    if not spans:
        raise ValueError("No active intervention spans were detected")
    if len(spans) != len(specs):
        print(
            f"warning: detected {len(spans)} active spans for {len(specs)} configured interventions; "
            "ranges will be labeled by detected order"
        )

    ranges = []
    first_start = spans[0][0]
    if first_start > 0:
        first_idx = specs[0]["index"] if specs else 0
        ranges.append((f"before_intervention_{first_idx}", 0, first_start))

    for order, ((_, prev_end), (next_start, _)) in enumerate(zip(spans, spans[1:])):
        if prev_end < next_start:
            prev_idx = specs[order]["index"] if order < len(specs) else order
            next_idx = specs[order + 1]["index"] if order + 1 < len(specs) else order + 1
            ranges.append((f"between_intervention_{prev_idx}_{next_idx}", prev_end, next_start))

    last_end = spans[-1][1]
    if last_end < num_frames:
        last_order = len(spans) - 1
        last_idx = specs[last_order]["index"] if last_order < len(specs) else last_order
        ranges.append((f"after_intervention_{last_idx}", last_end, num_frames))

    return ranges


def range_from_interventions(spans, start_idx, end_idx):
    if start_idx > end_idx:
        raise ValueError("--intervention-range START END must satisfy START <= END")
    if end_idx >= len(spans):
        raise ValueError(
            f"Requested intervention {end_idx}, but only {len(spans)} active spans were detected"
        )
    return f"intervention_{start_idx}_{end_idx}", spans[start_idx][0], spans[end_idx][1]


def range_from_frames(num_frames, start, end):
    if not (0 <= start < end <= num_frames):
        raise ValueError(f"Invalid frame range [{start}, {end}) for {num_frames} frames")
    return f"frames_{start}_{end}", start, end


def parse_manual_ranges(values):
    ranges = []
    for value in values:
        try:
            name, start, end = value.split(":")
        except ValueError as exc:
            raise ValueError(f"Bad range '{value}', expected name:start:end") from exc
        ranges.append((name, int(start), int(end)))
    return ranges


def copy_attrs(src, dst):
    for key, value in src.attrs.items():
        dst.attrs[key] = value


def copy_group(src_group, dst_group, frame_slice, num_frames):
    copy_attrs(src_group, dst_group)
    for key, item in src_group.items():
        if isinstance(item, h5py.Group):
            child = dst_group.create_group(key)
            copy_group(item, child, frame_slice, num_frames)
        else:
            data = item[frame_slice] if item.shape and item.shape[0] == num_frames else item[()]
            dst = dst_group.create_dataset(
                key,
                data=data,
                dtype=item.dtype,
                compression=item.compression,
                compression_opts=item.compression_opts,
                shuffle=item.shuffle,
                fletcher32=item.fletcher32,
            )
            copy_attrs(item, dst)


def crop_hdf5(src_path, dst_path, start, end):
    with h5py.File(src_path, "r") as src:
        num_frames = src["joint_action/vector"].shape[0]
        if not (0 <= start < end <= num_frames):
            raise ValueError(f"Invalid crop [{start}, {end}) for {src_path} with {num_frames} frames")

        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        with h5py.File(dst_path, "w") as dst:
            copy_group(src, dst, slice(start, end), num_frames)
            dst.attrs["source_file"] = os.path.abspath(src_path)
            dst.attrs["crop_start"] = start
            dst.attrs["crop_end"] = end


def crop_video(src_path, dst_path, start, end):
    if not os.path.exists(src_path):
        return False

    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=r_frame_rate",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            src_path,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rate = probe.stdout.strip()
    if "/" in rate:
        num, den = rate.split("/")
        fps = float(num) / float(den)
    else:
        fps = float(rate)

    start_sec = start / fps
    duration_sec = (end - start) / fps
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-ss",
            f"{start_sec:.9f}",
            "-i",
            src_path,
            "-t",
            f"{duration_sec:.9f}",
            "-an",
            "-map",
            "0:v:0",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            dst_path,
        ],
        check=True,
    )
    return True


def crop_traj_data(src_path, dst_path, specs, start_idx, end_idx):
    if not os.path.exists(src_path):
        return False

    with open(src_path, "rb") as f:
        traj_data = pickle.load(f)

    left_paths = traj_data.get("left_joint_path", [])
    right_paths = traj_data.get("right_joint_path", [])
    kept = {"left_joint_path": [], "right_joint_path": []}

    left_cursor = 0
    right_cursor = 0
    for spec in specs:
        idx = spec["index"]
        path_count = PLANNED_INTERVENTION_PATH_COUNTS.get(spec["type"], 0)
        for _ in range(path_count):
            if left_cursor < len(left_paths):
                if start_idx <= idx <= end_idx:
                    kept["left_joint_path"].append(left_paths[left_cursor])
                left_cursor += 1
            elif right_cursor < len(right_paths):
                if start_idx <= idx <= end_idx:
                    kept["right_joint_path"].append(right_paths[right_cursor])
                right_cursor += 1

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with open(dst_path, "wb") as f:
        pickle.dump(kept, f)
    return {
        "left_joint_path": len(kept["left_joint_path"]),
        "right_joint_path": len(kept["right_joint_path"]),
    }


def copy_episode_instruction(run_dir, output_dir, episode_name):
    src = os.path.join(run_dir, "instructions", f"{episode_name}.json")
    if not os.path.exists(src):
        return False
    dst = os.path.join(output_dir, "instructions", f"{episode_name}.json")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    return True


def copy_run_sidecars(run_dir, output_dir):
    for name in ("seed.txt",):
        src = os.path.join(run_dir, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(output_dir, name))

    src_scene = os.path.join(run_dir, "scene_info.json")
    if os.path.exists(src_scene):
        shutil.copy2(src_scene, os.path.join(output_dir, "scene_info.json"))


def find_episode_files(run_dir):
    data_dir = os.path.join(run_dir, "data")
    files = []
    for name in os.listdir(data_dir):
        if name.startswith("episode") and name.endswith(".hdf5"):
            stem = name[len("episode") : -len(".hdf5")]
            if stem.isdigit():
                files.append((int(stem), os.path.join(data_dir, name)))
    if not files:
        raise FileNotFoundError(f"No episode*.hdf5 files found in {data_dir}")
    return [path for _, path in sorted(files)]


def main():
    parser = argparse.ArgumentParser(
        description="Crop dummy_task/multiple_interventions output data by intervention or frame range."
    )
    parser.add_argument("--task-name", default="dummy_task")
    parser.add_argument("--task-config", default="multiple_interventions")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--eps", type=float, default=1e-4)
    parser.add_argument(
        "--intervention-range",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        default=None,
        help="Keep data from START intervention start through END intervention end, inclusive.",
    )
    parser.add_argument(
        "--frame-range",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        default=None,
        help="Keep explicit frame range [START, END).",
    )
    parser.add_argument(
        "--ranges",
        nargs="*",
        default=None,
        help="Legacy manual ranges as name:start:end. Usually prefer --intervention-range or --frame-range.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = load_config(args.task_config)
    assert args.task_name == "dummy_task", "This post-process script is scoped to dummy_task"
    assert args.task_config == "multiple_interventions", "This script is scoped to multiple_interventions"
    assert config.get("num_of_interventions", 0) > 1, "Config must contain multiple interventions"
    specs = intervention_specs(config)

    run_dir = args.run_dir or resolve_default_run_dir(args.task_name, args.task_config)
    if sum(x is not None for x in (args.intervention_range, args.frame_range, args.ranges)) > 1:
        raise ValueError("Use only one of --intervention-range, --frame-range, or --ranges")

    output_root = args.output_dir or os.path.join(run_dir, "crop_intervention")
    prepared_outputs = set()

    metadata = {
        "task_name": args.task_name,
        "task_config": args.task_config,
        "run_dir": os.path.abspath(run_dir),
        "episodes": {},
    }

    for episode_path in find_episode_files(run_dir):
        episode_name = os.path.splitext(os.path.basename(episode_path))[0]
        with h5py.File(episode_path, "r") as f:
            num_frames = f["joint_action/vector"].shape[0]

        spans = find_active_intervention_spans(episode_path, args.eps)
        if args.intervention_range is not None:
            range_name, start, end = range_from_interventions(spans, *args.intervention_range)
            ranges = [(range_name, start, end)]
        elif args.frame_range is not None:
            range_name, start, end = range_from_frames(num_frames, *args.frame_range)
            ranges = [(range_name, start, end)]
        elif args.ranges is None:
            ranges = ranges_from_spans(num_frames, spans, specs)
        else:
            ranges = parse_manual_ranges(args.ranges)

        metadata["episodes"][episode_name] = {
            "num_frames": num_frames,
            "configured_interventions": specs,
            "detected_active_spans": spans,
            "ranges": [],
        }

        for range_name, start, end in ranges:
            output_dir = os.path.join(output_root, range_name)
            if output_dir not in prepared_outputs:
                if os.path.exists(output_dir):
                    if not args.overwrite:
                        raise FileExistsError(f"{output_dir} already exists; pass --overwrite to replace it")
                    shutil.rmtree(output_dir)
                os.makedirs(output_dir, exist_ok=True)
                copy_run_sidecars(run_dir, output_dir)
                prepared_outputs.add(output_dir)

            hdf5_dst = os.path.join(output_dir, "data", f"{episode_name}.hdf5")
            video_src = os.path.join(run_dir, "video", f"{episode_name}.mp4")
            video_dst = os.path.join(output_dir, "video", f"{episode_name}.mp4")
            traj_src = os.path.join(run_dir, "_traj_data", f"{episode_name}.pkl")
            traj_dst = os.path.join(output_dir, "_traj_data", f"{episode_name}.pkl")

            crop_hdf5(episode_path, hdf5_dst, start, end)
            video_written = crop_video(video_src, video_dst, start, end)
            instruction_copied = copy_episode_instruction(run_dir, output_dir, episode_name)

            if args.intervention_range is not None:
                traj_written = crop_traj_data(traj_src, traj_dst, specs, *args.intervention_range)
            else:
                traj_written = False

            metadata["episodes"][episode_name]["ranges"].append(
                {
                    "name": range_name,
                    "start": start,
                    "end": end,
                    "frames": end - start,
                    "hdf5": hdf5_dst,
                    "video_written": video_written,
                    "traj_data_written": traj_written,
                    "instruction_copied": instruction_copied,
                }
            )
            print(f"{episode_name}: {range_name} [{start}, {end}) -> {output_dir}")

    os.makedirs(output_root, exist_ok=True)
    with open(os.path.join(output_root, "crop_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    main()
