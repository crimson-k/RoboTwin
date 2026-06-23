import fcntl
import os
import re


RUN_DIR_PATTERN = re.compile(r"^run_(\d+)$")


def normalize_run_id(run_id):
    if run_id is None:
        return None

    run_id = str(run_id)
    if run_id.isdigit():
        return f"run_{int(run_id):04d}"

    match = RUN_DIR_PATTERN.fullmatch(run_id)
    if match:
        return f"run_{int(match.group(1)):04d}"

    raise ValueError("run_id must be an integer or use the format run_NNNN")


def _existing_run_numbers(base_path):
    run_numbers = []
    for entry in os.scandir(base_path):
        if not entry.is_dir():
            continue
        match = RUN_DIR_PATTERN.fullmatch(entry.name)
        if match:
            run_numbers.append(int(match.group(1)))
    return run_numbers


def resolve_run_directory(base_path, requested_run_id=None):
    os.makedirs(base_path, exist_ok=True)
    requested_run_id = normalize_run_id(requested_run_id)

    lock_path = os.path.join(base_path, ".run_id.lock")
    with open(lock_path, "a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        existing_ids = _existing_run_numbers(base_path)

        if requested_run_id is not None:
            run_id = requested_run_id
            run_path = os.path.join(base_path, run_id)
            os.makedirs(run_path, exist_ok=True)
        else:
            next_id = max(existing_ids, default=0) + 1
            run_id = f"run_{next_id:04d}"
            run_path = os.path.join(base_path, run_id)
            os.mkdir(run_path)

        run_count = len(_existing_run_numbers(base_path))
        with open(os.path.join(base_path, "run_count.txt"), "w", encoding="utf-8") as file:
            file.write(f"{run_count}\n")
        with open(os.path.join(base_path, "latest_run.txt"), "w", encoding="utf-8") as file:
            file.write(f"{run_id}\n")

    return run_id, run_path
