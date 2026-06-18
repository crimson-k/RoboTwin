#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create dummy pi05 norm stats assets for RoboTwin eval plumbing."
    )
    parser.add_argument("train_config_name", default="demo_clean")
    parser.add_argument("model_name", default = "adjust_bottle")
    parser.add_argument("--checkpoint-id", default="30000")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--asset-id", default="dummy")
    parser.add_argument("--dim", type=int, default=32)
    args = parser.parse_args()

    output_dir = (
        Path(args.repo_root)
        / "policy"
        / "pi05"
        / "checkpoints"
        / args.train_config_name
        / args.model_name
        / str(args.checkpoint_id)
        / "assets"
        / args.asset_id
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    zeros = [0.0] * args.dim
    ones = [1.0] * args.dim
    neg_ones = [-1.0] * args.dim

    stats = {
        "norm_stats" :{
            "state":{
                "mean": zeros,
                "std": ones,
                "q01": neg_ones,
                "q99": ones,
            },
        "actions":{
                "mean": zeros,
                "std": ones,
                "q01": neg_ones,
                "q99": ones,
            },
        }
    }

    output_path = output_dir / "norm_stats.json"
    output_path.write_text(json.dumps(stats, indent=2))
    print(output_path)

if __name__ == "__main__":
    main()
