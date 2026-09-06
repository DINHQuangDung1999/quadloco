#!/usr/bin/env python3
import argparse
import json
import statistics
from pathlib import Path


METRICS = (
    "success_rate",
    "geometric_success_rate",
    "semantic_accuracy",
    "mean_spl",
    "mean_final_goal_error_m",
    "mean_path_length_m",
    "mean_time_to_success_s",
    "collision_rate",
    "fall_rate",
    "timeout_rate",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate multi-seed PI0.5 evaluations.")
    parser.add_argument("output_base", type=Path)
    args = parser.parse_args()

    result = {"modalities": {}}
    for modality in ("rgb", "rgbd"):
        summaries = sorted(args.output_base.glob(f"{modality}/seed-*/direct/direct_summary.json"))
        runs = []
        for path in summaries:
            data = json.loads(path.read_text())
            runs.append({"seed": int(path.parts[-3].removeprefix("seed-")), **data})

        aggregate = {}
        for metric in METRICS:
            values = [float(run[metric]) for run in runs if run.get(metric) is not None]
            if values:
                aggregate[metric] = {
                    "mean": statistics.fmean(values),
                    "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                    "min": min(values),
                    "max": max(values),
                }
        result["modalities"][modality] = {
            "num_seeds": len(runs),
            "seeds": [run["seed"] for run in runs],
            "total_episodes": sum(run.get("num_episodes", 0) for run in runs),
            "aggregate": aggregate,
            "runs": runs,
        }

    output = args.output_base / "aggregate_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"[INFO] Wrote {output}")


if __name__ == "__main__":
    main()
