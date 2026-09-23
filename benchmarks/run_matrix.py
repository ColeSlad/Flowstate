#!/usr/bin/env python3
"""Reproducible serial backend sweep. Raw outputs stay in ignored benchmark-output/."""
import argparse
import csv
import io
import json
import platform
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone


def capture(command):
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=Path("build/flowstate_bench"))
    parser.add_argument("--output", type=Path, default=Path("benchmark-output/local"))
    parser.add_argument("--vectors", type=int, default=100000)
    parser.add_argument("--dimensions", type=int, nargs="+", default=[128, 384, 768])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--label", default="local")
    parser.add_argument("--require-all", action="store_true", help="Fail unless AVX2 and CUDA are available")
    args = parser.parse_args()
    binary = args.binary.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    cpu = capture(["sysctl", "-n", "machdep.cpu.brand_string"]) if sys.platform == "darwin" else capture(["lscpu"])
    metadata = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.platform(), "host_arch": platform.machine(), "cpu": cpu,
        "gpu": capture(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]),
        "nvcc": capture(["nvcc", "--version"]),
        "binary": str(binary), "label": args.label,
        "git_revision": capture(["git", "rev-parse", "HEAD"]),
        "git_status": capture(["git", "status", "--short"]),
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    compile_commands = binary.parent / "compile_commands.json"
    if compile_commands.exists():
        metadata["compile_commands"] = json.loads(compile_commands.read_text())
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    with (args.output / "results.csv").open("w", newline="") as output:
        writer = None
        for dimension in args.dimensions:
            for batch_size in args.batch_sizes:
                command = [str(binary), "--vectors", str(args.vectors), "--dimension", str(dimension),
                           "--batch-size", str(batch_size), "--iterations", str(args.iterations),
                           "--warmup", str(args.warmup), "--label", args.label]
                if args.require_all:
                    command.append("--require-all")
                print(f"dimension={dimension}, batch_size={batch_size}", file=sys.stderr, flush=True)
                result = subprocess.run(command, text=True, stdout=subprocess.PIPE, check=True)
                rows = csv.DictReader(io.StringIO(result.stdout))
                if writer is None:
                    writer = csv.DictWriter(output, fieldnames=rows.fieldnames)
                    writer.writeheader()
                writer.writerows(rows)
                output.flush()
    print(args.output / "results.csv")


if __name__ == "__main__":
    main()
