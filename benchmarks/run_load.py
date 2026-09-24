#!/usr/bin/env python3
"""Run the same workload trace and seed through static and adaptive policies."""
import argparse
import csv
import hashlib
import io
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from run_matrix import capture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, default=Path('build/flowstate_load'))
    parser.add_argument('--trace', type=Path, default=Path('benchmarks/traces/adaptive.csv'))
    parser.add_argument('--output', type=Path, default=Path('benchmark-output/load'))
    parser.add_argument('--vectors', type=int, default=100000)
    parser.add_argument('--dimension', type=int, default=384)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--queue-capacity', type=int, default=1024)
    parser.add_argument('--cuda', choices=['auto', 'on', 'off'], default='auto')
    parser.add_argument('--cpu-backend', choices=['auto', 'scalar', 'avx2'], default='auto')
    parser.add_argument('--modes', nargs='+', choices=['cpu_latency', 'gpu_immediate', 'gpu_batch', 'balanced', 'heuristic'],
                        default=['cpu_latency', 'gpu_immediate', 'gpu_batch', 'heuristic'])
    parser.add_argument('--label', default='local')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    binary = args.binary.resolve()
    metadata = {
        'utc': datetime.now(timezone.utc).isoformat(), 'host': platform.platform(),
        'cpu': capture(['lscpu']) if platform.system() == 'Linux' else capture(['sysctl', '-n', 'machdep.cpu.brand_string']),
        'gpu': capture(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader']),
        'nvcc': capture(['nvcc', '--version']),
        'git_revision': capture(['git', 'rev-parse', 'HEAD']), 'git_status': capture(['git', 'status', '--short']),
        'arguments': {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        'trace': args.trace.read_text(), 'trace_sha256': hashlib.sha256(args.trace.read_bytes()).hexdigest(),
        'commands': [],
    }
    compile_commands = binary.parent / 'compile_commands.json'
    if compile_commands.exists():
        metadata['compile_commands'] = json.loads(compile_commands.read_text())
    with (args.output / 'summary.csv').open('w', newline='') as output:
        writer = None
        for mode in args.modes:
            command = [str(binary), '--trace', str(args.trace), '--mode', mode, '--vectors', str(args.vectors),
                       '--dimension', str(args.dimension), '--seed', str(args.seed), '--workers', str(args.workers),
                       '--queue-capacity', str(args.queue_capacity), '--cuda', args.cuda, '--cpu-backend', args.cpu_backend,
                       '--telemetry', str(args.output / f'{mode}.csv')]
            metadata['commands'].append(command)
            (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
            print(f'Running {mode}', flush=True)
            result = subprocess.run(command, check=True, text=True, stdout=subprocess.PIPE)
            rows = csv.DictReader(io.StringIO(result.stdout))
            if writer is None:
                writer = csv.DictWriter(output, fieldnames=rows.fieldnames)
                writer.writeheader()
            writer.writerows(rows)
            output.flush()
    print(args.output / 'summary.csv')


if __name__ == '__main__':
    main()
