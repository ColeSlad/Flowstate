#!/usr/bin/env python3
"""Repeated, randomized load sweeps with resumable artifacts and a measured report."""
import argparse
import csv
import hashlib
import io
import json
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from run_matrix import capture
from summarize_report import render_report

ROOT = Path(__file__).resolve().parents[1]
RATES = {1000: [1000, 10000, 30000, 60000],
         100000: [100, 500, 1000, 1500],
         1000000: [10, 50, 100, 150]}
MODES = ['cpu_latency', 'gpu_immediate', 'gpu_batch', 'heuristic']


def digest(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def source_digest():
    """Include uncommitted/new source, without reading ignored data or secrets."""
    paths = [ROOT / 'CMakeLists.txt']
    for directory in ['src', 'include', 'benchmarks']:
        paths += [p for p in (ROOT / directory).rglob('*')
                  if p.suffix in {'.cpp', '.cu', '.hpp', '.py'}]
    value = hashlib.sha256()
    for path in sorted(paths):
        value.update(str(path.relative_to(ROOT)).encode())
        value.update(bytes.fromhex(digest(path)))
    return value.hexdigest()


def make_plan(config):
    cases = []
    for vectors in config['vectors']:
        rates = config['rates'] or RATES[vectors]
        for rate in rates:
            cases.append({'vectors': vectors, 'scenario': 'steady', 'rate': rate,
                          'phases': [('steady', config['duration_seconds'] * 1000, rate, 1, 10, 0)]})
        # Same generated workload for every policy; explicit real CUDA contention.
        phase_ms = config['phase_seconds'] * 1000
        phases = [('quiet', phase_ms, 10, 1, 10, 0),
                  ('burst', phase_ms, max(rates), 32, 10, 0)]
        if config['cuda'] == 'on':
            phases.append(('contention', phase_ms, max(rates), 32, 10, 1))
        phases.append(('recovery', phase_ms, 10, 1, 10, 0))
        cases.append({'vectors': vectors, 'scenario': 'burst', 'rate': max(rates), 'phases': phases})
    for case in cases:
        if sum(p[1] * p[2] // 1000 for p in case['phases']) > 2000000:
            raise ValueError('A trace exceeds two million requests; lower its rate or duration')
    rng = random.Random(config['order_seed'])
    plan = []
    for repeat in range(config['repetitions']):
        block = [dict(case, mode=mode, repeat=repeat + 1) for case in cases for mode in config['modes']]
        rng.shuffle(block)
        for case in block:
            case['id'] = f'run-{len(plan) + 1:04d}'
            plan.append(case)
    # JSON-normalize tuples for equality checks on resume.
    return json.loads(json.dumps(plan))


def trace_text(case):
    output = io.StringIO()
    writer = csv.writer(output, lineterminator='\n')
    writer.writerow(['name', 'duration_ms', 'requests_per_second', 'burst', 'top_k', 'gpu_contention'])
    writer.writerows(case['phases'])
    return output.getvalue()


def check_summary(text, case):
    rows = list(csv.DictReader(io.StringIO(text)))
    if len(rows) != 1:
        raise ValueError('Expected exactly one native summary row')
    row = rows[0]
    expected = sum(p[1] * p[2] // 1000 for p in case['phases'])
    if (int(row['offered']) != expected or
            sum(int(row[k]) for k in ['completed', 'rejected', 'failed']) != expected or
            row['mode'] != case['mode'] or int(row['vectors']) != case['vectors']):
        raise ValueError('Native summary accounting/configuration mismatch')
    if int(row['failed']) or int(row['controller_errors']):
        raise ValueError('Search or controller failures invalidate this benchmark')
    if any(p[5] for p in case['phases']) and int(row['contention_kernels']) == 0:
        raise ValueError('Contention trace executed no competing CUDA kernels')
    return row


def run(args, config):
    binary = args.binary.resolve()
    plan = make_plan(config)
    identity = {'binary_sha256': digest(binary), 'source_sha256': source_digest(),
                'host': platform.node(), 'platform': platform.platform()}
    manifest_path = args.output / 'manifest.json'
    if args.resume:
        manifest = json.loads(manifest_path.read_text())
        if manifest['config'] != config or manifest['identity'] != identity or manifest['plan'] != plan:
            raise ValueError('Resume requires identical settings, host, binary, and source; use a new output directory')
    else:
        if args.output.exists() and any(args.output.iterdir()):
            raise ValueError('Output is not empty; use --resume or a new output directory')
        args.output.mkdir(parents=True, exist_ok=True)
        compile_commands = binary.parent / 'compile_commands.json'
        manifest = {
            'schema': 1, 'utc': datetime.now(timezone.utc).isoformat(), 'identity': identity,
            'config': config, 'plan': plan, 'binary': str(binary), 'label': args.label,
            'git_revision': capture(['git', '-C', str(ROOT), 'rev-parse', 'HEAD']),
            'git_status': capture(['git', '-C', str(ROOT), 'status', '--short']),
            'cpu': capture(['lscpu']) if platform.system() == 'Linux' else capture(['sysctl', '-n', 'machdep.cpu.brand_string']),
            'gpu': capture(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader']),
            'nvcc': capture(['nvcc', '--version']), 'python': sys.version,
            'compile_commands': json.loads(compile_commands.read_text()) if compile_commands.exists() else None,
        }
        write_json(manifest_path, manifest)
    minutes = sum(sum(p[1] for p in c['phases']) for c in plan) / 60000
    print(f'{len(plan)} trials; {minutes:.1f} minutes of scheduled traffic, plus setup/drain. Artifacts: {args.output}', flush=True)
    for position, case in enumerate(plan, 1):
        if digest(binary) != identity['binary_sha256']:
            raise ValueError('Benchmark binary changed during the run; use a new output directory')
        directory = args.output / case['id']
        directory.mkdir(exist_ok=True)
        record_path = directory / 'record.json'
        if record_path.exists():
            record = json.loads(record_path.read_text())
            if record['case'] != case or any(digest(directory / name) != sha for name, sha in record['sha256'].items()):
                raise ValueError(f'Completed trial was modified: {directory}')
            check_summary((directory / 'summary.csv').read_text(), case)
            continue
        (directory / 'trace.csv').write_text(trace_text(case))
        command = [str(binary), '--trace', str((directory / 'trace.csv').resolve()),
                   '--mode', case['mode'], '--vectors', str(case['vectors']), '--dimension', str(config['dimension']),
                   '--seed', str(config['seed']), '--workers', str(config['workers']),
                   '--queue-capacity', str(config['queue_capacity']), '--batch-size', str(config['batch_size']),
                   '--max-wait-us', str(config['max_wait_us']), '--interval-ms', '500',
                   '--cuda', config['cuda'], '--cpu-backend', config['cpu_backend'],
                   '--slo-ms', str(config['slo_ms']), '--telemetry', str((directory / 'telemetry.csv').resolve())]
        print(f'[{position}/{len(plan)}] trial {case["repeat"]}: {case["vectors"]} vectors / {case["scenario"]} / {case["rate"]} QPS / {case["mode"]}', flush=True)
        write_json(directory / 'command.json', command)
        started = datetime.now(timezone.utc).isoformat()
        # Files survive interruption; only a validated record marks a completed trial.
        with (directory / 'summary.csv').open('w') as out, (directory / 'stderr.txt').open('w') as err:
            result = subprocess.run(command, stdout=out, stderr=err, timeout=args.timeout_seconds)
        if result.returncode:
            raise RuntimeError(f'Trial failed ({result.returncode}): {directory / "stderr.txt"}')
        check_summary((directory / 'summary.csv').read_text(), case)
        write_json(record_path, {'case': case, 'started': started, 'finished': datetime.now(timezone.utc).isoformat(),
                                'sha256': {name: digest(directory / name) for name in
                                           ['summary.csv', 'telemetry.csv', 'trace.csv', 'command.json', 'stderr.txt']}})
    render_report(args.output, plots=not args.no_plots)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, default=Path('build-gpu/flowstate_load'))
    parser.add_argument('--output', type=Path, default=Path('benchmark-output/report'))
    parser.add_argument('--vectors', type=int, nargs='+', default=list(RATES))
    parser.add_argument('--rates', type=int, nargs='+', help='Override the per-dataset default rate grids')
    parser.add_argument('--modes', nargs='+', choices=MODES + ['balanced'])
    parser.add_argument('--cuda', choices=['on', 'off'], default='on')
    parser.add_argument('--cpu-backend', choices=['auto', 'scalar', 'avx2'], default='auto')
    parser.add_argument('--dimension', type=int, default=384)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--order-seed', type=int, default=20260928)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--queue-capacity', type=int, default=1024)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--max-wait-us', type=int, default=2000)
    parser.add_argument('--repetitions', type=int, default=5)
    parser.add_argument('--duration-seconds', type=int, default=30)
    parser.add_argument('--phase-seconds', type=int, default=10)
    parser.add_argument('--slo-ms', type=int, default=15)
    parser.add_argument('--max-rejection-fraction', type=float, default=.001)
    parser.add_argument('--max-violation-fraction', type=float, default=.01)
    parser.add_argument('--min-samples', type=int, default=1000)
    parser.add_argument('--timeout-seconds', type=int, default=900)
    parser.add_argument('--label', default='local')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--render-only', action='store_true', help='Regenerate tables/figures from recorded trials; no benchmark execution')
    parser.add_argument('--no-plots', action='store_true', help='Skip Matplotlib; render later on another machine')
    args = parser.parse_args()
    if args.render_only:
        render_report(args.output, plots=not args.no_plots)
        return
    config = {k: v for k, v in vars(args).items() if k not in
              {'binary', 'output', 'label', 'resume', 'render_only', 'no_plots', 'timeout_seconds'}}
    config['modes'] = args.modes or (MODES if args.cuda == 'on' else ['cpu_latency', 'heuristic'])
    if args.cuda == 'off' and set(config['modes']) - {'cpu_latency', 'heuristic'}:
        parser.error('GPU policies require --cuda on')
    positive = ['dimension', 'workers', 'queue_capacity', 'batch_size', 'repetitions', 'duration_seconds',
                'phase_seconds', 'slo_ms', 'min_samples', 'timeout_seconds']
    if any(getattr(args, k) <= 0 for k in positive) or args.seed < 0 or args.seed > 2**32 - 1:
        parser.error('Sizes, counts, durations, and SLO must be positive; seed must fit uint32')
    if args.duration_seconds > 600 or args.phase_seconds > 600 or not 0 <= args.max_wait_us <= 60000000:
        parser.error('Phases must be at most 600 seconds; batch wait must be 0..60000000 microseconds')
    if any(v < 10 for v in args.vectors) or (not args.rates and any(v not in RATES for v in args.vectors)):
        parser.error('Custom vector counts (at least 10) require --rates')
    if args.rates and any(rate <= 0 or rate > 1000000 for rate in args.rates):
        parser.error('Rates must be 1..1000000 QPS')
    if not 0 <= args.max_rejection_fraction <= 1 or not 0 <= args.max_violation_fraction <= 1:
        parser.error('Rejection and violation fractions must be 0..1')
    if any(len(values) != len(set(values)) for values in [args.vectors, args.rates or [], config['modes']]):
        parser.error('Duplicate sizes, rates, or modes are not allowed')
    run(args, config)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        sys.exit(f'benchmark report: {error}')
