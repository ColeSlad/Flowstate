"""Aggregate independent trials; keep latency quantiles separate from trial variation."""
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

METRICS = ['queries_per_second', 'offered_p99_ms', 'p95_ms', 'rejection_fraction',
           'violation_fraction', 'max_producer_lag_ms', 'elapsed_seconds', 'completed']


def distribution(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return {'median': statistics.median(values), 'min': min(values), 'max': max(values),
            'stdev': statistics.stdev(values) if len(values) > 1 else None}


def qualifies(row, config, duration):
    """Conservative finite-duration pass, never an extrapolated maximum capacity."""
    return (row['completed'] >= config['min_samples'] and row['failed'] == 0 and
            row['controller_errors'] == 0 and row['offered_p99_ms'] is not None and
            row['offered_p99_ms'] <= config['slo_ms'] and
            row['rejection_fraction'] <= config['max_rejection_fraction'] and
            row['violation_fraction'] <= config['max_violation_fraction'] and
            row['elapsed_seconds'] <= duration * 1.05)


def aggregate(manifest, rows):
    config = manifest['config']
    grouped = defaultdict(list)
    checksums = defaultdict(set)
    for row in rows:
        key = tuple(row[k] for k in ['vectors', 'scenario', 'rate', 'mode'])
        grouped[key].append(row)
        if row['completed'] == row['offered']:
            checksums[key[:3]].add(row['checksum'])
    if any(len(values) > 1 for values in checksums.values()):
        raise ValueError('Result checksums differ across fully completed equivalent workloads')
    groups = []
    for (vectors, scenario, rate, mode), trials in sorted(grouped.items()):
        if len({r['repeat'] for r in trials}) != len(trials):
            raise ValueError('Duplicate trial in a benchmark group')
        passes = sum(qualifies(r, config, config['duration_seconds']) for r in trials) if scenario == 'steady' else None
        complete = {r['repeat'] for r in trials} == set(range(1, config['repetitions'] + 1))
        groups.append({'vectors': vectors, 'scenario': scenario, 'rate': rate, 'mode': mode,
                       'trials': len(trials), 'complete': complete, 'passing_trials': passes,
                       'all_pass': scenario == 'steady' and complete and config['repetitions'] >= 3 and passes == len(trials),
                       'metrics': {k: distribution([r[k] for r in trials]) for k in METRICS}})
    capacity = []
    for vectors in config['vectors']:
        for mode in config['modes']:
            cells = [g for g in groups if g['vectors'] == vectors and g['mode'] == mode and g['scenario'] == 'steady']
            expected = [c for c in manifest['plan'] if c['vectors'] == vectors and c['mode'] == mode and c['scenario'] == 'steady']
            complete = len(expected) == sum(g['trials'] for g in cells) and all(g['complete'] for g in cells)
            passing = [g['rate'] for g in cells if g['all_pass']]
            best = max(passing) if passing and complete else None
            tested_max = max((c['rate'] for c in expected), default=None)
            status = ('incomplete' if not complete else 'insufficient_repetitions' if config['repetitions'] < 3 else
                      'no_passing_rate' if best is None else 'highest_tested_rate_passed' if best == tested_max else 'measured')
            capacity.append({'vectors': vectors, 'mode': mode, 'qps': best, 'status': status})
    return {'schema': 1, 'utc': manifest['utc'], 'label': manifest['label'], 'git_revision': manifest['git_revision'],
            'identity': manifest['identity'], 'config': config, 'cpu': manifest['cpu'], 'gpu': manifest['gpu'],
            'cpu_backends': sorted({r.get('cpu_backend', 'unspecified') for r in rows}),
            'completed_trials': len(rows), 'planned_trials': len(manifest['plan']), 'groups': groups, 'capacity': capacity}


def read_rows(directory, manifest):
    rows = []
    for case in manifest['plan']:
        path = directory / case['id']
        if not (path / 'record.json').exists():
            continue
        record = json.loads((path / 'record.json').read_text())
        if record['case'] != case:
            raise ValueError(f'Trial does not match manifest: {path}')
        for name, expected in record['sha256'].items():
            with (path / name).open('rb') as source:
                if hashlib.file_digest(source, 'sha256').hexdigest() != expected:
                    raise ValueError(f'Trial artifact has changed: {path / name}')
        with (path / 'summary.csv').open() as source:
            native = list(csv.DictReader(source))
        if len(native) != 1:
            raise ValueError(f'Expected one summary row: {path}')
        row = {}
        for key, value in native[0].items():
            if key in {'mode', 'cpu_backend'}:
                row[key] = value
            else:
                row[key] = int(value) if value.isdigit() else float(value) if value else None
                if row[key] is not None and not math.isfinite(row[key]):
                    raise ValueError(f'Non-finite measurement: {path}/{key}')
        # Preserve the exact integer checksum; it need not fit a float mantissa.
        row['checksum'] = native[0]['checksum']
        row.update({k: case[k] for k in ['id', 'repeat', 'scenario', 'rate']})
        row['rejection_fraction'] = row['rejected'] / row['offered']
        row['violation_fraction'] = row['slo_violations'] / row['offered']
        rows.append(row)
    return rows


def fmt(value):
    return '—' if value is None else f'{value:,.2f}'


def render_report(directory, plots=True):
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    rows = read_rows(directory, manifest)
    summary = aggregate(manifest, rows)
    config = manifest['config']
    cpu = manifest['cpu'] or 'unavailable'
    cpu = next((line.split(':', 1)[1].strip() for line in cpu.splitlines()
                if line.startswith('Model name:')), cpu)
    (directory / 'report.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    if rows:
        with (directory / 'trials.csv').open('w', newline='') as output:
            writer = csv.DictWriter(output, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines = ['# Repeated benchmark report', '', f'{manifest["label"]} · started {manifest["utc"]}', '',
             f'Completed **{len(rows)} / {len(manifest["plan"])} trials**. Source commit: `{manifest["git_revision"]}`.', '',
             f'CPU: {cpu}', '',
             f'CPU search backend(s): {", ".join(summary["cpu_backends"]) or "no completed trials"}', '',
             f'GPU / driver / VRAM: {manifest["gpu"] or "unavailable"}', '',
             f'{config["dimension"]} dimensions, float32, top-K 10, seed {config["seed"]}; '
             f'{config["workers"]} CPU workers, queue {config["queue_capacity"]}, batches ≤{config["batch_size"]}, '
             f'wait ≤{config["max_wait_us"]} µs; {config["repetitions"]} independent process trials per case.', '',
             '## Method', '',
             'Each repetition visits every case in a seeded shuffled order. Three batches warm each native backend '
             'before measurement; dataset creation and persistent GPU upload are excluded. The runtime/controller '
             'start with empty queues. The same 32 deterministic queries cycle through each trace.', '',
             f'Steady trials offer traffic for {config["duration_seconds"]} seconds; elapsed time and throughput include final drain. '
             'Successful-request offered latency includes producer lateness, queueing, and execution. '
             'Rejections/failures also count as SLO violations. End-to-end summaries use exact nearest-rank quantiles.', '',
             'Tables show medians across trials with full observed min–max ranges, not confidence intervals. '
             'No pooled p99 or maximum-capacity extrapolation is reported. Desktop activity, WSL scheduling, '
             'temperature and clock speeds are uncontrolled. These are finite-duration tests, not production guarantees.', '',
             '## Highest tested rate meeting the criteria', '',
             f'Every trial at a rate must have offered p99 ≤{config["slo_ms"]} ms, '
             f'rejections ≤{config["max_rejection_fraction"]:.2%}, total SLO violations ≤{config["max_violation_fraction"]:.2%}, '
             f'at least {config["min_samples"]:,} successful samples, zero search/controller errors, and elapsed time '
             '≤105% of the traffic duration. At least three repetitions and a complete rate grid are required. '
             '“No passing rate” does not mean zero capacity. A passing highest grid point means the limit was not reached.', '',
             '| Vectors | Policy | Highest passing offered QPS | Status |', '| ---: | --- | ---: | --- |']
    for cell in summary['capacity']:
        lines.append(f'| {cell["vectors"]:,} | {cell["mode"]} | {fmt(cell["qps"])} | {cell["status"].replace("_", " ")} |')
    lines += ['', '## All measured cases', '',
              'Latency is offered p99 of successful requests. Brackets show the range across trials. '
              'Rejection and violation percentages are medians. Burst rows cover the entire trace, including recovery.', '',
              '| Vectors | Scenario | Offered QPS¹ | Policy | Trials | Completed QPS | Offered p99 ms | Rejected % | SLO violations % | Passes² |',
              '| ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for group in summary['groups']:
        metrics = group['metrics']
        def interval(key):
            value = metrics[key]
            return '—' if value is None else f'{fmt(value["median"])} [{fmt(value["min"])}–{fmt(value["max"])}]'
        lines.append(f'| {group["vectors"]:,} | {group["scenario"]} | {group["rate"]:,} | {group["mode"]} | '
                     f'{group["trials"]} | {interval("queries_per_second")} | {interval("offered_p99_ms")} | '
                     f'{100 * metrics["rejection_fraction"]["median"]:.2f} | {100 * metrics["violation_fraction"]["median"]:.2f} | '
                     f'{group["passing_trials"] if group["passing_trials"] is not None else "—"} |')
    lines += ['', '¹ Burst rows list peak offered rate. ² Passing trials apply only to steady traffic; the full capacity criteria above still apply.', '',
              '## Figures', '', '![Highest tested passing rate](capacity.png)', '',
              '![Latency versus offered load](latency.png)', '', '![First-trial burst behavior](burst.png)', '',
              'Burst figure always uses repetition 1 for each policy at the default dataset if present, otherwise the middle size; '
              'it is not chosen for favorable behavior. Rolling p99 in that figure is a histogram upper bound, '
              'unlike the exact summary quantiles. CUDA traces use the same competing kernel as the dashboard; '
              'CPU-only traces omit the contention phase.', '',
              'See `manifest.json` for compiler commands, binary/source hashes, complete randomized plan, and configuration; '
              '`trials.csv` for individual summaries; `report.json` for machine-readable aggregates and standard deviations. '
              'Each trial retains its exact trace, command, stderr, telemetry, timestamps, and artifact hashes.', '']
    if not plots:
        lines[lines.index('## Figures')] = '## Figures (run --render-only without --no-plots to generate)'
    (directory / 'report.md').write_text('\n'.join(lines))
    if plots and rows:
        from plot_report import plot_report
        plot_report(directory, manifest, summary)
    print(f'Report: {directory / "report.md"}', flush=True)
    return summary
