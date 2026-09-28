#!/usr/bin/env python3
"""Regression checks for benchmark claims, accounting, integrity, and native traces."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'benchmarks'))
from run_report import make_plan
from summarize_report import aggregate, qualifies, read_rows


def config():
    return dict(vectors=[1000], rates=[100, 200], modes=['cpu_latency', 'gpu_batch'], repetitions=3,
                duration_seconds=30, phase_seconds=10, cuda='on', order_seed=42, slo_ms=15,
                min_samples=1000, max_rejection_fraction=.001, max_violation_fraction=.01)


def measurement(rate=100, repeat=1):
    return dict(vectors=1000, scenario='steady', rate=rate, mode='cpu_latency', repeat=repeat,
                completed=3000, offered=3000, failed=0, controller_errors=0,
                offered_p99_ms=10, p95_ms=5, rejection_fraction=0, violation_fraction=0,
                elapsed_seconds=30, queries_per_second=100, max_producer_lag_ms=.1, checksum='123')


class StatisticsTests(unittest.TestCase):
    def test_latency_alone_cannot_qualify_capacity(self):
        row = measurement()
        self.assertTrue(qualifies(row, config(), 30))
        for key, value in [('rejection_fraction', .01), ('violation_fraction', .02), ('completed', 50),
                           ('failed', 1), ('controller_errors', 1), ('offered_p99_ms', None),
                           ('offered_p99_ms', 16), ('elapsed_seconds', 32)]:
            with self.subTest(key=key, value=value):
                bad = dict(row, **{key: value})
                self.assertFalse(qualifies(bad, config(), 30))

    def test_capacity_needs_all_trials_and_full_rate_grid(self):
        settings = config()
        settings['modes'] = ['cpu_latency']
        manifest = dict(config=settings, plan=make_plan(settings), utc='test', label='fixture',
                        git_revision='fixture', identity={}, cpu='fixture', gpu=None)
        rows = [measurement(rate, repeat) for rate in [100, 200] for repeat in [1, 2, 3]]
        result = aggregate(manifest, rows)
        self.assertEqual(result['capacity'][0]['qps'], 200)
        self.assertEqual(result['capacity'][0]['status'], 'highest_tested_rate_passed')
        rows[-1]['offered_p99_ms'] = 50
        result = aggregate(manifest, rows)
        self.assertEqual(result['capacity'][0]['qps'], 100)
        self.assertEqual(result['groups'][1]['metrics']['offered_p99_ms']['max'], 50)
        self.assertEqual(result['groups'][1]['metrics']['offered_p99_ms']['median'], 10)
        self.assertIsNone(aggregate(manifest, rows[:-1])['capacity'][0]['qps'])
        self.assertEqual(aggregate(manifest, rows[:3])['capacity'][0]['status'], 'incomplete')

    def test_checksum_mismatch_and_duplicate_trial_fail(self):
        settings = config()
        manifest = dict(config=settings, plan=make_plan(settings), utc='test', label='fixture',
                        git_revision='fixture', identity={}, cpu='fixture', gpu=None)
        a, b = measurement(), measurement(repeat=2)
        b['checksum'] = '456'
        with self.assertRaisesRegex(ValueError, 'checksums'):
            aggregate(manifest, [a, b])
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            aggregate(manifest, [a, a])

    def test_random_order_preserves_balanced_repetitions(self):
        settings = config()
        plan = make_plan(settings)
        self.assertEqual(plan, make_plan(settings))
        signatures = []
        for repeat in range(1, 4):
            signatures.append([(c['vectors'], c['scenario'], c['rate'], c['mode']) for c in plan if c['repeat'] == repeat])
        self.assertCountEqual(signatures[0], signatures[1])
        self.assertNotEqual(signatures[0], signatures[1])
        self.assertTrue(any(p[5] == 1 for c in plan for p in c['phases']))
        settings['cuda'] = 'off'
        self.assertFalse(any(p[5] for c in make_plan(settings) for p in c['phases']))
        settings['rates'] = [1000000]
        with self.assertRaisesRegex(ValueError, 'two million'):
            make_plan(settings)


def integration(binary, gpu):
    with tempfile.TemporaryDirectory(prefix='flowstate-report-') as temporary:
        output = Path(temporary) / 'report'
        if gpu:
            probe = Path(temporary) / 'probe.csv'
            probe.write_text('name,duration_ms,requests_per_second,burst,top_k\nprobe,1,1000,1,1\n')
            result = subprocess.run([str(binary.resolve()), '--trace', str(probe), '--vectors', '64',
                                     '--dimension', '9', '--cuda', 'on'], capture_output=True, timeout=15)
            if result.returncode and b'CUDA unavailable' in result.stderr:
                print('CUDA unavailable: skipping native contention integration')
                sys.exit(77)
            result.check_returncode()
        command = [sys.executable, str(ROOT / 'benchmarks/run_report.py'), '--binary', str(binary.resolve()),
                   '--output', str(output), '--vectors', '64', '--rates', '300', '--dimension', '9',
                   '--modes', 'gpu_batch' if gpu else 'cpu_latency', '--cuda', 'on' if gpu else 'off',
                   '--duration-seconds', '1', '--phase-seconds', '1', '--repetitions', '3', '--no-plots']
        subprocess.run(command, check=True, timeout=80)
        manifest = json.loads((output / 'manifest.json').read_text())
        rows = read_rows(output, manifest)
        assert len(rows) == 6
        for row in rows:
            assert row['completed'] + row['rejected'] + row['failed'] == row['offered']
            assert row['failed'] == 0
        if gpu:
            assert all(r['contention_kernels'] > 0 for r in rows if r['scenario'] == 'burst')
        before = (output / 'run-0001/record.json').read_bytes()
        subprocess.run(command + ['--resume'], check=True, timeout=10)
        assert (output / 'run-0001/record.json').read_bytes() == before
        # A resume with different parameters must never mix old and new measurements.
        mismatch = subprocess.run(command + ['--resume', '--seed', '43'], capture_output=True, timeout=10)
        assert mismatch.returncode != 0 and b'Resume requires identical' in mismatch.stderr
        native = output / 'run-0001/summary.csv'
        native.write_text(native.read_text() + '\n')
        try:
            read_rows(output, manifest)
        except ValueError as error:
            assert 'artifact has changed' in str(error)
        else:
            raise AssertionError('Modified measurements must not be accepted')
        if not gpu:
            trace = Path(temporary) / 'contention.csv'
            trace.write_text('name,duration_ms,requests_per_second,burst,top_k,gpu_contention\nload,1000,10,1,1,1\n')
            result = subprocess.run([str(binary.resolve()), '--trace', str(trace), '--cuda', 'off'], capture_output=True, timeout=10)
            assert result.returncode != 0 and b'contention requires CUDA' in result.stderr
    print('Native benchmark/report integration passed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary', type=Path)
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()
    if args.binary:
        integration(args.binary, args.gpu)
    else:
        unittest.main(argv=[sys.argv[0]])
