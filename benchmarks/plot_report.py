"""Static, exportable figures for the repeated benchmark report."""
import csv
import json
import math
import os
import textwrap

LABELS = {'cpu_latency': 'CPU', 'gpu_immediate': 'GPU immediate', 'gpu_batch': 'GPU batch',
          'heuristic': 'Heuristic', 'balanced': 'Balanced'}
COLORS = {'cpu_latency': '#45556c', 'gpu_immediate': '#007c91', 'gpu_batch': '#166534',
          'heuristic': '#b45309', 'balanced': '#7e22ce'}


def plot_report(directory, manifest, summary):
    os.environ.setdefault('MPLCONFIGDIR', str((directory / '.mplconfig').resolve()))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.titleweight': 'bold', 'figure.facecolor': 'white',
                         'axes.grid': True, 'grid.alpha': .18, 'savefig.dpi': 160})
    config = manifest['config']
    modes = config['modes']
    sizes = sorted(config['vectors'])
    stamp = (f'{manifest["label"]} · {config["repetitions"]} planned trials/case · '
             f'{summary["completed_trials"]}/{summary["planned_trials"]} complete · {config["dimension"]}D · K=10')

    def save(fig, name, bottom=.065):
        fig.text(.01, .01, textwrap.fill(stamp, 130), color='#475569', fontsize=8)
        fig.tight_layout(rect=(0, bottom, 1, .90))
        for extension in ['png', 'svg']:
            fig.savefig(directory / f'{name}.{extension}', bbox_inches='tight')
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    fig.suptitle('Throughput within the latency and rejection limits', fontweight='bold')
    ax.set_title(f'Highest tested passing offered rate · p99 ≤{config["slo_ms"]} ms · every trial must pass', fontsize=10)
    for index, mode in enumerate(modes):
        cells = [next(c for c in summary['capacity'] if c['vectors'] == size and c['mode'] == mode) for size in sizes]
        ax.plot(sizes, [c['qps'] if c['qps'] is not None else math.nan for c in cells],
                color=COLORS[mode], marker=['o', 's', '^', 'D', 'v'][index % 5], label=LABELS[mode])
    ax.set_xscale('log')
    ax.set_xlim(sizes[0] / 1.6, sizes[-1] * 1.6)
    if any(c['qps'] is not None for c in summary['capacity']):
        ax.set_yscale('log')
    else:
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.text(.5, .5, 'No capacity claim from these trials', transform=ax.transAxes,
                ha='center', color='#475569')
    ax.set_xticks(sizes, [f'{s:,}' for s in sizes])
    ax.set_xlabel('Dataset vectors')
    ax.set_ylabel('Offered queries / second')
    ax.legend(loc='upper right', fontsize=9)
    # Keep missing-evidence labels outside the data area so they cannot hide a passing point.
    missing = []
    for size in sizes:
        absent = [LABELS[c['mode']] for c in summary['capacity'] if c['vectors'] == size and c['qps'] is None]
        if absent:
            missing.append(f'{size:,} vectors: {", ".join(absent)}')
    if missing:
        note = textwrap.fill('No capacity claim — ' + '; '.join(missing), 130)
        fig.text(.01, .065, note, fontsize=8, color='#475569', va='bottom')
        save(fig, 'capacity', bottom=.10 + .035 * (note.count('\n') + 1))
    else:
        save(fig, 'capacity')

    fig, axes = plt.subplots(1, len(sizes), figsize=(5 * len(sizes), 4.8), squeeze=False)
    fig.suptitle('Tail latency as offered traffic increases', fontweight='bold')
    for ax, size in zip(axes[0], sizes):
        for mode in modes:
            cells = sorted((g for g in summary['groups'] if g['vectors'] == size and g['mode'] == mode and
                            g['scenario'] == 'steady' and g['metrics']['offered_p99_ms']), key=lambda g: g['rate'])
            if not cells:
                continue
            y = [g['metrics']['offered_p99_ms']['median'] for g in cells]
            ax.errorbar([g['rate'] for g in cells], y,
                        yerr=[[v - g['metrics']['offered_p99_ms']['min'] for v, g in zip(y, cells)],
                              [g['metrics']['offered_p99_ms']['max'] - v for v, g in zip(y, cells)]],
                        color=COLORS[mode], marker='o', capsize=3, label=LABELS[mode])
            bad = [g for g in cells if not g['all_pass']]
            ax.scatter([g['rate'] for g in bad], [g['metrics']['offered_p99_ms']['median'] for g in bad],
                       marker='x', color='black', s=65, zorder=5)
        ax.axhline(config['slo_ms'], color='#be123c', linestyle='--', linewidth=1)
        ax.set_title(f'{size:,} vectors')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Offered queries / second')
        ax.set_ylabel('Successful-request offered p99 (ms)')
        ax.legend(fontsize=8)
    fig.text(.5, .92, 'Median and min–max across trials · × means full capacity criteria did not pass', ha='center', fontsize=9)
    save(fig, 'latency')

    size = 100000 if 100000 in sizes else sizes[len(sizes) // 2]
    selected = [c for c in manifest['plan'] if c['vectors'] == size and c['scenario'] == 'burst' and c['repeat'] == 1]
    fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
    fig.suptitle(f'Burst and recovery · {size:,} vectors · predetermined trial 1', fontweight='bold')
    keys = ['throughput', 'p99_ms', 'queue_depth', 'rejected']
    labels = ['Completed QPS\n(rolling 1 s)', 'p99 upper bound\n(ms)', 'Queued requests', 'Cumulative rejections']
    for case in selected:
        path = directory / case['id']
        if not (path / 'record.json').exists():
            continue
        with (path / 'telemetry.csv').open() as source:
            telemetry = [r for r in csv.DictReader(source) if r['phase'] != 'drained']
        times = [float(r['elapsed_ms']) / 1000 for r in telemetry]
        for ax, key in zip(axes, keys):
            ax.plot(times, [float(r[key]) if r[key] else math.nan for r in telemetry],
                    color=COLORS[case['mode']], label=LABELS[case['mode']], linewidth=1.5)
    if selected:
        elapsed = 0
        for name, duration, rate, _, _, contention in selected[0]['phases']:
            end = elapsed + duration / 1000
            axes[0].plot([elapsed, end], [rate, rate], color='#111827', linestyle=':', linewidth=1.5)
            for ax in axes:
                if contention:
                    ax.axvspan(elapsed, end, color='#fbbf24', alpha=.18)
                ax.axvline(elapsed, color='#64748b', alpha=.3, linewidth=.6)
            axes[0].text((elapsed + end) / 2, 1.04, name, ha='center', transform=axes[0].get_xaxis_transform(), fontsize=9)
            elapsed = end
        axes[0].plot([], [], color='#111827', linestyle=':', label='Offered rate')
    for ax, label in zip(axes, labels):
        ax.set_ylabel(label)
    axes[0].set_ylim(bottom=0)
    for ax in axes[2:]:
        ax.set_ylim(0, max(1, ax.get_ylim()[1]))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    axes[1].set_yscale('symlog', linthresh=1)
    axes[1].set_ylim(bottom=0)
    axes[1].axhline(config['slo_ms'], color='#be123c', linestyle='--', linewidth=1)
    handles, labels = axes[0].get_legend_handles_labels()
    legend = dict(zip(labels, handles))
    labels = [name for name in [*(LABELS[mode] for mode in modes), 'Offered rate'] if name in legend]
    fig.legend([legend[name] for name in labels], labels, loc='upper center',
               bbox_to_anchor=(.5, .94), fontsize=9, ncol=min(5, len(labels)), frameon=False)
    axes[-1].set_xlabel('Elapsed seconds · shaded interval requests real CUDA contention')
    if config['cuda'] == 'off':
        axes[-1].set_xlabel('Elapsed seconds · CPU-only, no GPU contention')
    save(fig, 'burst')
    (directory / 'plot_metadata.json').write_text(json.dumps({'matplotlib': matplotlib.__version__}, indent=2) + '\n')
