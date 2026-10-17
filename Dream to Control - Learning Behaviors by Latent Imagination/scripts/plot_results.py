"""Render traceable short-run diagnostics from saved JSON records."""
import argparse
import json
import pathlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--training', required=True, type=pathlib.Path)
    parser.add_argument('--evaluation', required=True, type=pathlib.Path)
    parser.add_argument('--output', required=True, type=pathlib.Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    metrics = [json.loads(x) for x in (args.training / 'metrics.jsonl').read_text().splitlines() if x.strip()]
    episodes = [json.loads(x) for x in (args.evaluation / 'episodes.jsonl').read_text().splitlines() if x.strip()]
    summary = json.loads((args.evaluation / 'summary.json').read_text())
    assert summary['status'] == 'completed' and summary['weights_unchanged']
    plt.rcParams.update({'font.family': ['DejaVu Sans', 'sans-serif'], 'font.size': 12,
                         'axes.spines.right': False, 'axes.spines.top': False,
                         'axes.linewidth': 1.5, 'legend.frameon': False,
                         'svg.fonttype': 'none'})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for key, label, color, marker in [('train/return', 'Train (with exploration)', '#767676', '.'),
                                       ('test/return', 'Periodic evaluation', '#0F4D92', 'o')]:
        values = [(r['step'], r[key]) for r in metrics if key in r]
        axes[0].plot([v[0] / 1000 for v in values], [v[1] for v in values],
                     label=label, color=color, marker=marker, linewidth=1.8)
    axes[0].set(xlabel='Environment steps (thousands)', ylabel='Episode return', title='(a) Collection and evaluation')
    axes[0].legend(fontsize=10, loc='best')
    values = [(r['step'], r['model_loss']) for r in metrics if 'model_loss' in r]
    axes[1].plot([v[0] / 1000 for v in values], [v[1] for v in values],
                 color='#42949E', linewidth=2, marker='.')
    axes[1].set(xlabel='Environment steps (thousands)', ylabel='World-model loss', title='(b) Model optimization')
    returns = np.array([r['return'] for r in episodes])
    axes[2].scatter(np.arange(1, len(returns) + 1), returns, color='#0F4D92', s=38)
    axes[2].axhline(returns.mean(), color='#B64342', linestyle='--', label=f'Mean {returns.mean():.2f}')
    axes[2].set(xlabel='Evaluation episode', ylabel='Episode return', title='(c) Frozen checkpoint evaluation',
                xticks=np.arange(1, len(returns) + 1))
    axes[2].legend(fontsize=10)
    for ax in axes:
        ax.grid(axis='y', color='#EEEEEE', linewidth=.7)
        ax.set_axisbelow(True)
    fig.suptitle(f"Dreamer v1 | Walker Walk | {summary['training_environment_steps']:,} environment steps", fontsize=16)
    fig.text(.5, .015, 'One training seed; short-run pipeline verification. Evaluation spread is across episodes, not training seeds.',
             ha='center', fontsize=10, color='#4D4D4D')
    fig.tight_layout(rect=(0, .045, 1, .97), pad=1.5)
    for extension in ('png', 'svg'):
        fig.savefig(args.output / ('short_run_diagnostics.' + extension), dpi=300, facecolor='white')
    plt.close(fig)
    provenance = {'training_metrics': str((args.training / 'metrics.jsonl').resolve()),
                  'evaluation_episodes': str((args.evaluation / 'episodes.jsonl').resolve()),
                  'checkpoint_sha256': summary['checkpoint_sha256'],
                  'smoothing': 'none', 'aggregation': 'unweighted arithmetic mean across evaluation episodes'}
    (args.output / 'figure_provenance.json').write_text(json.dumps(provenance, indent=2))

if __name__ == '__main__':
    main()
