'''
Paired per-image significance tests between a reference run and one or more variant runs.

Every model is evaluated on the SAME 650 test images, so each image gives one matched pair and the
tests below are paired. For each variant and each stratum this reports the mean difference, its 95%
CI, a paired t-test, a Wilcoxon signed-rank test, Cohen's d_z and the Holm-adjusted p-values.

SCOPE OF THE CLAIM. The unit of analysis here is the IMAGE, so the null being tested is "these two
trained checkpoints score the same on this image population". It is NOT "these two architectures
score the same": for that the unit would have to be the training run, of which there is one per
configuration. Measured on this repo (6 retrainings of one configuration, seed 42): a paired test
between two runs of the SAME architecture returns p < 0.05 for 6 of 15 pairs on the full test set
and 5 of 15 on the small-lesion group. Quote results from this script as single-run comparisons and
treat differences below ~1.3 mIoU / ~2.3 DSC on the small group as within retraining noise
(exp_docs/04 section 11b).

Inputs are the CSVs written by analysis/eval_per_image.py. The first --run is the reference:
  python analysis/significance_tests.py --out-dir results/.../significance \
      --run learnable=results/.../egeunet_isic17_learnable_s42 \
      --run sum-deep3=results/.../egeunet_isic17_learnable_fuse-sum-deep3_s42
'''
import os
import sys
import csv
import argparse

import numpy as np

try:
    from scipy.stats import ttest_rel, wilcoxon, t as tdist
except ImportError:  # pragma: no cover
    sys.exit('this script needs scipy (pip install scipy)')

METRICS = ['dsc', 'iou', 'oracle_dsc']


# Same two helpers as analysis/compare_runs.py, inlined so this script pulls in no plotting stack.
def load_csv(path):
    with open(path, newline='') as f:
        return {r['filename']: r for r in csv.DictReader(f)}


def resolve_csv(spec, csv_name):
    if os.path.isfile(spec):
        return spec
    cand = os.path.join(spec, 'analysis', csv_name)
    if os.path.isfile(cand):
        return cand
    raise FileNotFoundError(f'no csv found for {spec!r} (looked for {cand}); run analysis/eval_per_image.py first')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', action='append', required=True, metavar='LABEL=DIR',
                   help='repeat once per run; the FIRST one is the reference everything is compared to')
    p.add_argument('--csv-name', default='per_image_metrics_full.csv')
    p.add_argument('--metric', default='dsc', choices=METRICS)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--n-tertiles', type=int, default=3,
                   help='split images by lesion area into this many equal groups (default 3: small/mid/large)')
    return p.parse_args()


def holm(pvals):
    '''Holm-Bonferroni adjusted p-values, same order as the input.'''
    p = np.asarray(pvals, dtype=np.float64)
    m = p.size
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj


def paired_stats(a, b):
    '''a = variant, b = reference; both indexed by the same images.'''
    d = a - b
    n = d.size
    mean = float(d.mean())
    sd = float(d.std(ddof=1))
    se = sd / np.sqrt(n)
    t = ttest_rel(a, b)
    # t-based CI of the mean paired difference (same distributional assumption as the t-test itself)
    half = tdist.ppf(0.975, n - 1) * se
    try:
        pw = float(wilcoxon(a, b).pvalue) if np.any(d != 0) else float('nan')
    except ValueError:
        pw = float('nan')
    return {
        'n': n,
        'mean_ref': float(b.mean()),
        'mean_var': float(a.mean()),
        'diff': mean,
        'ci_lo': mean - half,
        'ci_hi': mean + half,
        # same statistic the user's notebook prints (ttest_rel(variant, reference): t > 0 = variant higher)
        't_stat': float(t.statistic),
        'df': n - 1,
        'p_ttest': float(t.pvalue),
        'p_wilcoxon': pw,
        'cohen_dz': mean / sd if sd > 0 else float('nan'),
        'frac_better': float((d > 0).mean()),
    }


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    runs = []
    for spec in args.run:
        if '=' not in spec:
            sys.exit(f'--run expects LABEL=DIR, got {spec!r}')
        label, path = spec.split('=', 1)
        runs.append((label, load_csv(resolve_csv(path, args.csv_name))))

    ref_label, ref_rows = runs[0]
    names = sorted(ref_rows)
    for label, rows in runs[1:]:
        if sorted(rows) != names:
            sys.exit(f'run {label!r} does not cover the same images as the reference {ref_label!r}')

    area = np.array([float(ref_rows[n]['area_frac']) for n in names])
    edges = np.quantile(area, np.linspace(0, 1, args.n_tertiles + 1)[1:-1])
    bins = np.digitize(area, edges)
    strata = [('all', np.ones(len(names), dtype=bool))]
    labels = {0: 'smallest', args.n_tertiles - 1: 'largest'}
    for k in range(args.n_tertiles):
        strata.append((f'area_T{k + 1}' + (f' ({labels[k]})' if k in labels else ''), bins == k))

    values = {label: np.array([float(rows[n][args.metric]) for n in names]) for label, rows in runs}

    out = []
    for sname, mask in strata:
        block = []
        for label, _ in runs[1:]:
            row = {'stratum': sname, 'group': label, 'vs': ref_label, 'metric': args.metric}
            row.update(paired_stats(values[label][mask], values[ref_label][mask]))
            block.append(row)
        for key in ('p_ttest', 'p_wilcoxon'):
            adj = holm([r[key] for r in block])
            for r, a in zip(block, adj):
                r[key + '_holm'] = float(a)
        out.extend(block)

    fields = ['stratum', 'group', 'vs', 'metric', 'n', 'mean_ref', 'mean_var', 'diff', 'ci_lo', 'ci_hi',
              'cohen_dz', 'frac_better', 't_stat', 'df', 'p_ttest', 'p_ttest_holm', 'p_wilcoxon',
              'p_wilcoxon_holm']
    path = os.path.join(args.out_dir, f'significance_{args.metric}.csv')
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out)

    def stars(p):
        return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''

    print(f'reference: {ref_label}   metric: {args.metric}   images: {len(names)}')
    print(f'p_holm = Holm-adjusted over the {len(runs) - 1} comparisons inside each stratum; '
          f'stars use the ADJUSTED p of the t-test\n')
    for sname, _ in strata:
        block = [r for r in out if r['stratum'] == sname]
        print(f'--- {sname} (n={block[0]["n"]}, reference mean {100 * block[0]["mean_ref"]:.2f}) ---')
        print(f'{"variant":22s} {"mean":>7s} {"diff":>7s} {"95% CI":>16s} {"d_z":>6s} {"t":>7s} {"p_t":>8s} '
              f'{"p_t_holm":>9s} {"p_W":>8s} {"p_W_holm":>9s}')
        for r in block:
            ci = f'[{100 * r["ci_lo"]:+.2f},{100 * r["ci_hi"]:+.2f}]'
            print(f'{r["group"]:22s} {100 * r["mean_var"]:7.2f} {100 * r["diff"]:+7.2f} {ci:>16s} '
                  f'{r["cohen_dz"]:+6.2f} {r["t_stat"]:+7.2f} {r["p_ttest"]:8.4f} {r["p_ttest_holm"]:9.4f} '
                  f'{r["p_wilcoxon"]:8.4f} {r["p_wilcoxon_holm"]:9.4f} {stars(r["p_ttest_holm"])}')
        print()
    print(f'wrote {path}')


if __name__ == '__main__':
    main()
