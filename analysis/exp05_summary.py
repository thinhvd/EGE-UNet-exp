'''
EXP-5 summary: one Markdown report (+ a long-format CSV) over the datasets of the batch.

Per dataset:
  Table 1 - per model: params, best epoch, pooled mIoU / DSC (from test_results.json, i.e. the
            training-time test), and per-image mean DSC / IoU on all images, the smallest and the
            largest lesion tertile (from the significance CSVs).
  Table 2 - per variant vs the baseline, per stratum and metric: mean difference, 95% CI, paired
            t statistic, p, Holm-adjusted p, Cohen d_z, n (analysis/significance_tests.py output).

Reads only files that survive the LEAN sync (test_results.json, log/train.info.log, the analysis
CSVs), so it runs identically on the server and on the synced local batch folder. Stdlib only.

  python analysis/exp05_summary.py --results-dir results --datasets "isic17 isic18" --seed 42 \
      --out results/exp05_summary.md
'''
import os
import re
import csv
import json
import argparse

MODELS = [   # label, run-name pattern ({ds}, {seed})
    ('learnable', 'egeunet_{ds}_learnable_s{seed}'),
    ('none+sum-deep3', 'egeunet_{ds}_none_fuse-sum-deep3_s{seed}'),
    ('none+sum_attn-all5', 'egeunet_{ds}_none_fuse-sum_attn-all5_s{seed}'),
]
STRATA = [('all', 'all'), ('area_T1 (smallest)', 'small'), ('area_T3 (largest)', 'large')]
METRICS = ['dsc', 'iou']


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--results-dir', default='results')
    p.add_argument('--datasets', default='isic17 isic18', help='space-separated')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out', required=True, help='markdown path; a .csv with the same stem is written too')
    return p.parse_args()


def read_log_facts(run_dir):
    '''params count, rotation angle line and code revision from log/train.info.log (None if absent).'''
    facts = {'params': None, 'rotation': None, 'revision': None}
    path = os.path.join(run_dir, 'log', 'train.info.log')
    if not os.path.isfile(path):
        return facts
    with open(path) as f:
        for line in f:
            m = re.search(r'params: (\d+) total', line)
            if m and facts['params'] is None:
                facts['params'] = int(m.group(1))
            m = re.search(r'rotation angle: (.*)$', line.strip())
            if m and facts['rotation'] is None:
                facts['rotation'] = m.group(1)
            m = re.search(r'code revision: (.*)$', line.strip())
            if m and facts['revision'] is None:
                facts['revision'] = m.group(1)
    return facts


def read_significance(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def fmt(x, nd=2, scale=100.0, sign=False):
    if x is None:
        return '—'
    v = float(x) * scale
    return f'{v:+.{nd}f}' if sign else f'{v:.{nd}f}'


def main():
    args = parse_args()
    md = ['# EXP-5 summary', '',
          f'results dir: `{args.results_dir}` · seed {args.seed} · one training run per configuration; '
          'p-values are paired per-image tests (unit = image, not training run).', '']
    long_rows = []

    for ds in args.datasets.split():
        md += [f'## {ds}', '']
        sig_dir = os.path.join(args.results_dir, f'exp05_{ds}', 'significance')
        sig = {}
        for metric in METRICS:
            path = os.path.join(sig_dir, f'significance_{metric}.csv')
            if os.path.isfile(path):
                sig[metric] = read_significance(path)
        if not sig:
            md += [f'_no significance CSVs under {sig_dir} — run scripts/exp05_analyze.sh first_', '']
            continue

        # per-stratum means: the reference's from mean_ref of any row, the variants' from mean_var
        def stratum_mean(metric, label, stratum):
            for r in sig.get(metric, []):
                if r['stratum'] == stratum and (r['group'] == label or (label == MODELS[0][0] and r['vs'] == label)):
                    return r['mean_var'] if r['group'] == label else r['mean_ref']
            return None

        # provenance
        facts = {}
        for label, pat in MODELS:
            run_dir = os.path.join(args.results_dir, pat.format(ds=ds, seed=args.seed))
            facts[label] = (run_dir, read_log_facts(run_dir))
        rotations = sorted({f['rotation'] for _, f in facts.values() if f['rotation']})
        revisions = sorted({f['revision'] for _, f in facts.values() if f['revision']})
        n_images = sig[next(iter(sig))][0]['n'] if sig[next(iter(sig))] else '?'
        md += [f'- images: {n_images} · rotation angle line(s): {rotations or "n/a"} · code revision(s): {revisions or "n/a"}']
        meta_path = os.path.join(args.results_dir, f'exp05_{ds}', 'compare_dsc', 'compare_runs_dsc_meta.json')
        if os.path.isfile(meta_path):
            edges = json.load(open(meta_path)).get('tertile_edges', {}).get('area_frac')
            if edges:
                md += [f'- lesion-area tertile edges (fraction of image): small ≤ {100 * edges[0]:.2f}% < mid ≤ {100 * edges[1]:.2f}% < large']
        md += ['']

        # Table 1
        head = ['Model', 'params', 'best epoch', 'mIoU pooled', 'DSC pooled']
        head += [f'DSC {s}' for _, s in STRATA] + [f'IoU {s}' for _, s in STRATA]
        md += ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
        for label, _ in MODELS:
            run_dir, f = facts[label]
            tr_path = os.path.join(run_dir, 'test_results.json')
            tr = json.load(open(tr_path)) if os.path.isfile(tr_path) else {}
            row = [label, str(f['params'] or '—'), str(tr.get('min_epoch', '—')),
                   fmt(tr.get('miou')), fmt(tr.get('f1_or_dsc'))]
            for metric in METRICS:
                for stratum, _ in STRATA:
                    row.append(fmt(stratum_mean(metric, label, stratum)))
            md += ['| ' + ' | '.join(row) + ' |']
            long_rows.append({'dataset': ds, 'model': label, 'kind': 'level', 'params': f['params'],
                              'best_epoch': tr.get('min_epoch'), 'miou_pooled': tr.get('miou'),
                              'dsc_pooled': tr.get('f1_or_dsc'),
                              **{f'{m}_{s}': stratum_mean(m, label, st) for m in METRICS for st, s in STRATA}})
        md += ['', 'Per-image means are ×100; "small"/"large" = lowest/highest lesion-area tertile of this dataset.', '']

        # Table 2
        head2 = ['Variant', 'stratum', 'metric', 'Δ vs learnable', '95% CI', 't', 'p', 'p Holm', 'd_z', 'n']
        md += ['| ' + ' | '.join(head2) + ' |', '|' + '---|' * len(head2)]
        for label, _ in MODELS[1:]:
            for stratum, sname in STRATA:
                for metric in METRICS:
                    r = next((r for r in sig.get(metric, []) if r['group'] == label and r['stratum'] == stratum), None)
                    if r is None:
                        continue
                    p_h = float(r['p_ttest_holm'])
                    star = '***' if p_h < 0.001 else '**' if p_h < 0.01 else '*' if p_h < 0.05 else ''
                    md += ['| ' + ' | '.join([
                        label, sname, metric.upper(), fmt(r['diff'], sign=True),
                        f'[{fmt(r["ci_lo"], sign=True)}, {fmt(r["ci_hi"], sign=True)}]',
                        f'{float(r["t_stat"]):+.2f}', f'{float(r["p_ttest"]):.4f}', f'{p_h:.4f}{star}',
                        f'{float(r["cohen_dz"]):+.2f}', r['n']]) + ' |']
                    long_rows.append({'dataset': ds, 'model': label, 'kind': 'test', 'stratum': sname,
                                      'metric': metric, 'diff': r['diff'], 'ci_lo': r['ci_lo'], 'ci_hi': r['ci_hi'],
                                      't_stat': r['t_stat'], 'df': r['df'], 'p_ttest': r['p_ttest'],
                                      'p_ttest_holm': r['p_ttest_holm'], 'p_wilcoxon': r['p_wilcoxon'],
                                      'p_wilcoxon_holm': r['p_wilcoxon_holm'], 'cohen_dz': r['cohen_dz'],
                                      'n': r['n']})
        md += ['', 'Δ and CI are ×100 (variant − learnable, paired per image). Holm is over the 2 variants '
               'within each stratum. Stars: * <0.05, ** <0.01, *** <0.001 on the Holm-adjusted p.', '']

    with open(args.out, 'w') as f:
        f.write('\n'.join(md) + '\n')
    csv_path = os.path.splitext(args.out)[0] + '.csv'
    fields = []
    for r in long_rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(long_rows)
    print('\n'.join(md))
    print(f'wrote {args.out} and {csv_path}')


if __name__ == '__main__':
    main()
