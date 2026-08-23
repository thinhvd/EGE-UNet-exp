'''
Compare per-image metrics of several runs (e.g. hpa_mode = learnable / frozen_ones / none), stratified by
lesion position (centroid offset from the image center) and lesion size (area fraction).

The hypothesis test for "the learned prior P exploits the centered-lesion bias of ISIC" is the
INTERACTION: does the gain of a run over the reference differ between the most-central stratum (T1)
and the most-off-center stratum (T3)? Reported as diff_T1 - diff_T3 with a bootstrap 95% CI.

Inputs are the CSVs written by analysis/eval_per_image.py. Pass either a work_dir (uses
<work_dir>/analysis/<csv-name>) or a CSV path. Several seeds of the same variant can be grouped:
  --run learnable=results/egeunet_isic17_learnable_s42 --run frozen=results/egeunet_isic17_frozen_ones_s42
  --group learnable=dirA,dirB,dirC --group frozen=dirD,dirE,dirF --group none=dirG,dirH,dirI
The first run/group is the reference. numpy + csv only (scipy optional for the Wilcoxon test).
'''
import os
import sys
import csv
import json
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis import common as C

try:
    from scipy.stats import wilcoxon
except Exception:  # scipy is optional
    wilcoxon = None


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', '--group', dest='specs', action='append', required=True, metavar='LABEL=DIR[,DIR...]',
                   help='a run (one dir/csv) or a group of seeds (comma-separated); repeatable; first = reference')
    p.add_argument('--csv-name', default='per_image_metrics.csv',
                   help='csv file name under <dir>/analysis/ (e.g. per_image_metrics_shift32_0.csv)')
    p.add_argument('--metric', default='dsc', choices=['dsc', 'iou'])
    p.add_argument('--n-strata', type=int, default=3)
    p.add_argument('--n-boot', type=int, default=10000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out-dir', default=None, help='default: <first dir>/analysis/compare')
    p.add_argument('--dpi', type=int, default=150)
    return p.parse_args()


def load_csv(path):
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    return {r['filename']: r for r in rows}


def resolve_csv(spec, csv_name):
    if os.path.isfile(spec):
        return spec
    cand = os.path.join(spec, 'analysis', csv_name)
    if os.path.isfile(cand):
        return cand
    raise FileNotFoundError(f'no csv found for {spec!r} (looked for {cand}); run analysis/eval_per_image.py first')


def boot_ci(values, rng, n_boot, stat=np.mean, alpha=0.05):
    values = np.asarray(values, dtype=np.float64)
    n = values.size
    if n == 0:
        return float('nan'), float('nan')
    idx = rng.integers(0, n, size=(n_boot, n))
    s = stat(values[idx], axis=1)
    return float(np.percentile(s, 100 * alpha / 2)), float(np.percentile(s, 100 * (1 - alpha / 2)))


def boot_ci_diff_of_means(a, b, rng, n_boot, alpha=0.05):
    '''CI of mean(a) - mean(b) with independent resampling of the two (disjoint) groups.'''
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return float('nan'), float('nan')
    sa = a[rng.integers(0, a.size, size=(n_boot, a.size))].mean(axis=1)
    sb = b[rng.integers(0, b.size, size=(n_boot, b.size))].mean(axis=1)
    d = sa - sb
    return float(np.percentile(d, 100 * alpha / 2)), float(np.percentile(d, 100 * (1 - alpha / 2)))


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    # ---- load
    groups = []  # (label, [csv paths])
    for spec in args.specs:
        if '=' not in spec:
            raise ValueError(f'--run/--group expects LABEL=DIR[,DIR...], got {spec!r}')
        label, dirs = spec.split('=', 1)
        paths = [resolve_csv(d.strip(), args.csv_name) for d in dirs.split(',') if d.strip()]
        groups.append((label, paths))
    ref_label = groups[0][0]
    out_dir = args.out_dir or os.path.join(os.path.dirname(groups[0][1][0]), 'compare')
    os.makedirs(out_dir, exist_ok=True)

    tables = {label: [load_csv(p) for p in paths] for label, paths in groups}
    first = tables[ref_label][0]
    files = list(first.keys())
    common = [f for f in files if all(f in t for tabs in tables.values() for t in tabs)]
    if len(common) < len(files):
        print(f'[warn] {len(files) - len(common)} images missing in some csv; using the {len(common)} common ones')
    if not common:
        raise SystemExit('no common images across the given csv files')
    n = len(common)

    # per group: (n_seeds, n_images) metric matrix; descriptors from the reference
    M = {}
    for label, tabs in tables.items():
        M[label] = np.array([[float(t[f][args.metric]) for f in common] for t in tabs], dtype=np.float64)
    desc = {k: np.array([float(first[f][k]) for f in common]) for k in ('centroid_offset', 'area_frac')}
    for label, tabs in tables.items():
        for t in tabs:
            for k in desc:
                other = np.array([float(t[f][k]) for f in common])
                if not np.allclose(other, desc[k], equal_nan=True, atol=1e-9):
                    print(f'[warn] descriptor {k} differs between {ref_label} and {label} (different GT pipeline?)')
                    break

    # ---- strata (tertiles of the reference descriptors)
    def strata(values, name, low_label, high_label):
        edges = np.quantile(values, np.linspace(0, 1, args.n_strata + 1)[1:-1])
        bins = np.digitize(values, edges)  # 0..n_strata-1
        out = [('all', np.arange(n))]
        for s in range(args.n_strata):
            tag = f'{name}_T{s + 1}'
            if s == 0:
                tag += f' ({low_label})'
            if s == args.n_strata - 1:
                tag += f' ({high_label})'
            out.append((tag, np.nonzero(bins == s)[0]))
        return out, edges

    strat_offset, edges_off = strata(desc['centroid_offset'], 'offset', 'most central', 'most off-center')
    strat_area, edges_area = strata(desc['area_frac'], 'area', 'smallest', 'largest')

    # ---- per stratum x group
    ref_img = M[ref_label].mean(axis=0)  # per-image mean over seeds
    rows = []
    for strat_name, strat_list in (('centroid_offset', strat_offset), ('area_frac', strat_area)):
        for tag, idx in strat_list:
            for label in M:
                mat = M[label][:, idx]
                img_mean = mat.mean(axis=0)                 # per-image mean over seeds
                per_seed = mat.mean(axis=1)                 # per-seed stratum mean
                row = {'stratification': strat_name, 'stratum': tag, 'n_images': int(idx.size), 'group': label,
                       'n_seeds': int(mat.shape[0]), 'mean': float(img_mean.mean()),
                       'std_over_seeds': float(per_seed.std(ddof=1)) if mat.shape[0] > 1 else float('nan')}
                lo, hi = boot_ci(img_mean, rng, args.n_boot)
                row['ci_lo'], row['ci_hi'] = lo, hi
                if label != ref_label:
                    d = img_mean - ref_img[idx]
                    row['diff_vs_ref'] = float(d.mean())
                    row['diff_ci_lo'], row['diff_ci_hi'] = boot_ci(d, rng, args.n_boot)
                    row['frac_better'] = float((d > 0).mean())
                    if wilcoxon is not None and d.size >= 10 and np.any(d != 0):
                        try:
                            row['wilcoxon_p'] = float(wilcoxon(d).pvalue)
                        except Exception:
                            row['wilcoxon_p'] = float('nan')
                    else:
                        row['wilcoxon_p'] = float('nan')
                rows.append(row)

    # ---- interaction: diff in T1 minus diff in T_last
    inter = []
    for strat_name, strat_list in (('centroid_offset', strat_offset), ('area_frac', strat_area)):
        idx1, idxk = strat_list[1][1], strat_list[-1][1]
        for label in M:
            if label == ref_label:
                continue
            d_all = M[label].mean(axis=0) - ref_img
            d1, dk = d_all[idx1], d_all[idxk]
            lo, hi = boot_ci_diff_of_means(d1, dk, rng, args.n_boot)
            inter.append({'stratification': strat_name, 'group': label, 'vs': ref_label,
                          'diff_T1': float(d1.mean()) if d1.size else float('nan'),
                          'diff_Tlast': float(dk.mean()) if dk.size else float('nan'),
                          'interaction': float(d1.mean() - dk.mean()) if d1.size and dk.size else float('nan'),
                          'ci_lo': lo, 'ci_hi': hi, 'n_T1': int(idx1.size), 'n_Tlast': int(idxk.size)})

    # ---- write
    strata_csv = os.path.join(out_dir, f'compare_runs_{args.metric}_strata.csv')
    fields = ['stratification', 'stratum', 'n_images', 'group', 'n_seeds', 'mean', 'std_over_seeds', 'ci_lo', 'ci_hi',
              'diff_vs_ref', 'diff_ci_lo', 'diff_ci_hi', 'frac_better', 'wilcoxon_p']
    with open(strata_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, restval='')
        w.writeheader(); w.writerows(rows)
    inter_csv = os.path.join(out_dir, f'compare_runs_{args.metric}_interaction.csv')
    with open(inter_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['stratification', 'group', 'vs', 'diff_T1', 'diff_Tlast', 'interaction',
                                          'ci_lo', 'ci_hi', 'n_T1', 'n_Tlast'])
        w.writeheader(); w.writerows(inter)
    json.dump({'metric': args.metric, 'n_images': n, 'groups': {l: p for l, p in groups},
               'tertile_edges': {'centroid_offset': edges_off.tolist(), 'area_frac': edges_area.tolist()},
               'n_boot': args.n_boot, 'seed': args.seed},
              open(os.path.join(out_dir, f'compare_runs_{args.metric}_meta.json'), 'w'), indent=2)

    # ---- figure: grouped bars per stratum, one panel per stratification
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    labels = list(M.keys())
    w = 0.8 / len(labels)
    for ax, (strat_name, strat_list) in zip(axes, (('centroid_offset', strat_offset), ('area_frac', strat_area))):
        tags = [t for t, _ in strat_list]
        x = np.arange(len(tags))
        for gi, label in enumerate(labels):
            rs = [r for r in rows if r['stratification'] == strat_name and r['group'] == label]
            means = np.array([r['mean'] for r in rs]); lo = np.array([r['ci_lo'] for r in rs]); hi = np.array([r['ci_hi'] for r in rs])
            ax.bar(x + (gi - (len(labels) - 1) / 2) * w, means, width=w * 0.9, color=C.CATEGORICAL[gi % len(C.CATEGORICAL)],
                   label=label, yerr=[means - lo, hi - means], capsize=2, error_kw={'lw': 0.8})
        ax.set_xticks(x); ax.set_xticklabels([t.replace(' (', '\n(') for t in tags], fontsize=8)
        ymin = min(r['ci_lo'] for r in rows if r['stratification'] == strat_name)
        ax.set_ylim(max(0.0, ymin - 0.05), 1.0)
        ax.set_ylabel(args.metric.upper())
        ax.set_title(f'{args.metric.upper()} by {strat_name} tertile (bars = mean, whiskers = bootstrap 95% CI)', fontsize=9)
        ax.grid(axis='y', alpha=0.25)
        ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'compare_runs_{args.metric}.png'), dpi=args.dpi, bbox_inches='tight')
    plt.close(fig)

    # ---- stdout
    print(f'\n{args.metric.upper()} per stratum  (n_images={n}; reference = {ref_label})')
    hdr = f'{"stratification":16s} {"stratum":28s} {"n":>4s} {"group":14s} {"seeds":>5s} {"mean":>8s} {"sd_seed":>8s} {"diff":>8s} {"95% CI":>19s} {"p_wilc":>8s}'
    print(hdr); print('-' * len(hdr))
    for r in rows:
        diff = C.fmt(r.get('diff_vs_ref')) if 'diff_vs_ref' in r else ''
        ci = f'[{C.fmt(r["diff_ci_lo"])}, {C.fmt(r["diff_ci_hi"])}]' if 'diff_vs_ref' in r else ''
        pw = C.fmt(r.get('wilcoxon_p'), 4) if 'diff_vs_ref' in r else ''
        print(f'{r["stratification"]:16s} {r["stratum"]:28s} {r["n_images"]:4d} {r["group"]:14s} {r["n_seeds"]:5d} '
              f'{C.fmt(r["mean"], 4):>8s} {C.fmt(r["std_over_seeds"], 4):>8s} {diff:>8s} {ci:>19s} {pw:>8s}')
    print(f'\nInteraction (diff in T1 - diff in T{args.n_strata}); CI excluding 0 => the gain over {ref_label} depends on the stratum')
    for r in inter:
        print(f'  {r["stratification"]:16s} {r["group"]:14s} diff_T1={C.fmt(r["diff_T1"], 4)}  diff_T{args.n_strata}={C.fmt(r["diff_Tlast"], 4)}  '
              f'interaction={C.fmt(r["interaction"], 4)}  95% CI [{C.fmt(r["ci_lo"], 4)}, {C.fmt(r["ci_hi"], 4)}]')
    print(f'\nwrote {out_dir}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
