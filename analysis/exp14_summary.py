'''
EXP-14 summary on the held-out TEST split of the 70:10:20 protocol. Headline = per-image mean DSC with paired
per-image t-tests; pooled DSC (the author's metric) is reported for reference only.

Input: per-image csv files written by analysis/eval_per_image.py --split test --interior-tau TAU
(<run>/analysis/per_image_metrics_test.csv). For every run: per-image mean DSC (all images and per lesion-size
tertile of the test split), per-image mean IoU, precision, recall, the interior-miss rate (missed / all lesion pixels
with normalized depth >= TAU, pooled) and pooled DSC / precision / recall. Every run is compared with --ref (the
same-batch original model) and, when given, with --ref2 (the fusion-only model) and in --pair VAR=REF couples:
paired t-test per image on DSC, precision, recall and the per-image interior-miss count.

  python analysis/exp14_summary.py --run A0=... --run A1=... --ref A0 --ref2 A1 --pair A2=A3 --pair A2=A4 \
      --title 'EXP-14 (isic17)' --out results/exp14_summary_isic17.md
'''
import os
import sys
import csv
import argparse

import numpy as np
from scipy import stats

TERT = ('small', 'mid', 'large')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', action='append', required=True, metavar='LABEL=DIR')
    p.add_argument('--describe', action='append', default=[], metavar='LABEL=TEXT')
    p.add_argument('--csv-name', default='per_image_metrics_test.csv')
    p.add_argument('--ref', required=True)
    p.add_argument('--ref2', default=None)
    p.add_argument('--pair', action='append', default=[], metavar='VAR=REF')
    p.add_argument('--title', default='EXP-14')
    p.add_argument('--out', required=True)
    return p.parse_args()


def vn(x, d=2, sign=False):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return '—'
    return (f'{x:+.{d}f}' if sign else f'{x:.{d}f}').replace('.', ',')


def px(n, sign=False):
    return (f'{n:+,}' if sign else f'{n:,}').replace(',', '.')


def load(path):
    rows = {r['filename']: r for r in csv.DictReader(open(path))}
    out = {}
    for name, r in rows.items():
        tp, fp, fn = int(r['tp']), int(r['fp']), int(r['fn'])
        out[name] = {'dsc': float(r['dsc']), 'iou': float(r['iou']), 'tp': tp, 'fp': fp, 'fn': fn, 'area': float(r['area_frac']),
                     'precision': tp / (tp + fp) if tp + fp else np.nan, 'recall': tp / (tp + fn) if tp + fn else np.nan,
                     'gt_int': int(r['gt_interior']) if r.get('gt_interior', '') != '' else 0,
                     'fn_int': int(r['fn_interior']) if r.get('fn_interior', '') != '' else 0}
    return out


def summary(rows, names, tert):
    a = lambda k: np.array([rows[n][k] for n in names], dtype=np.float64)
    tp, fp, fn = a('tp').sum(), a('fp').sum(), a('fn').sum()
    s = {'n': len(names), 'dsc': 100 * a('dsc').mean(), 'iou': 100 * a('iou').mean(),
         'precision': 100 * np.nanmean(a('precision')), 'recall': 100 * np.nanmean(a('recall')),
         'interior_miss_rate': 100 * a('fn_int').sum() / max(a('gt_int').sum(), 1), 'fn_interior': int(a('fn_int').sum()),
         'fp': int(fp), 'fn': int(fn), 'dsc_pooled': 100 * 2 * tp / (2 * tp + fp + fn),
         'precision_pooled': 100 * tp / (tp + fp), 'recall_pooled': 100 * tp / (tp + fn)}
    d = a('dsc')
    for k, t in enumerate(TERT):
        s[f'dsc_{t}'] = 100 * d[tert == k].mean()
    return s


def paired(rows_v, rows_r, names, key, mask=None):
    v = np.array([rows_v[n][key] for n in names], dtype=np.float64)
    r = np.array([rows_r[n][key] for n in names], dtype=np.float64)
    ok = np.isfinite(v) & np.isfinite(r) & (np.ones(len(names), bool) if mask is None else mask)
    if ok.sum() < 3:
        return float('nan'), float('nan')
    t = stats.ttest_rel(v[ok], r[ok])
    scale = 1 if key == 'fn_int' else 100
    return float(scale * (v[ok] - r[ok]).mean()), float(t.pvalue)


def main():
    a = parse_args()
    runs, desc = {}, dict(s.split('=', 1) for s in a.describe)
    for spec in a.run:
        label, path = spec.split('=', 1)
        f = path if os.path.isfile(path) else os.path.join(path, 'analysis', a.csv_name)
        if os.path.isfile(f):
            runs[label] = load(f)
        else:
            print(f'skip {label}: {f} missing')
    if a.ref not in runs:
        sys.exit(f'reference {a.ref} missing')
    names = sorted(runs[a.ref])
    for lab, r in runs.items():
        if sorted(r) != names:
            sys.exit(f'{lab} does not cover the same test images as {a.ref}')
    area = np.array([runs[a.ref][n]['area'] for n in names])
    tert = np.digitize(area, np.quantile(area, [1 / 3, 2 / 3]))
    summ = {lab: summary(r, names, tert) for lab, r in runs.items()}
    comps = [(lab, a.ref) for lab in runs if lab != a.ref]
    if a.ref2 and a.ref2 in runs:
        comps += [(lab, a.ref2) for lab in runs if lab not in (a.ref, a.ref2)]
    for spec in a.pair:
        v, r = spec.split('=', 1)
        if v in runs and r in runs and (v, r) not in comps:
            comps.append((v, r))
    tests = []
    for v, r in comps:
        row = {'variant': v, 'reference': r}
        for key in ('dsc', 'precision', 'recall', 'fn_int'):
            row[f'd_{key}'], row[f'p_{key}'] = paired(runs[v], runs[r], names, key)
        for k, t in enumerate(TERT):
            row[f'd_dsc_{t}'], row[f'p_dsc_{t}'] = paired(runs[v], runs[r], names, 'dsc', tert == k)
        tests.append(row)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(os.path.splitext(a.out)[0] + '.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['run'] + list(next(iter(summ.values()))))
        w.writeheader()
        for lab, s in summ.items():
            w.writerow({'run': lab, **s})
    with open(os.path.splitext(a.out)[0] + '_tests.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(tests[0]) if tests else ['variant'])
        w.writeheader()
        w.writerows(tests)
    st = lambda p: '' if not np.isfinite(p) else ('***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else '')
    md = [f'# {a.title} — tập test riêng ({len(names)} ảnh), giao thức 70:10:20', '',
          'Số chính: **DSC trung bình theo ảnh** (tính DSC từng ảnh rồi lấy trung bình) và paired t-test theo ảnh. DSC gộp '
          '(cách của paper gốc) chỉ để tham khảo. "Bỏ sót interior" = tỉ lệ pixel lesion có độ sâu chuẩn hóa ≥ τ mà model không tô.', '',
          '| Run | Mô tả | DSC TB theo ảnh | nhỏ · vừa · to | IoU TB | Precision TB | Recall TB | Bỏ sót interior | FP | DSC gộp (tham khảo) |',
          '|---|---|---|---|---|---|---|---|---|---|']
    for lab, s in summ.items():
        md.append(f'| {lab} | {desc.get(lab, "")} | **{vn(s["dsc"])}** | {vn(s["dsc_small"])} · {vn(s["dsc_mid"])} · {vn(s["dsc_large"])} '
                  f'| {vn(s["iou"])} | {vn(s["precision"])} | {vn(s["recall"])} | {vn(s["interior_miss_rate"])} % ({px(s["fn_interior"])} px) '
                  f'| {px(s["fp"])} | {vn(s["dsc_pooled"])} |')
    md += ['', '## Paired t-test theo ảnh (Δ = run − mốc; * p < 0,05, ** p < 0,01, *** p < 0,001)', '',
           '| Run | Mốc | Δ DSC (p) | Δ DSC nhỏ · vừa · to | Δ precision (p) | Δ recall (p) | Δ pixel bỏ sót interior / ảnh (p) |',
           '|---|---|---|---|---|---|---|']
    for t in tests:
        md.append(f'| {t["variant"]} | {t["reference"]} | **{vn(t["d_dsc"], 2, True)}**{st(t["p_dsc"])} ({vn(t["p_dsc"], 4)}) '
                  f'| {vn(t["d_dsc_small"], 2, True)}{st(t["p_dsc_small"])} · {vn(t["d_dsc_mid"], 2, True)}{st(t["p_dsc_mid"])} · '
                  f'{vn(t["d_dsc_large"], 2, True)}{st(t["p_dsc_large"])} | {vn(t["d_precision"], 2, True)}{st(t["p_precision"])} '
                  f'({vn(t["p_precision"], 4)}) | {vn(t["d_recall"], 2, True)}{st(t["p_recall"])} ({vn(t["p_recall"], 4)}) '
                  f'| {vn(t["d_fn_int"], 0, True)}{st(t["p_fn_int"])} ({vn(t["p_fn_int"], 4)}) |')
    with open(a.out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md) + '\n')
    print('\n'.join(md))
    print(f'wrote {a.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
