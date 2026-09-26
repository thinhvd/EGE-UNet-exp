'''
EXP-6 summary: one table per dataset, decided on pooled DSC (the user's rule, 26/09/2026).

Columns per loss variant: pooled DSC (from test_results.json, the number papers report) and its
difference to the baseline of the same batch, mIoU, per-image mean DSC, DSC on the smallest and the
largest lesion tertile, how much bigger or smaller than the truth the small / large lesions are drawn
(geometric mean of predicted area / true area), oracle DSC on small lesions (DSC once the area is forced
to be right) and the best epoch. A variant passes to the confirmation round (E5) when its pooled DSC
beats the baseline by at least --min-gain (0.6 points = 2 x the retrain spread measured on 6 repeats)
on EVERY dataset of the batch.

Reads test_results.json and analysis/per_image_metrics_full.csv, both kept by the LEAN sync, so it
runs the same on the server and on the synced local folder.

  python analysis/exp06_summary.py --results-dir results --datasets "isic17 isic18" --out results/exp06_summary.md
'''
import os
import csv
import json
import math
import argparse

import numpy as np

# label, run-name pattern, what it is (plain words, for the table)
VARIANTS = [
    ('Baseline', 'egeunet_{ds}_learnable_s{seed}', 'luật hiện tại (BCE + Dice)'),
    ('E4', 'egeunet_{ds}_learnable_loss-area_s{seed}', '+ phạt sai diện tích'),
    ('E1b', 'egeunet_{ds}_learnable_loss-tvmatch_s{seed}', '+ viền dài đúng bằng viền thật'),
    ('E3a', 'egeunet_{ds}_learnable_loss-bl_s{seed}', '+ phạt vệt lem theo pixel'),
    ('E3b', 'egeunet_{ds}_learnable_loss-snbl_s{seed}', '+ phạt vệt lem theo cỡ tổn thương'),
    ('E1a', 'egeunet_{ds}_learnable_loss-tv_s{seed}', '+ viền càng ngắn càng tốt (ACL gốc)'),
    ('E2', 'egeunet_{ds}_learnable_loss-region_s{seed}', 'bỏ Dice'),
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--results-dir', default='results')
    p.add_argument('--datasets', default='isic17 isic18', help='space-separated')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--min-gain', type=float, default=0.6, help='pooled DSC points needed to pass')
    p.add_argument('--out', required=True, help='markdown path; a .csv with the same stem is written too')
    return p.parse_args()


def load_run(run_dir):
    tr = os.path.join(run_dir, 'test_results.json')
    pi = os.path.join(run_dir, 'analysis', 'per_image_metrics_full.csv')
    if not (os.path.isfile(tr) and os.path.isfile(pi)):
        return None
    res = json.load(open(tr))
    rows = list(csv.DictReader(open(pi)))
    col = lambda k: np.array([float(r[k]) for r in rows])
    area, tp, fp, fn, dsc, orc = col('area_frac'), col('tp'), col('fp'), col('fn'), col('dsc'), col('oracle_dsc')
    t = np.digitize(area, np.quantile(area, [1 / 3, 2 / 3]))
    ratio = np.log((tp + fp + 1) / (tp + fn + 1))
    pct = lambda m: 100 * (math.exp(ratio[m].mean()) - 1)
    return {
        'pooled_dsc': 100 * res['f1_or_dsc'], 'miou': 100 * res['miou'], 'best_epoch': res.get('min_epoch'),
        'dsc_img': 100 * dsc.mean(), 'dsc_small': 100 * dsc[t == 0].mean(), 'dsc_large': 100 * dsc[t == 2].mean(),
        'area_small_pct': pct(t == 0), 'area_large_pct': pct(t == 2),
        'oracle_small': 100 * orc[t == 0].mean(), 'n': len(rows),
    }


def fmt(x, d=2, sign=False):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return '—'
    s = f'{x:+.{d}f}' if sign else f'{x:.{d}f}'
    return s.replace('.', ',')


def main():
    a = parse_args()
    datasets = a.datasets.split()
    md = ['# EXP-6 — kết quả theo DSC pooled', '',
          f'Quyết thắng thua bằng **DSC pooled**. Một phiên bản được vào vòng xác nhận (E5) khi DSC pooled '
          f'cao hơn Baseline cùng đợt **ít nhất {fmt(a.min_gain, 1)} điểm ở mọi dataset**. Các cột còn lại chỉ để tham khảo.', '']
    long_rows, passed = [], {v[0]: [] for v in VARIANTS[1:]}
    for ds in datasets:
        runs = {lab: load_run(os.path.join(a.results_dir, pat.format(ds=ds, seed=a.seed)))
                for lab, pat, _ in VARIANTS}
        base = runs['Baseline']
        md += [f'## {ds.upper()}', '']
        if base is None:
            md += ['Baseline chưa có kết quả — bỏ qua dataset này.', '']
            continue
        md += ['| Phiên bản | Luật | DSC pooled | Δ so với Baseline | Qua ngưỡng? | mIoU | DSC theo ảnh | DSC nhỏ | DSC to '
               '| Tô to/nhỏ hơn thật: nhỏ · to | DSC nhỏ nếu đúng diện tích | best epoch |',
               '|---|---|---|---|---|---|---|---|---|---|---|---|']
        for lab, _, what in VARIANTS:
            r = runs[lab]
            if r is None:
                md.append(f'| {lab} | {what} | chưa có | | | | | | | | | |')
                continue
            delta = None if lab == 'Baseline' else r['pooled_dsc'] - base['pooled_dsc']
            ok = '' if delta is None else ('**có**' if delta >= a.min_gain else 'không')
            if delta is not None:
                passed[lab].append(delta >= a.min_gain)
            md.append(f'| {lab} | {what} | **{fmt(r["pooled_dsc"])}** | {fmt(delta, sign=True)} | {ok} | {fmt(r["miou"])} '
                      f'| {fmt(r["dsc_img"])} | {fmt(r["dsc_small"])} | {fmt(r["dsc_large"])} '
                      f'| {fmt(r["area_small_pct"], 1, True)} % · {fmt(r["area_large_pct"], 1, True)} % '
                      f'| {fmt(r["oracle_small"])} | {r["best_epoch"]} |')
            long_rows.append({'dataset': ds, 'variant': lab, **{k: r[k] for k in r},
                              'delta_pooled_dsc': delta})
        md.append('')
    md += ['## Vào vòng xác nhận E5?', '']
    for lab, _, what in VARIANTS[1:]:
        res = passed[lab]
        if len(res) < len(datasets):
            verdict = 'chưa đủ dữ liệu'
        else:
            verdict = '**CÓ** — qua ngưỡng ở mọi dataset' if all(res) else 'không'
        md.append(f'- {lab} ({what}): {verdict}')
    md += ['', 'Ghi chú: "Tô to/nhỏ hơn thật" là trung bình (hình học) của diện tích tô / diện tích thật trên nhóm; '
           'Baseline EXP-5 là +18,7 % · −9,0 % (ISIC17) và +9,7 % · −8,8 % (ISIC18). Nhóm nhỏ/to = một phần ba '
           'số ảnh có tổn thương nhỏ nhất / to nhất của từng dataset.', '']
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md))
    if long_rows:
        with open(os.path.splitext(a.out)[0] + '.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(long_rows[0].keys()))
            w.writeheader()
            w.writerows(long_rows)
    print('\n'.join(md))


if __name__ == '__main__':
    main()
