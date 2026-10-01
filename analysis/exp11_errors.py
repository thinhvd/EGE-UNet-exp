'''
EXP-11 error readouts: where the pixels a model gets wrong lie, for any set of runs, on the val or the train
split. Nothing here decides win / lose (that is pooled DSC, exp06_summary.py).

For every run, one forward pass per image; the probability map is thresholded at every value of
--thresholds. Errors are split by distance to the true contour as in analysis/exp09_mechanism.py (FP measured
outside the lesion, FN inside; near <= 3 px, mid 3-10 px, far > 10 px). "Interior miss" = FN far, i.e. lesion
pixels more than 10 px inside the true contour that the model does not paint.

  summary   at --threshold (0.5): pooled DSC, pooled and per-image precision / recall, FP, FP far, FN far,
            total errors, per group: all / small / mid / large (size tertiles of the split), every --focus set
            (e.g. the 21 / 36 heavy images of reports/large_interior_fn/), and the same groups without the
            --drop images.
  curve     per run and threshold: pooled DSC, FP total, FN far on the large tertile, total errors.
  vs-curve  (--curve-ref REF) for every other run: its 0.5 point (interior miss of the large tertile, FP total)
            against REF's threshold curve: the FP REF would need, by lowering or raising its threshold, to reach
            the same interior miss (linear interpolation), and the saving. A run that only shifts its decision
            along REF's curve saves ~0 %.
  tests     paired per-image tests of the error counts at 0.5 (every run against the first, plus --pair).
  noise     (--noise) mean, SD and SD of a difference between two runs (SD * sqrt 2) across the given runs,
            for the headline counts: meant for the repeated baseline runs of EXP-5..10 (step 0 of EXP-11).

  python analysis/exp11_errors.py --dataset isic17 --data-path data/data_isic1718/isic2017 \
      --run E5=results/EGE-UNet-results-exp5/egeunet_isic17_learnable_s42 ... --noise \
      --focus heavy=reports/large_interior_fn/danh_sach_isic17.csv --drop ISIC_0014979 --out-dir <dir>

Outputs in --out-dir: errors.md, summary.csv, curve.csv, vs_curve.csv, tests.csv, noise.csv, per_image.csv.
'''
import os
import sys
import csv
import math
import argparse

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis import common as C                                                    # noqa: E402
from analysis.exp09_mechanism import GroundTruth, pooled, error_tests, write_csv, vn, px, BINS  # noqa: E402
from datasets.dataset import NPY_datasets                                           # noqa: E402

TERTILES = ('small', 'mid', 'large')
DEFAULT_THRESHOLDS = ','.join(f'{t / 100:g}' for t in range(10, 91, 5))
NOISE_KEYS = (('large', 'fn_far'), ('all', 'fp'), ('all', 'fp_far'), ('small', 'fp'), ('large', 'fp'),
              ('large', 'fp_far'), ('all', 'err'), ('all', 'dsc'))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', action='append', required=True, metavar='LABEL=DIR')
    p.add_argument('--pair', action='append', default=[], metavar='VAR=REF')
    p.add_argument('--dataset', required=True, choices=['isic17', 'isic18'])
    p.add_argument('--data-path', required=True)
    p.add_argument('--split', default='val', choices=['val', 'train'],
                   help='train = the train images with the test transform (no augmentation)')
    p.add_argument('--focus', action='append', default=[], metavar='NAME=CSV',
                   help='an image set to report on its own: a csv with an "image" column (file name without extension)')
    p.add_argument('--drop', action='append', default=[], metavar='IMAGE',
                   help='also report every group without these images')
    p.add_argument('--threshold', type=float, default=0.5)
    p.add_argument('--thresholds', default=DEFAULT_THRESHOLDS)
    p.add_argument('--curve-ref', default=None, metavar='LABEL')
    p.add_argument('--noise', action='store_true')
    p.add_argument('--near', type=float, default=3.0)
    p.add_argument('--mid', type=float, default=10.0)
    p.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    p.add_argument('--max-images', type=int, default=None)
    p.add_argument('--out-dir', required=True)
    return p.parse_args()


def load_focus(specs, names):
    out = {}
    index = {n: i for i, n in enumerate(names)}
    for spec in specs:
        label, path = spec.split('=', 1)
        wanted = [r['image'] for r in csv.DictReader(open(path))]
        missing = [w for w in wanted if w not in index]
        if missing:
            sys.exit(f'--focus {label}: {len(missing)} image(s) not in this split, e.g. {missing[:3]}')
        out[label] = sorted(index[w] for w in wanted)
    return out


def groups_of(gt, focus, drop):
    g = {'all': list(range(gt.n))}
    for k, t in enumerate(TERTILES):
        g[t] = [i for i in range(gt.n) if gt.tertile[i] == k]
    g.update(focus)
    if drop:
        bad = {i for i, n in enumerate(gt.names) if n in drop}
        for name in list(g):
            g[f'{name} (no {"/".join(sorted(drop))})'] = [i for i in g[name] if i not in bad]
    return g


def headline(rows, idx):
    p = pooled(rows, idx)
    return {'n_images': len(idx), 'dsc': p['dsc'], 'precision': p['precision'], 'recall': p['recall'],
            'precision_img': p['precision_img'], 'recall_img': p['recall_img'], 'tp': p['tp'], 'fp': p['fp'],
            'fn': p['fn'], 'err': p['fp'] + p['fn'], **{f'fp_{b}': p[f'fp_{b}'] for b in BINS},
            **{f'fn_{b}': p[f'fn_{b}'] for b in BINS}}


def run_pass(model, ds, gt, device, thresholds):
    '''rows[t][i] = GroundTruth.counts of image i at threshold t.'''
    rows = {t: [] for t in thresholds}
    with torch.no_grad():
        for i in range(gt.n):
            img, _ = ds[i]
            _, out = model(img.float()[None].to(device))
            prob = out[0, 0].cpu().numpy()
            for t in thresholds:
                rows[t].append(gt.counts(i, prob >= t))
    return rows


def interp_fp(curve, x):
    '''FP of a threshold curve at interior miss x (linear in x between the two nearest points).'''
    pts = sorted((c['fn_far_large'], c['fp_all']) for c in curve)
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    if x <= xs[0] or x >= xs[-1]:
        return float('nan')
    return float(np.interp(x, xs, ys))


def main():
    a = parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    device = C.pick_device(a.device)
    config = C.make_config(a.dataset, 256)
    ds = NPY_datasets(a.data_path, config, train=(a.split == 'train'))
    ds.transformer = config.test_transformer
    n = len(ds) if a.max_images is None else min(a.max_images, len(ds))
    if a.focus and a.split != 'val':
        sys.exit('--focus sets are image lists of the val split')
    thresholds = sorted({round(float(t), 4) for t in a.thresholds.split(',')} | {a.threshold})
    print(f'{a.dataset} {a.split}: {n} images; thresholds {thresholds}')
    gt = GroundTruth(ds, n, a.near, a.mid)
    groups = groups_of(gt, load_focus(a.focus, gt.names), set(a.drop))

    runs, rows = [], {}
    for spec in a.run:
        label, path = spec.split('=', 1)
        ckpt = C.resolve_checkpoint(path)
        sd, _ = C.load_state_dict(ckpt)
        model, cfg = C.build_model(sd, device, None, strict=True)
        print(f'  {label}: {ckpt}')
        rows[label] = run_pass(model, ds, gt, device, thresholds)
        runs.append(label)

    t0 = a.threshold
    summ = [{'run': r, 'group': g, **headline(rows[r][t0], idx)} for r in runs for g, idx in groups.items() if idx]
    curve = []
    for r in runs:
        for t in thresholds:
            al, lg = headline(rows[r][t], groups['all']), headline(rows[r][t], groups['large'])
            curve.append({'run': r, 'threshold': t, 'dsc_all': al['dsc'], 'fp_all': al['fp'], 'fn_far_large': lg['fn_far'],
                          'err_all': al['err'], 'recall_all': al['recall'], 'precision_all': al['precision']})
    vs = []
    if a.curve_ref:
        if a.curve_ref not in rows:
            sys.exit(f'--curve-ref {a.curve_ref}: unknown run label')
        ref_curve = [c for c in curve if c['run'] == a.curve_ref]
        for r in runs:
            if r == a.curve_ref:
                continue
            al, lg = headline(rows[r][t0], groups['all']), headline(rows[r][t0], groups['large'])
            fp_ref = interp_fp(ref_curve, lg['fn_far'])
            vs.append({'run': r, 'reference': a.curve_ref, 'fn_far_large': lg['fn_far'], 'fp_all': al['fp'],
                       'fp_ref_at_same_miss': fp_ref,
                       'fp_saving_pct': 100 * (1 - al['fp'] / fp_ref) if fp_ref and math.isfinite(fp_ref) else float('nan')})
    pairs = [(r, runs[0]) for r in runs[1:]]
    for spec in a.pair:
        v, r = spec.split('=', 1)
        if v not in rows or r not in rows:
            sys.exit(f'--pair {spec}: unknown run label')
        if (v, r) not in pairs:
            pairs.append((v, r))
    tests = []
    for v, r in pairs:
        tests += error_tests(v, r, rows[v][t0], rows[r][t0], gt)
    noise = []
    if a.noise and len(runs) >= 2:
        by = {(s['run'], s['group']): s for s in summ}
        extra = [(g, k) for g in groups if g not in ('all', 'small', 'mid', 'large') for k in ('fn_far', 'fp', 'recall_img', 'precision_img')]
        for g, k in NOISE_KEYS + tuple(extra):
            v = np.array([by[(r, g)][k] for r in runs if (r, g) in by], dtype=np.float64)
            m, s = float(v.mean()), float(v.std(ddof=1))
            noise.append({'group': g, 'metric': k, 'n_runs': len(v), 'mean': m, 'sd': s,
                          'sd_pct_of_mean': 100 * s / abs(m) if m else float('nan'),
                          'sd_pair': s * math.sqrt(2), 'two_sd_pair_pct_of_mean': 200 * s * math.sqrt(2) / abs(m) if m else float('nan'),
                          'min': float(v.min()), 'max': float(v.max())})
    per_img = [{'run': r, 'image': gt.names[i], 'tertile': TERTILES[gt.tertile[i]], **rows[r][t0][i]}
               for r in runs for i in range(gt.n)]
    for name, data in (('summary', summ), ('curve', curve), ('vs_curve', vs), ('tests', tests), ('noise', noise),
                       ('per_image', per_img)):
        write_csv(os.path.join(a.out_dir, f'{name}.csv'), data)

    # ------------------------------------------------------------------ markdown
    md = [f'# EXP-11 — pixel sai theo vị trí ({a.dataset.upper()}, {a.split}, {n} ảnh, ngưỡng {vn(t0, 2)})', '',
          f'"Bỏ sót bên trong" = FN xa: pixel tổn thương nằm sâu hơn {a.mid:g} px bên trong viền thật mà model không tô. '
          f'FP xa = pixel tô thừa cách viền thật hơn {a.mid:g} px về phía ngoài. Nhóm nhỏ / vừa / to = ba phần ba theo diện '
          'tích tổn thương của split này.', '',
          '| Run | Nhóm | Số ảnh | DSC gộp | Precision / recall gộp | Precision / recall TB theo ảnh | FP | FP xa | Bỏ sót bên trong | Tổng pixel sai |',
          '|---|---|---|---|---|---|---|---|---|---|']
    for s in summ:
        md.append(f'| {s["run"]} | {s["group"]} | {s["n_images"]} | {vn(s["dsc"])} | {vn(s["precision"])} / {vn(s["recall"])} '
                  f'| {vn(s["precision_img"])} / {vn(s["recall_img"])} | {px(s["fp"])} | {px(s["fp_far"])} | {px(s["fn_far"])} '
                  f'| {px(s["err"])} |')
    md += ['', '## Đường cong ngưỡng (toàn tập; bỏ sót bên trong đo trên nhóm to)', '',
           '| Run | Ngưỡng | DSC gộp | FP | Bỏ sót bên trong (to) | Tổng pixel sai |', '|---|---|---|---|---|---|']
    md += [f'| {c["run"]} | {vn(c["threshold"], 2)} | {vn(c["dsc_all"])} | {px(c["fp_all"])} | {px(c["fn_far_large"])} '
           f'| {px(c["err_all"])} |' for c in curve]
    if vs:
        md += ['', f'## So với đường cong ngưỡng của {a.curve_ref}', '',
               f'Cột "FP {a.curve_ref} cần" = số FP mà {a.curve_ref} phải chịu (chỉ bằng cách đổi ngưỡng) để bỏ sót bên trong '
               'bằng đúng run này. Tiết kiệm ≈ 0 % nghĩa là run này chỉ trượt dọc đường cong của mốc.', '',
               f'| Run | Bỏ sót bên trong (to) | FP | FP {a.curve_ref} cần | Tiết kiệm FP |', '|---|---|---|---|---|']
        md += [f'| {v["run"]} | {px(v["fn_far_large"])} | {px(v["fp_all"])} '
               f'| {px(int(round(v["fp_ref_at_same_miss"]))) if math.isfinite(v["fp_ref_at_same_miss"]) else "ngoài đường cong"} '
               f'| {vn(v["fp_saving_pct"], 1)} % |' for v in vs]
    if noise:
        md += ['', f'## Dao động giữa {len(runs)} lần train (dùng làm thước đo nhiễu)', '',
               '"SD cặp" = độ lệch chuẩn của hiệu giữa hai lần train = SD × √2. Một khác biệt giữa hai run chỉ đáng tin khi vượt '
               '2 × SD cặp.', '',
               '| Nhóm | Chỉ số | TB | SD | SD / TB | 2 × SD cặp / TB | nhỏ nhất – lớn nhất |', '|---|---|---|---|---|---|---|']
        for r in noise:
            is_px = r['metric'] in ('fn_far', 'fp', 'fp_far', 'err')
            f = (lambda v: px(int(round(v)))) if is_px else (lambda v: vn(v))
            md.append(f'| {r["group"]} | {r["metric"]} | {f(r["mean"])} | {f(r["sd"])} | {vn(r["sd_pct_of_mean"], 1)} % '
                      f'| {vn(r["two_sd_pair_pct_of_mean"], 1)} % | {f(r["min"])} – {f(r["max"])} |')
    if tests:
        md += ['', '## Kiểm định cặp theo ảnh trên số pixel sai (dương = run nhiều lỗi hơn mốc)', '',
               '| Run | Mốc | Nhóm | Dải | Δ px/ảnh | t | p |', '|---|---|---|---|---|---|---|']
        md += [f'| {t["variant"]} | {t["reference"]} | {t["stratum"]} | {t["errors"]} | {vn(t["diff"], 1, True)} '
               f'| {vn(t["t_stat"], 2, True)} | {t["p_ttest"]:.4f} |' for t in tests]
    with open(os.path.join(a.out_dir, 'errors.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(md) + '\n')
    print(f'wrote {a.out_dir}/errors.md')
    return 0


if __name__ == '__main__':
    sys.exit(main())
