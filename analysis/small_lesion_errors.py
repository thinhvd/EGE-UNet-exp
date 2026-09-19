'''
Per-image error breakdown of several runs on one lesion-size stratum (default: the smallest tertile).

Answers "what goes wrong on small lesions": for every image in the stratum and every run it reports
TP / FP / FN, recall, precision and DSC, sorts each failure into an error type, measures how much of
the error sits right at the lesion boundary, and draws prediction vs ground truth.

Where the numbers come from. TP / FP / FN are read from each run's per-image evaluation csv
(analysis/eval_per_image.py), i.e. the official evaluation behind the reported results; they are
never recomputed here. The script cross-checks those csv files first (same images and ground truth in
every run, TP+FP+FN+TN = H*W, TP+FN = GT area, stored DSC = 2TP/(2TP+FP+FN)) and refuses to continue
on any mismatch. A fresh inference pass
is run only to get the prediction masks for the pictures and the boundary measure, and every such
mask is compared pixel-count by pixel-count against the official counts; disagreements are reported.

Error types (for images with DSC < --dsc-high), in this order:
  Type 1  not found     recall < --detect                          (almost no overlap with the lesion)
  Type 2  over-predict  recall >= --cover and precision < --clean  (covers the lesion, spills onto skin)
  Type 4  under-predict precision >= --clean and recall < --cover  (inside the lesion, but too small)
  Type 3  boundary      everything else                            (wrong on both sides of the contour)

Usage:
  python analysis/small_lesion_errors.py \
      --run learnable=results/.../egeunet_isic17_learnable_s42 --run csaa-deep3=results/... \
      --data-path data/data_isic1718/isic2017 --out-dir results/.../small_lesion_analysis
The first --run is the reference whose lesion areas define the strata (as in compare_runs.py).
Outputs: per_image.csv, summary.csv, <name>.xlsx (formulas; needs openpyxl), hist_dsc.png,
error_types.png, viz/<run>/DSC<x.xxx>_<image>.png
'''
import os
import sys
import csv
import argparse
from concurrent.futures import ProcessPoolExecutor

import matplotlib
matplotlib.use('Agg')
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis import common as C
from datasets.dataset import NPY_datasets

TYPE_GOOD = 'Tốt'
TYPE_1 = 'Kiểu 1 – Không tìm thấy'
TYPE_2 = 'Kiểu 2 – Tô tràn'
TYPE_3 = 'Kiểu 3 – Lệch biên'
TYPE_4 = 'Kiểu 4 – Tô thiếu'
ERROR_TYPES = [TYPE_1, TYPE_2, TYPE_3, TYPE_4]

# chart palette: validated for colour-vision deficiency (dataviz skill, validate_palette.js)
INK, INK_2, MUTED, GRID, SURFACE = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#fcfcfb'
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']          # type 1..4, fixed order
C_TP, C_FP, C_FN = '#1baf7a', '#eb6834', '#2a78d6'              # all-pairs safe triad

STRATA = {'small': 0, 'medium': 1, 'large': 2}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', action='append', required=True, metavar='LABEL=DIR',
                   help='run label and work_dir; repeatable. The first one defines the strata')
    p.add_argument('--csv-name', default='per_image_metrics_full.csv')
    p.add_argument('--data-path', required=True)
    p.add_argument('--dataset', default='isic17', choices=['isic17', 'isic18'])
    p.add_argument('--input-size', type=int, default=256)
    p.add_argument('--stratum', default='small', choices=list(STRATA))
    p.add_argument('--threshold', type=float, default=0.5, help='must match the official evaluation')
    p.add_argument('--dsc-low', type=float, default=0.5)
    p.add_argument('--dsc-high', type=float, default=0.85)
    p.add_argument('--detect', type=float, default=0.1)
    p.add_argument('--cover', type=float, default=0.8)
    p.add_argument('--clean', type=float, default=0.8)
    p.add_argument('--band', type=int, default=3, help='px: error within this distance of the GT contour '
                                                       'counts as "at the boundary"')
    p.add_argument('--out-dir', required=True)
    p.add_argument('--xlsx-name', default='phan_tich_loi_lesion_nho.xlsx')
    p.add_argument('--no-viz', action='store_true')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    return p.parse_args()


# ----------------------------------------------------------------------------- source checks

def load_runs(specs, csv_name):
    runs = []
    for spec in specs:
        if '=' not in spec:
            raise SystemExit(f'--run expects LABEL=DIR, got {spec!r}')
        label, d = spec.split('=', 1)
        path = os.path.join(d, 'analysis', csv_name)
        with open(path) as f:
            rows = list(csv.DictReader(f))
        runs.append((label, d, rows))
    return runs


def verify_source(runs, hw):
    '''Stop unless the official per-image csv files are internally consistent and agree on the GT.'''
    problems = []
    ref = runs[0][2]
    names = [r['filename'] for r in ref]
    for label, _, rows in runs:
        if [r['filename'] for r in rows] != names:
            problems.append(f'{label}: image list differs from {runs[0][0]}')
            continue
        for a, r in zip(ref, rows):
            tp, fp, fn, tn = (int(r[k]) for k in ('tp', 'fp', 'fn', 'tn'))
            if a['area_frac'] != r['area_frac']:
                problems.append(f'{label} {r["filename"]}: GT area differs from the reference')
            if tp + fp + fn + tn != hw:
                problems.append(f'{label} {r["filename"]}: counts do not sum to {hw}')
            if abs((tp + fn) - float(r['area_frac']) * hw) > 1e-6:
                problems.append(f'{label} {r["filename"]}: TP+FN != GT area')
            den = 2 * tp + fp + fn
            if (2 * tp / den if den else 1.0) != float(r['dsc']):
                problems.append(f'{label} {r["filename"]}: stored DSC != 2TP/(2TP+FP+FN)')
            if tp + fn == 0:
                problems.append(f'{label} {r["filename"]}: empty ground truth')
    if problems:
        raise SystemExit('source check failed:\n  ' + '\n  '.join(problems[:20]))


def stratum_indices(ref_rows, which):
    '''Tertiles of the reference lesion area, built exactly as compare_runs.py builds them.'''
    area = np.array([float(r['area_frac']) for r in ref_rows])
    edges = np.quantile(area, np.linspace(0, 1, 4)[1:-1])
    bins = np.digitize(area, edges)
    return np.nonzero(bins == STRATA[which])[0], edges


# ----------------------------------------------------------------------------- per-image metrics

def classify(dsc, recall, precision, a):
    if dsc >= a.dsc_high:
        return TYPE_GOOD
    if recall < a.detect:
        return TYPE_1
    if recall >= a.cover and precision < a.clean:
        return TYPE_2
    if precision >= a.clean and recall < a.cover:
        return TYPE_4
    return TYPE_3


def image_record(row, a, hw):
    tp, fp, fn = int(row['tp']), int(row['fp']), int(row['fn'])
    recall = tp / (tp + fn)
    precision = tp / (tp + fp) if tp + fp > 0 else float('nan')
    dsc = float(row['dsc'])
    return {
        'image': os.path.splitext(row['filename'])[0],
        'gt_px': tp + fn, 'gt_pct': 100.0 * (tp + fn) / hw, 'pred_px': tp + fp,
        'tp': tp, 'fp': fp, 'fn': fn,
        'recall': recall, 'precision': precision, 'dsc': dsc,
        'error_type': classify(dsc, recall, precision if tp + fp > 0 else 0.0, a),
    }


def boundary_shares(gt, pred, band):
    '''Share (%) of FP pixels within `band` px outside the GT contour and of FN pixels within `band`
    px inside it. None when there is no FP (resp. FN) pixel.'''
    fp = pred & ~gt
    fn = ~pred & gt
    d_out = distance_transform_edt(~gt)   # outside GT: distance to the nearest GT pixel
    d_in = distance_transform_edt(gt)     # inside GT: distance to the nearest non-GT pixel
    fp_near = 100.0 * float((fp & (d_out <= band)).sum()) / fp.sum() if fp.any() else None
    fn_near = 100.0 * float((fn & (d_in <= band)).sum()) / fn.sum() if fn.any() else None
    return fp_near, fn_near


def infer_masks(run_dir, ds, indices, device, threshold):
    ckpt = C.resolve_checkpoint(run_dir)
    sd, _ = C.load_state_dict(ckpt)
    model, _ = C.build_model(sd, device, None, strict=True)
    out = {}
    with torch.no_grad():
        for i in indices:
            img, msk = ds[i]
            _, pred = model(img.float()[None].to(device))
            prob = pred[0, 0].cpu().numpy()
            out[i] = (prob >= threshold, msk.float()[0].numpy() >= 0.5)
    return out


# ----------------------------------------------------------------------------- pictures

def vn(x, d=3):
    return f'{x:.{d}f}'.replace('.', ',')


def render_one(task):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    img, gt, pred, rec, label, path = task
    tp, fp, fn = pred & gt, pred & ~gt, ~pred & gt
    drawn = (int(tp.sum()), int(fp.sum()), int(fn.sum()))
    official = (rec['tp'], rec['fp'], rec['fn'])

    gray = np.dot(img[..., :3].astype(np.float32), [0.299, 0.587, 0.114])
    base = np.repeat((0.35 + 0.65 * gray / 255.0)[..., None], 3, axis=2)
    over = base.copy()
    for m, c in ((tp, C_TP), (fp, C_FP), (fn, C_FN)):
        rgb = np.array(matplotlib.colors.to_rgb(c))
        over[m] = 0.15 * over[m] + 0.85 * rgb

    ys, xs = np.nonzero(gt)
    cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
    side = max(ys.max() - ys.min(), xs.max() - xs.min()) + 1
    half = int(max(16, np.ceil(side * 0.9)))
    y0, y1 = int(max(0, cy - half)), int(min(gt.shape[0], cy + half + 1))
    x0, x1 = int(max(0, cx - half)), int(min(gt.shape[1], cx + half + 1))

    fig, axes = plt.subplots(1, 5, figsize=(12.5, 3.35), facecolor=SURFACE)
    panels = [(img, 'Ảnh gốc'), (gt, 'Ground truth'), (pred, 'Dự đoán'),
              (over, 'Chồng lớp (toàn ảnh)'), (over[y0:y1, x0:x1], 'Phóng to vùng tổn thương')]
    for ax, (im, title) in zip(axes, panels):
        if im.ndim == 2:
            ax.imshow(im, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
        else:
            ax.imshow(im, interpolation='nearest')
        ax.set_title(title, fontsize=9, color=INK_2)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(GRID)
    axes[3].add_patch(matplotlib.patches.Rectangle((x0 - 0.5, y0 - 0.5), x1 - x0, y1 - y0, fill=False,
                                                   edgecolor=INK, linewidth=0.8, linestyle='--'))
    prec = '—' if np.isnan(rec['precision']) else vn(rec['precision'])
    fig.suptitle(f'{rec["image"]}   |   {label}   |   GT {vn(rec["gt_pct"], 2)}% ảnh   |   '
                 f'DSC {vn(rec["dsc"])}   Recall {vn(rec["recall"])}   Precision {prec}   |   '
                 f'{rec["error_type"]}', fontsize=10, color=INK, y=0.99)
    # the legend counts what is drawn; where the fresh mask differs from the official evaluation by
    # a few threshold-edge pixels, the official count is shown next to it
    names = ('TP (tô đúng)', 'FP (tô thừa)', 'FN (bỏ sót)')
    handles = []
    for name, c, d, o in zip(names, (C_TP, C_FP, C_FN), drawn, official):
        text = f'{name} {d} px' + (f'  [chính thức: {o}]' if d != o else '')
        handles.append(Patch(color=c, label=text))
    fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 0.05))
    if drawn != official:
        fig.text(0.5, 0.008, 'Mask vẽ lại trên CPU lệch vài pixel sát ngưỡng 0,5 so với lượt đánh giá '
                             'chính thức (GPU). Số ở tiêu đề là số chính thức.',
                 ha='center', va='bottom', fontsize=7.5, color=MUTED)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.84, bottom=0.18, wspace=0.05)
    fig.savefig(path, dpi=90, facecolor=SURFACE)
    plt.close(fig)
    return path


def plot_histograms(records, labels, a, path):
    import matplotlib.pyplot as plt
    n = len(labels)
    cols = 4
    rows_ = int(np.ceil(n / cols))
    bins = np.linspace(0, 1, 21)
    counts = {lab: np.histogram([r['dsc'] for r in records[lab]], bins=bins)[0] for lab in labels}
    ymax = max(c.max() for c in counts.values())
    fig, axes = plt.subplots(rows_, cols, figsize=(14, 3.3 * rows_), sharex=True, sharey=True,
                             facecolor=SURFACE)
    axes = np.atleast_1d(axes).ravel()
    for ax, lab in zip(axes, labels):
        d = np.array([r['dsc'] for r in records[lab]])
        ax.bar(bins[:-1], counts[lab], width=0.05, align='edge', color=SERIES[0],
               edgecolor=SURFACE, linewidth=1.2, zorder=3)
        for x in (a.dsc_low, a.dsc_high):
            ax.axvline(x, color=MUTED, linestyle='--', linewidth=1, zorder=4)
        low, high = int((d < a.dsc_low).sum()), int((d >= a.dsc_high).sum())
        ax.text(0.02, 0.97, f'DSC < {vn(a.dsc_low, 2)}: {low} ảnh\nDSC ≥ {vn(a.dsc_high, 2)}: {high} ảnh\n'
                            f'DSC trung bình: {vn(d.mean())}',
                transform=ax.transAxes, ha='left', va='top', fontsize=8.5, color=INK)
        ax.set_title(lab, fontsize=11, color=INK)
        ax.set_ylim(0, ymax * 1.08)
        ax.grid(axis='y', color=GRID, linewidth=0.6, zorder=0)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        for s in ('left', 'bottom'):
            ax.spines[s].set_color(MUTED)
        ax.tick_params(colors=INK_2, labelsize=8)
    for ax in axes[n:]:
        ax.set_visible(False)
    for ax in axes[(rows_ - 1) * cols:]:
        ax.set_xlabel('DSC của từng ảnh', fontsize=9, color=INK_2)
    for r in range(rows_):
        axes[r * cols].set_ylabel('Số ảnh', fontsize=9, color=INK_2)
    fig.suptitle(f'Phân bố DSC của từng ảnh trên nhóm tổn thương nhỏ ({len(records[labels[0]])} ảnh) — '
                 f'vạch đứt: ngưỡng {vn(a.dsc_low, 2)} và {vn(a.dsc_high, 2)}', fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_error_types(records, labels, path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 4.6), facecolor=SURFACE)
    x = np.arange(len(labels))
    bottom = np.zeros(len(labels))
    for t, color in zip(ERROR_TYPES, SERIES):
        vals = np.array([sum(r['error_type'] == t for r in records[lab]) for lab in labels], dtype=float)
        ax.bar(x, vals, bottom=bottom, width=0.62, color=color, label=t, edgecolor=SURFACE,
               linewidth=2, zorder=3)
        for xi, (v, b) in enumerate(zip(vals, bottom)):
            if v >= 3:          # thinner segments cannot hold a legible label; the workbook has every count
                ax.text(xi, b + v / 2, f'{int(v)}', ha='center', va='center', fontsize=9, color=INK)
        bottom += vals
    for xi, tot in enumerate(bottom):
        ax.text(xi, tot + 0.8, f'{int(tot)}', ha='center', va='bottom', fontsize=10, color=INK,
                fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5, color=INK)
    ax.set_ylabel('Số ảnh có DSC < 0,85', fontsize=10, color=INK_2)
    ax.set_ylim(0, bottom.max() * 1.15)
    ax.grid(axis='y', color=GRID, linewidth=0.6, zorder=0)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK_2)
    ax.set_title('Các ảnh tổn thương nhỏ bị lỗi (DSC < 0,85), phân theo kiểu lỗi', fontsize=12, color=INK)
    ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=9.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


# ----------------------------------------------------------------------------- tables

PER_IMAGE_FIELDS = ['model', 'image', 'gt_px', 'gt_pct', 'pred_px', 'tp', 'fp', 'fn', 'recall',
                    'precision', 'dsc', 'error_type', 'fp_near_boundary_pct', 'fn_near_boundary_pct',
                    'matches_official']


def summarise(recs, a):
    d = np.array([r['dsc'] for r in recs])
    p = np.array([r['precision'] for r in recs])
    out = {
        'n_images': len(recs),
        'mean_dsc': float(d.mean()),
        'mean_recall': float(np.mean([r['recall'] for r in recs])),
        'mean_precision': float(np.nanmean(p)),
        'median_dsc': float(np.median(d)),
        'n_dsc_low': int((d < a.dsc_low).sum()),
        'n_dsc_mid': int(((d >= a.dsc_low) & (d < a.dsc_high)).sum()),
        'n_dsc_high': int((d >= a.dsc_high).sum()),
        'n_empty_prediction': int(np.isnan(p).sum()),
    }
    for i, t in enumerate(ERROR_TYPES, 1):
        out[f'n_type{i}'] = sum(r['error_type'] == t for r in recs)
    t2 = [r['fp_near_boundary_pct'] for r in recs if r['error_type'] == TYPE_2
          and r['fp_near_boundary_pct'] is not None]
    out['type2_median_fp_near_pct'] = float(np.median(t2)) if t2 else float('nan')
    return out


def write_csv(path, fields, rows):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: ('' if r.get(k) is None or (isinstance(r.get(k), float) and np.isnan(r[k]))
                            else r[k]) for k in fields})


def main():
    a = parse_args()
    hw = a.input_size * a.input_size
    os.makedirs(a.out_dir, exist_ok=True)

    runs = load_runs(a.run, a.csv_name)
    verify_source(runs, hw)
    idx, edges = stratum_indices(runs[0][2], a.stratum)
    labels = [lab for lab, _, _ in runs]
    print(f'source check passed for {len(runs)} runs; stratum "{a.stratum}": {idx.size} images '
          f'(area_frac edges {edges[0]:.6f} / {edges[1]:.6f})')

    records = {lab: [image_record(rows[i], a, hw) for i in idx] for lab, _, rows in runs}

    # the masks are needed even without pictures: the boundary measure is computed on them
    config = C.make_config(a.dataset, a.input_size)
    ds = NPY_datasets(a.data_path, config, train=False)
    if [os.path.basename(p[0]) for p in ds.data] != [r['filename'] for r in runs[0][2]]:
        raise SystemExit('the dataset order does not match the evaluation csv')

    tasks, mismatches, checks = [], {}, []
    for lab, run_dir, rows in runs:
        masks = infer_masks(run_dir, ds, idx, a.device, a.threshold)
        mismatches[lab] = 0
        vdir = os.path.join(a.out_dir, 'viz', lab)
        if not a.no_viz:
            os.makedirs(vdir, exist_ok=True)
        for rec, i in zip(records[lab], idx):
            pred, gt = masks[i]
            tp, fp, fn, _ = C.binary_counts(pred, gt)
            rec['matches_official'] = (tp, fp, fn) == (rec['tp'], rec['fp'], rec['fn'])
            if not rec['matches_official']:
                mismatches[lab] += 1
                cpu = image_record({'filename': rec['image'], 'tp': tp, 'fp': fp, 'fn': fn,
                                    'dsc': 2 * tp / (2 * tp + fp + fn)}, a, hw)
                checks.append({'model': lab, 'image': rec['image'],
                               'official_tp': rec['tp'], 'official_fp': rec['fp'], 'official_fn': rec['fn'],
                               'cpu_tp': tp, 'cpu_fp': fp, 'cpu_fn': fn,
                               'gt_px_equal': (tp + fn) == rec['gt_px'],
                               'pred_px_diff': (tp + fp) - rec['pred_px'],
                               'official_dsc': rec['dsc'], 'cpu_dsc': cpu['dsc'],
                               'abs_dsc_diff': abs(cpu['dsc'] - rec['dsc']),
                               'official_type': rec['error_type'], 'cpu_type': cpu['error_type'],
                               'dsc_bucket_changes': ((rec['dsc'] < a.dsc_low) != (cpu['dsc'] < a.dsc_low))
                               or ((rec['dsc'] >= a.dsc_high) != (cpu['dsc'] >= a.dsc_high))})
            rec['fp_near_boundary_pct'], rec['fn_near_boundary_pct'] = boundary_shares(gt, pred, a.band)
            if not a.no_viz:
                img = np.array(Image.open(ds.data[i][0]).convert('RGB').resize(
                    (a.input_size, a.input_size), Image.BILINEAR))
                tasks.append((img, gt, pred, dict(rec), lab,
                              os.path.join(vdir, f'DSC{rec["dsc"]:.3f}_{rec["image"]}.png')))
        print(f'{lab:14s} inference done; masks that differ from the official counts: {mismatches[lab]}')

    check_fields = ['model', 'image', 'official_tp', 'official_fp', 'official_fn', 'cpu_tp', 'cpu_fp',
                    'cpu_fn', 'gt_px_equal', 'pred_px_diff', 'official_dsc', 'cpu_dsc', 'abs_dsc_diff',
                    'official_type', 'cpu_type', 'dsc_bucket_changes']
    write_csv(os.path.join(a.out_dir, 'kiem_tra_mask_cpu_vs_chinh_thuc.csv'), check_fields, checks)
    cpu_check = {
        'n_pairs': idx.size * len(labels), 'n_differ': len(checks),
        'gt_all_equal': all(c['gt_px_equal'] for c in checks),
        'max_px': max((abs(c['pred_px_diff']) for c in checks), default=0),
        'median_px': float(np.median([abs(c['pred_px_diff']) for c in checks])) if checks else 0.0,
        'max_dsc': max((c['abs_dsc_diff'] for c in checks), default=0.0),
        'type_changes': sum(c['official_type'] != c['cpu_type'] for c in checks),
        'bucket_changes': sum(bool(c['dsc_bucket_changes']) for c in checks),
    }
    print('fresh-inference check: {n_differ}/{n_pairs} pairs differ; GT identical everywhere: {gt_all_equal}; '
          'predicted-area diff max {max_px} px (median {median_px:g}); max |dDSC| {max_dsc:.5f}; '
          'error-type changes {type_changes}; DSC-bucket changes {bucket_changes}'.format(**cpu_check))

    per_image = [dict(r, model=lab) for lab in labels for r in records[lab]]
    write_csv(os.path.join(a.out_dir, 'per_image.csv'), PER_IMAGE_FIELDS, per_image)
    summary = [dict(summarise(records[lab], a), model=lab) for lab in labels]
    sfields = ['model'] + [k for k in summary[0] if k != 'model']
    write_csv(os.path.join(a.out_dir, 'summary.csv'), sfields, summary)

    plot_histograms(records, labels, a, os.path.join(a.out_dir, 'hist_dsc.png'))
    plot_error_types(records, labels, os.path.join(a.out_dir, 'error_types.png'))

    if not a.no_viz:
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            for k, _ in enumerate(ex.map(render_one, tasks, chunksize=8), 1):
                if k % 200 == 0:
                    print(f'  pictures {k}/{len(tasks)}')

    try:
        from analysis.small_lesion_xlsx import write_workbook
        write_workbook(os.path.join(a.out_dir, a.xlsx_name), labels, records, edges, idx.size, a, hw,
                       cpu_check)
    except ImportError as e:
        print(f'[warn] xlsx skipped ({e}); install openpyxl to get the workbook')

    print(f'\nsummary ({a.stratum} stratum, {idx.size} images)')
    print(f'{"model":14s} {"DSC":>7s} {"Recall":>7s} {"Prec.":>7s} {"<0.5":>5s} {">=.85":>6s} '
          f'{"T1":>4s} {"T2":>4s} {"T3":>4s} {"T4":>4s}  {"T2 FP@edge%":>11s}')
    for s in summary:
        print(f'{s["model"]:14s} {s["mean_dsc"]:7.4f} {s["mean_recall"]:7.4f} {s["mean_precision"]:7.4f} '
              f'{s["n_dsc_low"]:5d} {s["n_dsc_high"]:6d} {s["n_type1"]:4d} {s["n_type2"]:4d} '
              f'{s["n_type3"]:4d} {s["n_type4"]:4d}  {s["type2_median_fp_near_pct"]:11.1f}')
    print(f'wrote {a.out_dir}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
