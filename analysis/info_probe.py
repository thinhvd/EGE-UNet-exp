'''
EXP-12: where does the evidence for the missed pale rim of large lesions live? Read-only diagnostic, no training.

Hard positives H (as in EXP-11): lesion pixels (GT = 1) more than 10 px inside the true contour that the model
leaves below probability 0.5. Negatives: the background band 10-30 px outside the contour ("band", the skin an
annotator compares the rim with) and the background beyond 30 px ("far"). Reference: the confident core (lesion,
> 10 px inside, probability >= 0.9). Every set is defined per model, from that model's own prediction.

Part A, probes. For every pixel of a fixed per-image sample, features are read at several levels:
  raw-pixel    RGB, CIELAB, HSV of the pixel (9)
  raw-context  raw-pixel + mean / std of L*, a*, b* over 7 / 15 / 31 px windows (27)
  E1..E6       encoder features t1..t6 of the model (8 / 16 / 24 / 32 / 48 / 64 channels, grids 128 .. 8)
  Dec1..Dec5   decoder features after the GAB (and fusion) add (48 / 32 / 24 / 16 / 8 channels, grids 8 .. 128);
               Dec5 is the input of the final 1x1 conv
  enc-all, dec-all, all   concatenations
  prob         the model's own probability (reference, no probe)
Feature maps are sampled at the pixel's position with bilinear interpolation (align_corners=True, as the network
upsamples). Probes (standardized logistic regression and a gradient-boosted tree model) are FITTED ON THE TRAIN
IMAGES (H_train vs band_train, test transform) and SCORED ON THE TEST IMAGES: pooled AUC of H vs band and of H vs
far, and the per-image AUC of H vs band (median over test images with enough pixels of both, all and heavy set).

Part B, image-based visibility of the rim (no model fitting), per test image with enough H and core pixels:
  t      position of the mean CIELAB colour of H on the axis from the band (0) to the core (1)
  dE     CIELAB distances band-H and band-core
  axis AUC  AUC of H vs band when pixels are projected on that band->core axis
  edges  gradient magnitude of the smoothed CIELAB image on the GT contour and on the inner contour of the
         prediction (where it runs > 5 px inside the lesion), each divided by the median over the band
Part C, sheets for the heavy images: image with GT (yellow) and the model's contour (cyan), contrast-stretched L*,
CIELAB distance to the band colour, gradient magnitude.

  python analysis/info_probe.py --dataset isic17 --data-path data/data_isic1718/isic2017 \
      --run D0=results/EGE-UNet-results-exp11/egeunet_isic17_learnable_s42 --run D1=... \
      --heavy analysis/lists/exp11_heavy_isic17.csv --out-dir results/exp12_info/isic17
'''
import os
import sys
import csv
import json
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt, uniform_filter, gaussian_filter, sobel, binary_erosion
from scipy.stats import rankdata

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis import common as C                                   # noqa: E402
from datasets.dataset import NPY_datasets                          # noqa: E402

ENC = ['E1', 'E2', 'E3', 'E4', 'E5', 'E6']
DEC = ['Dec1', 'Dec2', 'Dec3', 'Dec4', 'Dec5']
HOOKS = [('E1', 'encoder2'), ('E2', 'encoder3'), ('E3', 'encoder4'), ('E4', 'encoder5'), ('E5', 'encoder6'),
         ('E6', 'decoder1'), ('Dec1', 'decoder2'), ('Dec2', 'decoder3'), ('Dec3', 'decoder4'), ('Dec4', 'decoder5'),
         ('Dec5', 'final')]
SETS = ('H', 'band', 'far', 'core')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', action='append', required=True, metavar='LABEL=DIR')
    p.add_argument('--dataset', required=True, choices=['isic17', 'isic18'])
    p.add_argument('--data-path', required=True)
    p.add_argument('--heavy', default=None, help='csv with an "image" column (test images)')
    p.add_argument('--cap-train', type=int, default=100, help='max sampled px per image and set, train split')
    p.add_argument('--cap-test', type=int, default=300, help='max sampled px per image for H and band, test split')
    p.add_argument('--cap-test-other', type=int, default=60, help='max sampled px per image for far and core, test split')
    p.add_argument('--min-img', type=int, default=30, help='min H and band samples for a per-image AUC')
    p.add_argument('--hgb-max', type=int, default=200000, help='max train samples for the tree model')
    p.add_argument('--max-images', type=int, default=None)
    p.add_argument('--no-sheets', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out-dir', required=True)
    return p.parse_args()


# ----------------------------------------------------------------------------- colour

def rgb_to_lab(rgb):
    '''sRGB uint8 (H,W,3) -> CIELAB (D65), float32.'''
    c = rgb.astype(np.float64) / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    m = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]])
    xyz = c @ m.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    lab = np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)
    return lab.astype(np.float32)


def rgb_to_hsv(rgb):
    c = rgb.astype(np.float32) / 255.0
    mx, mn = c.max(-1), c.min(-1)
    d = mx - mn
    h = np.zeros_like(mx)
    r, g, b = c[..., 0], c[..., 1], c[..., 2]
    nz = d > 1e-6
    h = np.where(nz & (mx == r), ((g - b) / np.where(nz, d, 1)) % 6, h)
    h = np.where(nz & (mx == g), (b - r) / np.where(nz, d, 1) + 2, h)
    h = np.where(nz & (mx == b), (r - g) / np.where(nz, d, 1) + 4, h)
    s = np.where(mx > 1e-6, d / np.where(mx > 1e-6, mx, 1), 0)
    return np.stack([h / 6, s, mx], -1)


def raw_maps(rgb):
    '''(H,W,27) raw-context features and the CIELAB image.'''
    lab = rgb_to_lab(rgb)
    feats = [rgb.astype(np.float32) / 255.0, lab, rgb_to_hsv(rgb).astype(np.float32)]
    for w in (7, 15, 31):
        mu = np.stack([uniform_filter(lab[..., k], w, mode='reflect') for k in range(3)], -1)
        sq = np.stack([uniform_filter(lab[..., k] ** 2, w, mode='reflect') for k in range(3)], -1)
        feats += [mu, np.sqrt(np.maximum(sq - mu ** 2, 0))]
    return np.concatenate(feats, -1), lab


# ----------------------------------------------------------------------------- sampling

def pixel_sets(gt, prob):
    d_in = distance_transform_edt(gt) if gt.any() else np.zeros(gt.shape)
    d_out = distance_transform_edt(~gt) if gt.any() else np.full(gt.shape, np.inf)
    return {'H': gt & (d_in > 10) & (prob < 0.5), 'band': ~gt & (d_out > 10) & (d_out <= 30),
            'far': ~gt & (d_out > 30), 'core': gt & (d_in > 10) & (prob >= 0.9)}


def sample(mask, cap, rng):
    ys, xs = np.nonzero(mask)
    if len(ys) > cap:
        k = rng.choice(len(ys), cap, replace=False)
        ys, xs = ys[k], xs[k]
    return ys, xs


def read_maps(feats, ys, xs, size):
    '''Bilinear read (align_corners=True) of every hooked map at pixel positions of a size x size image.'''
    gy = torch.from_numpy(2 * ys / (size - 1) - 1).float()
    gx = torch.from_numpy(2 * xs / (size - 1) - 1).float()
    grid = torch.stack([gx, gy], -1).view(1, 1, -1, 2)
    out = {}
    for name, fm in feats.items():
        out[name] = F.grid_sample(fm, grid, mode='bilinear', align_corners=True)[0, :, 0].T.numpy().astype(np.float32)
    return out


def extract(model, ds, n, caps, seed):
    feats = {}
    hooks = [getattr(model, mod).register_forward_pre_hook(
        (lambda name: lambda m, inp: feats.__setitem__(name, inp[0].detach()))(name)) for name, mod in HOOKS]
    rows = {k: [] for k in ['raw'] + [h for h, _ in HOOKS]}
    meta = {'set': [], 'img': [], 'prob': []}
    with torch.no_grad():
        for i in range(n):
            img_t, msk = ds[i]
            _, out = model(img_t.float()[None])
            prob = out[0, 0].numpy()
            gt = msk.float()[0].numpy() >= 0.5
            sets = pixel_sets(gt, prob)
            rng = np.random.default_rng(seed * 100003 + i)
            # computed per image and dropped: caching the 27-channel maps of every image costs ~13 GB on ISIC18
            from PIL import Image
            raw = raw_maps(np.array(Image.open(ds.data[i][0]).convert('RGB')))[0]
            for s_id, s in enumerate(SETS):
                ys, xs = sample(sets[s], caps[s], rng)
                if len(ys) == 0:
                    continue
                vals = read_maps(feats, ys, xs, gt.shape[0])
                for name in vals:
                    rows[name].append(vals[name])
                rows['raw'].append(raw[ys, xs])
                meta['set'].append(np.full(len(ys), s_id, np.int8))
                meta['img'].append(np.full(len(ys), i, np.int32))
                meta['prob'].append(prob[ys, xs].astype(np.float32))
    for h in hooks:
        h.remove()
    return {k: np.concatenate(v) for k, v in rows.items()}, {k: np.concatenate(v) for k, v in meta.items()}


# ----------------------------------------------------------------------------- probes

def auc(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def level_matrix(X, level):
    if level == 'raw-pixel':
        return X['raw'][:, :9]
    if level == 'raw-context':
        return X['raw']
    if level == 'enc-all':
        return np.concatenate([X[k] for k in ENC], 1)
    if level == 'dec-all':
        return np.concatenate([X[k] for k in DEC], 1)
    if level == 'all':
        return np.concatenate([X['raw']] + [X[k] for k in ENC + DEC], 1)
    return X[level]


def score_test(scores, meta, heavy_idx, min_img):
    s = meta['set']
    H, band, far = s == 0, s == 1, s == 2
    res = {'auc_band': auc(scores[H], scores[band]), 'auc_far': auc(scores[H], scores[far])}
    per = {}
    for i in np.unique(meta['img'][H]):
        mi = meta['img'] == i
        if (mi & H).sum() >= min_img and (mi & band).sum() >= min_img:
            per[int(i)] = auc(scores[mi & H], scores[mi & band])
    v = np.array(list(per.values()))
    vh = np.array([a for i, a in per.items() if i in heavy_idx])
    res.update({'img_n': len(v), 'img_median': float(np.median(v)) if len(v) else float('nan'),
                'img_frac_ge_0.8': float((v >= 0.8).mean()) if len(v) else float('nan'),
                'heavy_n': len(vh), 'heavy_median': float(np.median(vh)) if len(vh) else float('nan')})
    return res, per


def run_probes(Xtr, Mtr, Xte, Mte, heavy_idx, a):
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    tr = (Mtr['set'] == 0) | (Mtr['set'] == 1)
    y = (Mtr['set'][tr] == 0).astype(int)
    rng = np.random.default_rng(a.seed)
    sub = np.flatnonzero(tr)
    if len(sub) > a.hgb_max:
        sub_h = np.sort(rng.choice(sub, a.hgb_max, replace=False))
    else:
        sub_h = sub
    yh = (Mtr['set'][sub_h] == 0).astype(int)
    out, per_img = [], {}
    res, per = score_test(Mte['prob'], Mte, heavy_idx, a.min_img)
    out.append({'level': 'prob', 'dims': 1, 'probe': 'none', 'n_train_H': int(y.sum()), 'n_train_band': int((1 - y).sum()), **res})
    per_img[('prob', 'none')] = per
    for level in ['raw-pixel', 'raw-context'] + ENC + DEC + ['enc-all', 'dec-all', 'all']:
        Atr, Ate = level_matrix(Xtr, level), level_matrix(Xte, level)
        sc = StandardScaler().fit(Atr[tr])
        lr = LogisticRegression(max_iter=3000, class_weight='balanced').fit(sc.transform(Atr[tr]), y)
        s_lr = lr.decision_function(sc.transform(Ate))
        hgb = HistGradientBoostingClassifier(max_iter=300, class_weight='balanced', random_state=a.seed).fit(Atr[sub_h], yh)
        s_hgb = hgb.predict_proba(Ate)[:, 1]
        for name, s in (('logistic', s_lr), ('trees', s_hgb)):
            res, per = score_test(s, Mte, heavy_idx, a.min_img)
            out.append({'level': level, 'dims': Atr.shape[1], 'probe': name, 'n_train_H': int(y.sum()),
                        'n_train_band': int((1 - y).sum()), **res})
            per_img[(level, name)] = per
        print(f'    {level:11s} dims {Atr.shape[1]:3d}  AUC H vs band: logistic {out[-2]["auc_band"]:.3f} trees {out[-1]["auc_band"]:.3f} '
              f'| per-image median {out[-1]["img_median"]:.3f} (heavy {out[-1]["heavy_median"]:.3f})')
    return out, per_img


# ----------------------------------------------------------------------------- visibility and sheets

def visibility(model, ds, n, heavy_idx, sheets_dir, label):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    rows = []
    with torch.no_grad():
        for i in range(n):
            img_t, msk = ds[i]
            gt = msk.float()[0].numpy() >= 0.5
            if not gt.any() or gt.all():
                continue
            prob = model(img_t.float()[None])[1][0, 0].numpy()
            sets = pixel_sets(gt, prob)
            if sets['H'].sum() < 50 or sets['core'].sum() < 50 or sets['band'].sum() < 50:
                continue
            rgb = np.array(Image.open(ds.data[i][0]).convert('RGB'))
            lab = rgb_to_lab(rgb)
            h, b, c = (lab[sets[k]].mean(0) for k in ('H', 'band', 'core'))
            axis = c - b
            t = float(np.dot(h - b, axis) / max(np.dot(axis, axis), 1e-6))
            proj = lab @ axis
            smooth = np.stack([gaussian_filter(lab[..., k], 1.5) for k in range(3)], -1)
            grad = sum(np.hypot(sobel(smooth[..., k], 0), sobel(smooth[..., k], 1)) for k in range(3))
            g_band = float(np.median(grad[sets['band']]))
            gt_edge = gt & ~binary_erosion(gt)
            pred = prob >= 0.5
            d_in = distance_transform_edt(gt)
            inner = pred & ~binary_erosion(pred) & (d_in > 5)
            row = {'image': os.path.splitext(os.path.basename(ds.data[i][0]))[0], 'heavy': int(i in heavy_idx),
                   'H_px': int(sets['H'].sum()), 't': t, 'dE_band_H': float(np.linalg.norm(h - b)),
                   'dE_band_core': float(np.linalg.norm(c - b)), 'axis_auc': auc(proj[sets['H']], proj[sets['band']]),
                   'edge_gt_over_band': float(np.median(grad[gt_edge]) / max(g_band, 1e-6)),
                   'edge_inner_over_band': float(np.median(grad[inner]) / max(g_band, 1e-6)) if inner.sum() >= 20 else float('nan')}
            rows.append(row)
            if sheets_dir and i in heavy_idx:
                fig, ax = plt.subplots(1, 4, figsize=(16, 4.6))
                L = lab[..., 0]
                lo, hi = np.percentile(L, [1, 99])
                panels = [(rgb, None, 'Ảnh gốc'), (np.clip((L - lo) / max(hi - lo, 1e-6), 0, 1), 'gray', 'L* (kéo giãn tương phản)'),
                          (np.clip(np.linalg.norm(lab - b, axis=-1), 0, 30), 'magma', 'ΔE tới màu da (band), 0–30'),
                          (np.clip(grad / max(np.percentile(grad, 99), 1e-6), 0, 1), 'viridis', 'Độ mạnh cạnh (gradient)')]
                for k, (im, cmap, title) in enumerate(panels):
                    ax[k].imshow(im, cmap=cmap)
                    ax[k].contour(gt.astype(float), levels=[0.5], colors=['#ffd400'], linewidths=1.0)
                    ax[k].contour(pred.astype(float), levels=[0.5], colors=['#00e5ff'], linewidths=1.0)
                    ax[k].set_title(title, fontsize=10)
                    ax[k].axis('off')
                fig.suptitle(f'{row["image"]} · vàng = GT boundary, xanh = boundary {label} dự đoán · t = {t:.2f} · '
                             f'ΔE band–H {row["dE_band_H"]:.1f}, band–core {row["dE_band_core"]:.1f} · axis AUC {row["axis_auc"]:.2f} · '
                             f'cạnh tại GT / band {row["edge_gt_over_band"]:.2f}, tại mép lõi / band {row["edge_inner_over_band"]:.2f}', fontsize=9)
                fig.tight_layout()
                fig.savefig(os.path.join(sheets_dir, f'{row["image"]}.png'), dpi=100)
                plt.close(fig)
    return rows


def write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0])
    for r in rows[1:]:
        keys += [k for k in r if k not in keys]
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys, restval='')
        w.writeheader()
        w.writerows(rows)


def main():
    a = parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    torch.set_num_threads(max(1, int(os.environ.get('OMP_NUM_THREADS', '4'))))
    config = C.make_config(a.dataset, 256)
    ds_tr = NPY_datasets(a.data_path, config, train=True)
    ds_tr.transformer = config.test_transformer
    ds_te = NPY_datasets(a.data_path, config, train=False)
    n_tr = len(ds_tr) if a.max_images is None else min(a.max_images, len(ds_tr))
    n_te = len(ds_te) if a.max_images is None else min(a.max_images, len(ds_te))
    names = [os.path.splitext(os.path.basename(p[0]))[0] for p in ds_te.data]
    heavy_idx = set()
    if a.heavy:
        want = {r['image'] for r in csv.DictReader(open(a.heavy))}
        heavy_idx = {i for i, nm in enumerate(names[:n_te]) if nm in want}
    caps_tr = {'H': a.cap_train, 'band': a.cap_train, 'far': 0, 'core': 0}
    caps_te = {'H': a.cap_test, 'band': a.cap_test, 'far': a.cap_test_other, 'core': a.cap_test_other}
    all_rows, vis_rows = [], []
    for spec in a.run:
        label, path = spec.split('=', 1)
        sd, _ = C.load_state_dict(C.resolve_checkpoint(path))
        model, _ = C.build_model(sd, 'cpu', None, strict=True)
        print(f'{a.dataset} {label}: extracting (train {n_tr}, test {n_te} images)')
        Xtr, Mtr = extract(model, ds_tr, n_tr, caps_tr, a.seed)
        Xte, Mte = extract(model, ds_te, n_te, caps_te, a.seed + 1)
        print(f'  samples: train H {int((Mtr["set"] == 0).sum())} band {int((Mtr["set"] == 1).sum())} | '
              f'test H {int((Mte["set"] == 0).sum())} band {int((Mte["set"] == 1).sum())} far {int((Mte["set"] == 2).sum())}')
        rows, per_img = run_probes(Xtr, Mtr, Xte, Mte, heavy_idx, a)
        for r in rows:
            r.update({'dataset': a.dataset, 'model': label})
        all_rows += rows
        write_csv(os.path.join(a.out_dir, f'per_image_auc_{label}.csv'),
                  [{'image': names[i], 'heavy': int(i in heavy_idx), **{f'{lv}|{pb}': per_img[(lv, pb)].get(i, '')
                                                                         for lv, pb in per_img}}
                   for i in sorted({i for d in per_img.values() for i in d})])
        sheets = None
        if not a.no_sheets and not vis_rows:
            sheets = os.path.join(a.out_dir, f'sheets_{label}')
            os.makedirs(sheets, exist_ok=True)
        vr = visibility(model, ds_te, n_te, heavy_idx, sheets, label)
        for r in vr:
            r['model'] = label
        vis_rows += vr
    write_csv(os.path.join(a.out_dir, 'probes.csv'), all_rows)
    write_csv(os.path.join(a.out_dir, 'visibility.csv'), vis_rows)
    summ = {}
    for label in {r['model'] for r in vis_rows}:
        for grp, keep in (('all', lambda r: True), ('heavy', lambda r: r['heavy'] == 1)):
            v = [r for r in vis_rows if r['model'] == label and keep(r)]
            if v:
                summ[f'{label}/{grp}'] = {'n': len(v), **{k: float(np.nanmedian([r[k] for r in v]))
                                                       for k in ('t', 'dE_band_H', 'dE_band_core', 'axis_auc',
                                                                 'edge_gt_over_band', 'edge_inner_over_band')}}
    json.dump(summ, open(os.path.join(a.out_dir, 'visibility_summary.json'), 'w'), indent=1)
    print(json.dumps(summ, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
