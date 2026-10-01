'''
EXP-11 mechanism readouts (region completion, models/completion.py). CPU or GPU, one pass over the val split
per model plus one per knockout. Nothing here decides win / lose (that is pooled DSC, exp06_summary.py); the
error counts and the threshold-curve comparison are in analysis/exp11_errors.py.

  K  same-checkpoint knockouts of the completion runs: alpha = 0 (nothing added, = T = 0), barrier off (B = 0
     in the growth; 'full' runs only), evidence = 1 (A = 1). Delta pooled DSC, delta interior miss of the
     large tertile and of the --focus set (relative, with a 95 % bootstrap CI over images), delta FP. The
     completion and the rest of the network were trained together, so a knockout says whether and where the
     module acts, not whether it helps.
  M  the maps (completion runs): alpha; AUC of the evidence A and of the base probability sigmoid(z_base) for
     telling the base interior miss (lesion > 10 px inside the contour, base probability < 0.5) from the
     background band 10-30 px outside the contour (pixels at 256, at most --auc-per-image of each per image);
     the barrier B against the +-3 px contour ring at 128 (soft Dice, mean B on / off the ring, share of ring
     cells with B < 0.5 = gaps) and on the "inner edge" (the contour of the base prediction where it runs more
     than 10 px inside the lesion: a barrier there stops the growth at the dark core); the share of the base
     interior miss that the grown region covers (R_T >= 0.5) and the share of grown cells (R_T >= 0.5, not
     seed) lying more than 10 px outside the lesion.
  P  polygon split: every val image gets a polygon score (share of the GT contour on straight runs >= 30 px
     that deviate < 1.5 px from a chord, Ramer-Douglas-Peucker) and the solidity of the GT; the large tertile is
     split at its median score and every run is compared with --ref on each half (interior miss, pooled DSC),
     with and without the --drop images.

  python analysis/exp11_mechanism.py --dataset isic18 --data-path data/data_isic1718/isic2018 \
      --run D1=... --run D2=... --run D3=... --run D4=... --ref D1 \
      --focus heavy=reports/large_interior_fn/danh_sach_isic18.csv --drop 390 --out-dir <dir>

Outputs in --out-dir: mechanism.md, knockout.csv, maps.csv, polygon.csv, polygon_per_image.csv.
'''
import os
import sys
import csv
import math
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
from scipy.stats import rankdata

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis import common as C                                                    # noqa: E402
from analysis.exp09_mechanism import GroundTruth, pooled, write_csv, vn, px        # noqa: E402
from analysis.exp11_errors import load_focus                                        # noqa: E402
from datasets.dataset import NPY_datasets                                           # noqa: E402
import contour_losses as CL                                                         # noqa: E402

TERTILES = ('small', 'mid', 'large')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', action='append', required=True, metavar='LABEL=DIR')
    p.add_argument('--ref', default=None, metavar='LABEL', help='reference run of the polygon split (D1)')
    p.add_argument('--dataset', required=True, choices=['isic17', 'isic18'])
    p.add_argument('--data-path', required=True)
    p.add_argument('--focus', action='append', default=[], metavar='NAME=CSV')
    p.add_argument('--drop', action='append', default=[], metavar='IMAGE')
    p.add_argument('--threshold', type=float, default=0.5)
    p.add_argument('--near', type=float, default=3.0)
    p.add_argument('--mid', type=float, default=10.0)
    p.add_argument('--ring-radius', type=int, default=3)
    p.add_argument('--auc-per-image', type=int, default=2000)
    p.add_argument('--boot', type=int, default=2000)
    p.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    p.add_argument('--max-images', type=int, default=None)
    p.add_argument('--out-dir', required=True)
    return p.parse_args()


# ----------------------------------------------------------------------------- helpers

def auc(pos, neg):
    '''Mann-Whitney AUC: probability that a random positive scores above a random negative (ties count 1/2).'''
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def boot_ratio(a, b, n_boot, seed=0):
    '''sum(a) / sum(b) - 1 and its 95 % bootstrap CI over the paired items.'''
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if len(a) == 0 or b.sum() == 0:
        return float('nan'), float('nan'), float('nan')
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    sb = b[idx].sum(1)
    r = np.where(sb > 0, a[idx].sum(1) / np.where(sb > 0, sb, 1) - 1, np.nan)
    lo, hi = np.nanquantile(r, [0.025, 0.975])
    return float(a.sum() / b.sum() - 1), float(lo), float(hi)


def rdp(points, eps):
    '''Ramer-Douglas-Peucker on an open polyline (N, 2); returns the kept indices.'''
    keep = np.zeros(len(points), bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(points) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, b = points[i], points[j]
        seg = b - a
        norm = np.hypot(*seg)
        rel = points[i + 1:j] - a
        d = np.abs(seg[0] * rel[:, 1] - seg[1] * rel[:, 0]) / norm if norm > 0 else np.hypot(rel[:, 0], rel[:, 1])
        k = int(np.argmax(d))
        if d[k] > eps:
            m = i + 1 + k
            keep[m] = True
            stack += [(i, m), (m, j)]
    return np.flatnonzero(keep)


def polygon_score(g, eps=1.5, min_len=30.0):
    '''Share of the GT contour length on RDP segments of at least min_len px (straight within eps px).
    NaN without contourpy or for an empty mask.'''
    try:
        import contourpy
    except ImportError:
        return float('nan')
    if not g.any() or g.all():
        return float('nan')
    z = np.pad(g.astype(np.float64), 1)
    lines = contourpy.contour_generator(z=z).lines(0.5)
    total = straight = 0.0
    for ln in lines:
        if len(ln) < 3:
            continue
        keep = rdp(ln, eps)
        v = ln[keep]
        seg = np.hypot(*np.diff(v, axis=0).T)
        total += float(np.hypot(*np.diff(ln, axis=0).T).sum())
        straight += float(seg[seg >= min_len].sum())
    return straight / total if total > 0 else float('nan')


def solidity(g):
    from scipy.spatial import ConvexHull
    ys, xs = np.nonzero(g)
    if len(ys) < 3:
        return float('nan')
    pts = np.stack([np.concatenate([xs - .5, xs + .5, xs - .5, xs + .5]), np.concatenate([ys - .5, ys - .5, ys + .5, ys + .5])], 1)
    try:
        return float(len(ys) / ConvexHull(pts).volume)
    except Exception:   # degenerate hull
        return float('nan')


class Probe:
    def __init__(self, model):
        self.rec = {}
        self.handles = [model.final.register_forward_hook(lambda m, i, o: self.rec.__setitem__('z_base', o.detach()))]
        if hasattr(model, 'complete'):
            self.handles.append(model.complete.register_forward_hook(
                lambda m, i, o: self.rec.__setitem__('c', [t.detach() for t in o])))

    def remove(self):
        for h in self.handles:
            h.remove()


def up(t):
    return F.interpolate(t, scale_factor=(2, 2), mode='bilinear', align_corners=True)


# ----------------------------------------------------------------------------- passes

def plain_rows(model, ds, gt, device, thr):
    rows = []
    with torch.no_grad():
        for i in range(gt.n):
            img, _ = ds[i]
            _, out = model(img.float()[None].to(device))
            rows.append(gt.counts(i, out[0, 0].cpu().numpy() >= thr))
    return rows


def intact_pass(model, ds, gt, device, thr, ring_r, auc_n):
    '''Counts per image plus the map readouts of a completion model.'''
    probe = Probe(model)
    rows = []
    acc = dict(pos_a=[], pos_p=[], neg_a=[], neg_p=[], inter=0.0, sum_b=0.0, sum_t=0.0, b_on=0.0, n_on=0, b_off=0.0,
               n_off=0, gaps=0, b_inner=0.0, n_inner=0, miss_cov=0, miss=0, grown=0, leak=0, seed=0, imgs_grown=0)
    with torch.no_grad():
        for i in range(gt.n):
            img, msk = ds[i]
            _, out = model(img.float()[None].to(device))
            rows.append(gt.counts(i, out[0, 0].cpu().numpy() >= thr))
            z_base = probe.rec['z_base']
            add, ring_logit, r_t, a_map, b_map = probe.rec['c']
            g = gt.gt[i]
            p256 = torch.sigmoid(up(z_base))[0, 0].cpu().numpy()
            a256 = up(a_map)[0, 0].cpu().numpy()
            d_in = distance_transform_edt(g) if g.any() else np.zeros(g.shape)
            d_out = distance_transform_edt(~g) if g.any() else np.full(g.shape, np.inf)
            miss = g & (d_in > gt_mid) & (p256 < 0.5)
            band = ~g & (d_out > gt_mid) & (d_out <= 30)
            rng = np.random.default_rng(i)
            for key, m in (('pos', miss), ('neg', band)):
                ys, xs = np.nonzero(m)
                if len(ys) > auc_n:
                    k = rng.choice(len(ys), auc_n, replace=False)
                    ys, xs = ys[k], xs[k]
                acc[f'{key}_a'].append(a256[ys, xs])
                acc[f'{key}_p'].append(p256[ys, xs])
            # barrier at 128 against the ring and on the inner edge of the base prediction
            t = msk.float()[None].to(device)
            grid = b_map.shape[2:4]
            ring = CL.band_at_grid(CL.boundary_ring(t, ring_r), grid)
            on = ring > 0.5
            acc['inter'] += float((b_map * ring).sum())
            acc['sum_b'] += float(b_map.sum())
            acc['sum_t'] += float(ring.sum())
            acc['b_on'] += float(b_map[on].sum())
            acc['n_on'] += int(on.sum())
            acc['b_off'] += float(b_map[~on].sum())
            acc['n_off'] += int((~on).sum())
            acc['gaps'] += int((b_map[on] < 0.5).sum())
            mb = (torch.sigmoid(z_base) >= 0.5).float()
            edge = (F.max_pool2d(mb, 3, 1, 1) - (-F.max_pool2d(-mb, 3, 1, 1))) > 0.5
            deep = F.avg_pool2d(torch.from_numpy((d_in > gt_mid).astype(np.float32))[None, None].to(device), 2) > 0.5
            inner = edge & deep
            acc['b_inner'] += float(b_map[inner].sum())
            acc['n_inner'] += int(inner.sum())
            miss128 = F.avg_pool2d(torch.from_numpy(miss.astype(np.float32))[None, None].to(device), 2) > 0.5
            far_out = F.avg_pool2d(torch.from_numpy((~g & (d_out > gt_mid)).astype(np.float32))[None, None].to(device), 2) > 0.5
            seed = torch.sigmoid(z_base) >= model.complete.seed_thr
            grown = (r_t >= 0.5) & ~seed
            acc['miss_cov'] += int((miss128 & (r_t >= 0.5)).sum())
            acc['miss'] += int(miss128.sum())
            acc['grown'] += int(grown.sum())
            acc['leak'] += int((grown & far_out).sum())
            acc['seed'] += int(seed.sum())
            acc['imgs_grown'] += int(grown.any())
    probe.remove()
    return rows, acc


def knockout_rows(model, ds, gt, device, thr, **switch):
    c = model.complete
    saved = {k: getattr(c, k) for k in switch}
    for k, v in switch.items():
        setattr(c, k, v)
    try:
        return plain_rows(model, ds, gt, device, thr)
    finally:
        for k, v in saved.items():
            setattr(c, k, v)


gt_mid = 10.0


def main():
    global gt_mid
    a = parse_args()
    gt_mid = a.mid
    os.makedirs(a.out_dir, exist_ok=True)
    device = C.pick_device(a.device)
    config = C.make_config(a.dataset, 256)
    ds = NPY_datasets(a.data_path, config, train=False)
    n = len(ds) if a.max_images is None else min(a.max_images, len(ds))
    print(f'{a.dataset}: {n} val images')
    gt = GroundTruth(ds, n, a.near, a.mid)
    focus = load_focus(a.focus, gt.names) if a.focus else {}
    large = [i for i in range(n) if gt.tertile[i] == 2]
    drop = {i for i, nm in enumerate(gt.names) if nm in set(a.drop)}

    runs, rows, ko, maps = [], {}, [], []
    for spec in a.run:
        label, path = spec.split('=', 1)
        ckpt = C.resolve_checkpoint(path)
        sd, _ = C.load_state_dict(ckpt)
        model, cfg = C.build_model(sd, device, None, strict=True)
        mode = cfg.get('complete_mode', 'none')
        runs.append(label)
        print(f'  {label}: {os.path.basename(ckpt)}  complete={mode}')
        if mode == 'none':
            rows[label] = plain_rows(model, ds, gt, device, a.threshold)
            continue
        rows[label], acc = intact_pass(model, ds, gt, device, a.threshold, a.ring_radius, a.auc_per_image)
        cat = lambda k: np.concatenate(acc[k]) if acc[k] else np.zeros(0)
        maps.append({'run': label, 'mode': mode, 'alpha': float(model.complete.alpha.detach().cpu()),
                     'auc_evidence': auc(cat('pos_a'), cat('neg_a')), 'auc_base_prob': auc(cat('pos_p'), cat('neg_p')),
                     'n_pos': int(len(cat('pos_a'))), 'n_neg': int(len(cat('neg_a'))),
                     'mean_A_miss': float(cat('pos_a').mean()) if len(cat('pos_a')) else float('nan'),
                     'mean_A_band': float(cat('neg_a').mean()) if len(cat('neg_a')) else float('nan'),
                     'barrier_soft_dice': 2 * acc['inter'] / max(acc['sum_b'] + acc['sum_t'], 1e-12),
                     'B_on_ring': acc['b_on'] / max(acc['n_on'], 1), 'B_off_ring': acc['b_off'] / max(acc['n_off'], 1),
                     'ring_gap_pct': 100 * acc['gaps'] / max(acc['n_on'], 1),
                     'B_inner_edge': acc['b_inner'] / max(acc['n_inner'], 1) if acc['n_inner'] else float('nan'),
                     'inner_edge_cells': acc['n_inner'],
                     'miss_covered_pct': 100 * acc['miss_cov'] / max(acc['miss'], 1),
                     'grown_cells_per_image': acc['grown'] / n, 'grown_leak_far_out_pct': 100 * acc['leak'] / max(acc['grown'], 1),
                     'images_with_growth': acc['imgs_grown']})
        maps[-1]['inner_over_ring'] = maps[-1]['B_inner_edge'] / maps[-1]['B_on_ring'] if maps[-1]['B_on_ring'] else float('nan')
        variants = [('alpha = 0 (nothing added)', dict(enabled=False))]
        if mode == 'full':
            variants.append(('barrier off (B = 0)', dict(barrier_off=True)))
        variants.append(('evidence = 1 (A = 1)', dict(evidence_one=True)))
        base_all = pooled(rows[label], range(n))
        for name, sw in variants:
            kr = knockout_rows(model, ds, gt, device, a.threshold, **sw)
            k_all = pooled(kr, range(n))
            row = {'run': label, 'variant': name, 'dsc_pooled': k_all['dsc'], 'delta_dsc_pooled': k_all['dsc'] - base_all['dsc'],
                   'delta_fp': k_all['fp'] - base_all['fp'], 'delta_fn': k_all['fn'] - base_all['fn']}
            sets = {'large': large, **focus}
            for sname, idx in sets.items():
                r_rel, lo, hi = boot_ratio([kr[i]['fn_far'] for i in idx], [rows[label][i]['fn_far'] for i in idx], a.boot)
                row[f'miss_{sname}_intact'] = sum(rows[label][i]['fn_far'] for i in idx)
                row[f'miss_{sname}_knockout'] = sum(kr[i]['fn_far'] for i in idx)
                row[f'miss_{sname}_rel_change_pct'] = 100 * r_rel
                row[f'miss_{sname}_ci_lo_pct'] = 100 * lo
                row[f'miss_{sname}_ci_hi_pct'] = 100 * hi
            ko.append(row)
            print(f'    {name}: pooled DSC {row["delta_dsc_pooled"]:+.2f}, interior miss (large) {row["miss_large_rel_change_pct"]:+.1f} %')

    # polygon split of the large tertile, every run against --ref
    poly_img = []
    for i in range(n):
        poly_img.append({'image': gt.names[i], 'tertile': TERTILES[gt.tertile[i]], 'polygon_score': polygon_score(gt.gt[i]),
                         'solidity': solidity(gt.gt[i])})
    poly = []
    if a.ref and a.ref in rows:
        sc = np.array([poly_img[i]['polygon_score'] for i in large], dtype=np.float64)
        if np.isfinite(sc).any():
            med = float(np.nanmedian(sc))
            halves = {'polygon-like (score >= median)': [i for i in large if poly_img[i]['polygon_score'] >= med],
                      'smooth (score < median)': [i for i in large if poly_img[i]['polygon_score'] < med]}
            for hname, idx in list(halves.items()):
                if drop:
                    halves[f'{hname}, no {"/".join(sorted(a.drop))}'] = [i for i in idx if i not in drop]
            for r in runs:
                if r == a.ref:
                    continue
                for hname, idx in halves.items():
                    pr, pv = pooled(rows[a.ref], idx), pooled(rows[r], idx)
                    poly.append({'run': r, 'reference': a.ref, 'half': hname, 'n_images': len(idx), 'median_score': med,
                                 'dsc_ref': pr['dsc'], 'dsc_run': pv['dsc'], 'delta_dsc': pv['dsc'] - pr['dsc'],
                                 'miss_ref': pr['fn_far'], 'miss_run': pv['fn_far'],
                                 'miss_rel_change_pct': 100 * (pv['fn_far'] / pr['fn_far'] - 1) if pr['fn_far'] else float('nan'),
                                 'fp_ref': pr['fp'], 'fp_run': pv['fp']})
    for name, data in (('knockout', ko), ('maps', maps), ('polygon', poly), ('polygon_per_image', poly_img)):
        write_csv(os.path.join(a.out_dir, f'{name}.csv'), data)

    # ------------------------------------------------------------------ markdown
    md = [f'# EXP-11 — đọc cơ chế của region completion ({a.dataset.upper()}, {n} ảnh val)', '',
          'Knockout trên cùng checkpoint chỉ cho biết module có tác động và tác động ở đâu, không cho biết nó có lợi. '
          f'"Bỏ sót bên trong" = pixel tổn thương sâu hơn {a.mid:g} px bên trong viền thật mà model không tô.', '']
    if ko:
        sets = ['large'] + list(focus)
        md += ['## K. Knockout (so với chính model đó còn nguyên)', '',
               '| Run | Knockout | DSC gộp | Δ DSC gộp | Δ FP | ' + ' | '.join(f'Bỏ sót bên trong ({s}): Δ % [CI 95 %]' for s in sets) + ' |',
               '|---|---|---|---|---|' + '---|' * len(sets)]
        for r in ko:
            cells = [f'{vn(r[f"miss_{s}_rel_change_pct"], 1, True)} % [{vn(r[f"miss_{s}_ci_lo_pct"], 1, True)}; '
                     f'{vn(r[f"miss_{s}_ci_hi_pct"], 1, True)}] ({px(r[f"miss_{s}_intact"])} → {px(r[f"miss_{s}_knockout"])})' for s in sets]
            md.append(f'| {r["run"]} | {r["variant"]} | {vn(r["dsc_pooled"])} | {vn(r["delta_dsc_pooled"], 2, True)} '
                      f'| {px(r["delta_fp"], True)} | ' + ' | '.join(cells) + ' |')
    if maps:
        md += ['', '## M. Các bản đồ', '',
               f'AUC: khả năng phân biệt vùng bỏ sót bên trong (theo dự đoán gốc z_base) với dải nền {a.mid:g}–30 px ngoài viền; '
               '0,5 = không phân biệt được, 1 = tách hoàn toàn. "Viền trong" = ranh giới của dự đoán gốc ở chỗ nó nằm sâu hơn '
               f'{a.mid:g} px trong tổn thương (ranh giới lõi sẫm – vành nhạt).', '',
               '| Run | α | AUC của A | AUC xác suất gốc | A TB: vùng bỏ sót · dải nền | B: soft Dice với vành ±3 px | B TB trên vành · ngoài vành '
               '| % ô vành có B < 0,5 | B TB trên viền trong (tỉ lệ so với vành) | % vùng bỏ sót được lan phủ | ô lan ra / ảnh | % ô lan ra nằm ngoài viền > 10 px |',
               '|---|---|---|---|---|---|---|---|---|---|---|---|']
        md += [f'| {m["run"]} | {vn(m["alpha"], 3)} | {vn(m["auc_evidence"], 3)} | {vn(m["auc_base_prob"], 3)} '
               f'| {vn(m["mean_A_miss"], 3)} · {vn(m["mean_A_band"], 3)} | {vn(m["barrier_soft_dice"], 3)} '
               f'| {vn(m["B_on_ring"], 3)} · {vn(m["B_off_ring"], 3)} | {vn(m["ring_gap_pct"], 1)} '
               f'| {vn(m["B_inner_edge"], 3)} ({vn(m["inner_over_ring"], 2)}) | {vn(m["miss_covered_pct"], 1)} '
               f'| {vn(m["grown_cells_per_image"], 0)} | {vn(m["grown_leak_far_out_pct"], 1)} |' for m in maps]
    if poly:
        md += ['', f'## P. Chia nhóm to theo độ "đa giác" của GT (so với {a.ref})', '',
               'Điểm đa giác = phần chu vi GT nằm trên các đoạn thẳng ≥ 30 px (lệch < 1,5 px). GT vẽ tay bằng đa giác thô có điểm cao.', '',
               '| Run | Nửa | Số ảnh | DSC gộp mốc → run | Bỏ sót bên trong mốc → run (Δ %) | FP mốc → run |', '|---|---|---|---|---|---|']
        md += [f'| {p["run"]} | {p["half"]} | {p["n_images"]} | {vn(p["dsc_ref"])} → {vn(p["dsc_run"])} '
               f'| {px(p["miss_ref"])} → {px(p["miss_run"])} ({vn(p["miss_rel_change_pct"], 1, True)} %) | {px(p["fp_ref"])} → {px(p["fp_run"])} |'
               for p in poly]
    with open(os.path.join(a.out_dir, 'mechanism.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(md) + '\n')
    print(f'wrote {a.out_dir}/mechanism.md and csv files')
    return 0


if __name__ == '__main__':
    sys.exit(main())
