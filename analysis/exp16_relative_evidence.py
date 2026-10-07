'''
EXP-16: core-conditioned relative evidence, a read-only diagnostic on the original model (EXP-15 E0); no training.

Question: is a missed lesion interior pixel (H) closer in feature space to the model's own confident lesion core than
to the nearby skin, even though its probability is low? Relative evidence of a pixel x at a feature level:
    A_L(x) = cos(F(x), mu_L)     mu_L = mean feature over the core C = {P > t_c} (from the prediction only)
    A_B(x) = cos(F(x), mu_B)     mu_B = mean feature over B^ = {P < 0.1, 10 < distance to the predicted mask <= 30}
    R(x)   = A_L(x) - A_B(x)
The prototypes never see the ground truth (the 'bgGT' reference configuration uses the GT skin ring for mu_B and is an
oracle, never selected). The ground truth only defines the evaluation sets:
    H  = GT & distance to background > 10 px & P < 0.5          (analysis/info_probe.pixel_sets, EXP-11/12)
    B  = not GT & 5 < distance to the lesion <= 20 px            (B12: 10 < d <= 30, the EXP-12 band)
Feature levels (forward pre-hooks as in info_probe.py): E3 = t3 (input of encoder4, 24 ch, grid 32), E4 = t4
(encoder5, 32 ch, 16), E5 = t5 (encoder6, 48 ch, 8), Dec3 = out3 (decoder4, 24 ch, 32), Dec5 = out1 (final, 8 ch, 128),
upsampled to 256 with bilinear interpolation, align_corners=True (the sampling of info_probe.read_maps).

Configurations: main (core > 0.8, global mu_L, mu_B from B^); core0.7 / core0.9; local16/32/64 (mu_L = mean core
feature in a (2r+1)^2 window around x; undefined where the window holds no core pixel); bgGT (oracle reference).

Per image: AUC of H vs B (and B12) for P, A_L, A_B, R; percentile ranks of H and B in H u B; capped samples (300 + 300
pixels, the EXP-12 cap) for pooled AUC and the P-R density; all-pixel histograms for pixel-weighted pooled AUC and the
fixed rule R > 0; oracle completion: candidates = unpredicted pixels within r px of the predicted mask (r = 8 / 16 /
32), ranked by R (every configuration and level), by P, or at random; rules top-k % of the candidates (k = 1 / 2 / 5 /
10) and score > tau (global grids). For every selection: selected, selected in GT, selected in H, selected outside GT
farther than 3 px, selected in the deep-interior miss set (normalized depth >= 0.15, the EXP-14 measure).

  python analysis/exp16_relative_evidence.py --dataset isic17 --split val  --out-dir results/exp16_relative_evidence
  python analysis/exp16_relative_evidence.py --dataset isic17 --split test --out-dir results/exp16_relative_evidence
  (the test run reads <out-dir>/<ds>/selection_val.json written by the val run)
'''
import os
import sys
import csv
import json
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis import common as C                                   # noqa: E402
from analysis.info_probe import auc as mw_auc, pixel_sets          # noqa: E402
from datasets.dataset import split_dataset                         # noqa: E402
import contour_losses as CL                                        # noqa: E402

LEVELS = [('E3', 'encoder4'), ('E4', 'encoder5'), ('E5', 'encoder6'), ('Dec3', 'decoder4'), ('Dec5', 'final')]
LEVEL_NAMES = [n for n, _ in LEVELS]
CONFIGS = [('main', 0.8, 'global', 'pred'), ('core0.7', 0.7, 'global', 'pred'), ('core0.9', 0.9, 'global', 'pred'),
           ('local16', 0.8, 16, 'pred'), ('local32', 0.8, 32, 'pred'), ('local64', 0.8, 64, 'pred'),
           ('bgGT', 0.8, 'global', 'gt')]
CONFIG_NAMES = [c[0] for c in CONFIGS]
SELECTABLE = [c[0] for c in CONFIGS if c[3] == 'pred']
MIN_PX = 50
H_DIST, B_RING, B12_RING = 10, (5, 20), (10, 30)
BHAT_P, BHAT_RING = 0.1, (10, 30)
CAND_R = (8, 16, 32)
TOPK = (1, 2, 5, 10)
TAU_R = tuple(np.round(np.arange(-0.30, 0.601, 0.05), 2))
TAU_P = tuple(np.round(np.arange(0.05, 0.451, 0.05), 2))
CAP = 300
NB = 2000                                       # histogram bins over [lo, hi] per signal
RANGES = {'P': (0.0, 1.0), 'A': (-1.0, 1.0), 'R': (-2.0, 2.0)}
DATA = {'isic17': 'data/data_isic1718_701020/isic2017', 'isic18': 'data/data_isic1718_701020/isic2018'}
RUN = 'results/EGE-UNet-results-exp15/egeunet_{ds}_learnable_s42'
EDGES = {'isic17': (0.0779, 0.2402), 'isic18': (0.0671, 0.2324)}    # EXP-15 step 0 S0-4, from train
RANKERS = ['P', 'random'] + [f'R:{c}:{l}' for c in CONFIG_NAMES for l in LEVEL_NAMES]
FIELDS = ['image_id', 'dataset', 'split', 'status', 'num_core_pixels', 'core_frac', 'num_H_pixels', 'num_B_pixels',
          'num_B12_pixels', 'num_Bhat_pixels', 'area_frac', 'size_group', 'feature_level', 'core_threshold',
          'prototype_type', 'bg_source', 'config', 'coverage_H', 'coverage_B', 'AUC_probability',
          'AUC_lesion_affinity', 'AUC_background_affinity', 'AUC_relative_evidence', 'AUC_probability_B12',
          'AUC_relative_evidence_B12', 'median_R_H', 'median_R_B', 'mean_R_H', 'mean_R_B', 'frac_H_R_pos',
          'frac_B_R_pos', 'pct_rank_H_R', 'pct_rank_B_R', 'pct_rank_H_P', 'pct_rank_B_P',
          'oracle_r16_k5_dsc', 'oracle_r16_k5_dH', 'oracle_r16_k5_newFP', 'oracle_r16_tau0_dsc',
          'oracle_r16_tau0_dH', 'oracle_r16_tau0_newFP', 'baseline_dsc']


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', required=True, choices=['isic17', 'isic18'])
    p.add_argument('--split', required=True, choices=['val', 'test'])
    p.add_argument('--checkpoint', default=None, help='default: the EXP-15 E0 run of the dataset')
    p.add_argument('--out-dir', default='results/exp16_relative_evidence')
    p.add_argument('--max-images', type=int, default=None)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--summarize-only', action='store_true', help='reuse the saved raw arrays, redo summary / plots')
    return p.parse_args()


# ---------------------------------------------------------------------------------------------- regions (pure)
def eval_regions(gt, prob):
    '''Ground-truth evaluation sets: H (as info_probe.pixel_sets), B (5-20 px skin ring), B12 (10-30 px ring).'''
    d_in = distance_transform_edt(gt) if gt.any() else np.zeros(gt.shape)
    d_out = distance_transform_edt(~gt) if gt.any() else np.full(gt.shape, np.inf)
    H = gt & (d_in > H_DIST) & (prob < 0.5)
    B = ~gt & (d_out > B_RING[0]) & (d_out <= B_RING[1])
    B12 = ~gt & (d_out > B12_RING[0]) & (d_out <= B12_RING[1])
    return {'H': H, 'B': B, 'B12': B12, 'd_out': d_out}


def prediction_regions(prob, core_t):
    '''Regions built from the prediction only: core C, skin-prototype region B^, distance to the predicted mask.'''
    pred = prob >= 0.5
    d_pred = distance_transform_edt(~pred) if pred.any() else np.full(prob.shape, np.inf)
    core = prob > core_t
    bhat = (prob < BHAT_P) & (d_pred > BHAT_RING[0]) & (d_pred <= BHAT_RING[1])
    return core, bhat, d_pred


# ---------------------------------------------------------------------------------------------- evidence (pure)
def global_proto(feat, mask):
    '''feat (C,H,W) torch, mask (H,W) bool numpy -> mean feature (C,) over the mask.'''
    m = torch.from_numpy(mask)
    return feat[:, m].mean(1)


def cos_to(feat, mu):
    '''cosine of every pixel feature with one vector mu: (H,W) numpy.'''
    num = torch.einsum('chw,c->hw', feat, mu)
    return (num / (feat.norm(dim=0) * mu.norm() + 1e-8)).numpy()


def window_sum(x, r):
    '''Sum over the (2r+1)^2 window around every pixel (clipped at the border) via an integral image. x (C,H,W).'''
    c, h, w = x.shape
    ii = F.pad(x.double().cumsum(1).cumsum(2), (1, 0, 1, 0))
    ys, xs = torch.arange(h), torch.arange(w)
    y0, y1 = (ys - r).clamp(min=0), (ys + r + 1).clamp(max=h)
    x0, x1 = (xs - r).clamp(min=0), (xs + r + 1).clamp(max=w)
    return (ii[:, y1][:, :, x1] - ii[:, y0][:, :, x1] - ii[:, y1][:, :, x0] + ii[:, y0][:, :, x0]).float()


def local_cos(feat, core, r):
    '''cosine of F(x) with the mean core feature in the window around x; NaN where the window holds no core pixel.'''
    m = torch.from_numpy(core.astype(np.float32))[None]
    s = window_sum(feat * m, r)
    n = window_sum(m, r)[0]
    mu = s / n.clamp(min=1e-8)
    a = (feat * mu).sum(0) / (feat.norm(dim=0) * mu.norm(dim=0) + 1e-8)
    a = a.numpy()
    a[(n < 0.5).numpy()] = np.nan
    return a


def relative_evidence(feat, core, bg, proto):
    '''A_L, A_B, R from the core mask and the skin-prototype mask (both from the prediction in every selectable
    configuration); this function never receives the ground truth.'''
    a_l = cos_to(feat, global_proto(feat, core)) if proto == 'global' else local_cos(feat, core, int(proto))
    a_b = cos_to(feat, global_proto(feat, bg))
    return a_l, a_b, a_l - a_b


# ---------------------------------------------------------------------------------------------- statistics (pure)
def hist_add(h, vals, kind):
    lo, hi = RANGES[kind]
    v = vals[np.isfinite(vals)]
    idx = np.clip(((v - lo) / (hi - lo) * NB).astype(np.int64), 0, NB - 1)
    h += np.bincount(idx, minlength=NB)


def auc_hist(hp, hn):
    '''AUC of positives over negatives from two histograms on the same bins (ties within a bin count 1/2).'''
    n_p, n_n = hp.sum(), hn.sum()
    if n_p == 0 or n_n == 0:
        return float('nan')
    below = np.concatenate([[0], np.cumsum(hn)[:-1]])
    return float((hp * (below + 0.5 * hn)).sum() / (n_p * n_n))


def frac_above(h, kind, t):
    lo, hi = RANGES[kind]
    k = int(np.floor((t - lo) / (hi - lo) * NB + 1e-9))
    return float(h[k:].sum() / max(h.sum(), 1))


def pct_ranks(vals):
    '''percentile rank in [0, 1] of every value within the array (average ranks for ties).'''
    if len(vals) < 2:
        return np.full(len(vals), np.nan)
    return (stats.rankdata(vals) - 1) / (len(vals) - 1)


def selection_counts(score, attrs, ks, taus):
    '''Candidates ranked by score (NaN never selected). attrs (n, A) int. Returns (len(ks) + len(taus), A + 1) counts:
    [n_selected, sum of each attribute over the selected].'''
    n = len(score)
    s = np.where(np.isfinite(score), score, -np.inf)
    order = np.argsort(-s, kind='stable')
    cum = np.vstack([np.zeros((1, attrs.shape[1]), np.int64), np.cumsum(attrs[order], 0)])
    n_def = int(np.isfinite(s).sum())
    out = np.zeros((len(ks) + len(taus), attrs.shape[1] + 1), np.int64)
    for i, k in enumerate(ks):
        m = min(int(round(k / 100 * n)), n_def)
        out[i, 0], out[i, 1:] = m, cum[m]
    ss = s[order]
    for j, t in enumerate(taus):
        m = int(np.sum(ss > t))
        out[len(ks) + j, 0], out[len(ks) + j, 1:] = m, cum[m]
    return out


# ---------------------------------------------------------------------------------------------- main pass
def run_pass(a, out_dir):
    torch.set_num_threads(a.threads)
    run = a.checkpoint or RUN.format(ds=a.dataset)
    ckpt = C.resolve_checkpoint(run)
    sd, _ = C.load_state_dict(ckpt)
    model = C.build_model(sd, 'cpu', None, strict=True)[0]
    feats = {}
    hooks = [getattr(model, mod).register_forward_pre_hook(
        (lambda name: lambda m, inp: feats.__setitem__(name, inp[0].detach()))(name)) for name, mod in LEVELS]
    config = C.make_config(a.dataset, 256)
    ds = split_dataset(DATA[a.dataset], config, a.split)
    n = len(ds) if a.max_images is None else min(a.max_images, len(ds))
    edges = EDGES[a.dataset]
    combos = [(c, l) for c in CONFIG_NAMES for l in LEVEL_NAMES]
    n_sel = len(TOPK) + max(len(TAU_R), len(TAU_P))
    oracle = np.zeros((n, len(RANKERS), len(CAND_R), n_sel, 6), np.int32)   # n_sel, in GT, in H, far FP, deep, valid
    base = np.zeros((n, 8), np.int64)          # tp, fp, fn, |H|, |deep miss|, |deep|, size group, area*1e6
    hcov = np.zeros((n, len(CAND_R), 2), np.int64)   # H pixels inside the candidate zone, candidate pixels
    hists = {}
    samples = {'P': [], 'img': [], 'set': []}
    for c, l in combos:
        samples[f'R:{c}:{l}'] = []
    for l in LEVEL_NAMES:
        samples[f'AL:{l}'], samples[f'AB:{l}'] = [], []
    quad = {}
    rows, names, status_all = [], [], []
    with torch.no_grad():
        for i in range(n):
            img, msk = ds[i]
            _, out = model(img.float()[None])
            prob = out[0, 0].numpy().astype(np.float64)
            gt = msk.float()[0].numpy() >= 0.5
            name = os.path.basename(ds.data[i][0])
            names.append(name)
            ev = eval_regions(gt, prob)
            assert np.array_equal(ev['H'], pixel_sets(gt, prob)['H'])
            pred = prob >= 0.5
            tp, fp, fn = int((pred & gt).sum()), int((pred & ~gt).sum()), int((~pred & gt).sum())
            deep = (CL.normalized_depth(torch.from_numpy(gt[None, None].astype(np.float32)))[0, 0].numpy() >= 0.15) & gt
            area = float(gt.mean())
            grp = int(np.digitize(area, edges))
            base[i] = [tp, fp, fn, ev['H'].sum(), (deep & ~pred).sum(), deep.sum(), grp, int(area * 1e6)]
            fm = {k: F.interpolate(feats[k], size=gt.shape, mode='bilinear', align_corners=True)[0] for k in LEVEL_NAMES}
            H, B, B12 = ev['H'], ev['B'], ev['B12']
            nH, nB = int(H.sum()), int(B.sum())
            rng = np.random.default_rng(16 * 100003 + i)
            hy, hx = np.nonzero(H)
            by, bx = np.nonzero(B)
            hk = rng.choice(len(hy), min(CAP, len(hy)), replace=False) if len(hy) else np.zeros(0, int)
            bk = rng.choice(len(by), min(CAP, len(by)), replace=False) if len(by) else np.zeros(0, int)
            sy, sx = np.concatenate([hy[hk], by[bk]]), np.concatenate([hx[hk], bx[bk]])
            samples['P'].append(prob[sy, sx].astype(np.float32))
            samples['img'].append(np.full(len(sy), i, np.int32))
            samples['set'].append(np.concatenate([np.zeros(len(hk), np.int8), np.ones(len(bk), np.int8)]))
            for sname, mask in (('H', H), ('B', B), ('B12', B12)):
                hist_add(hists.setdefault(('P', sname), np.zeros(NB, np.int64)), prob[mask], 'P')
            # oracle candidates
            cand_attr = {}
            d_pred_main = None
            for r in CAND_R:
                if not pred.any():
                    cand_attr[r] = None
                    continue
                if d_pred_main is None:
                    d_pred_main = distance_transform_edt(~pred)
                cm = ~pred & (d_pred_main <= r)
                cand_attr[r] = (cm, np.stack([gt[cm], H[cm], (~gt & (ev['d_out'] > 3))[cm], (deep & ~pred)[cm]], 1).astype(np.int64))
                hcov[i, CAND_R.index(r)] = [int((H & cm).sum()), int(cm.sum())]

            def add_oracle(ri, score_map, taus):
                for j, r in enumerate(CAND_R):
                    if cand_attr[r] is None:
                        continue
                    cm, attrs = cand_attr[r]
                    cnt = selection_counts(score_map[cm], attrs, TOPK, taus)
                    oracle[i, ri, j, :cnt.shape[0], :5] = cnt
                    oracle[i, ri, j, :, 5] = 1
            add_oracle(RANKERS.index('P'), prob, TAU_P)
            add_oracle(RANKERS.index('random'), rng.random(prob.shape), [])
            # per configuration and level
            core_cache = {}
            img_rows = []
            p_auc = mw_auc(prob[H], prob[B]) if nH >= MIN_PX and nB >= MIN_PX else float('nan')
            p_auc12 = mw_auc(prob[H], prob[B12]) if nH >= MIN_PX and B12.sum() >= MIN_PX else float('nan')
            hb = H | B
            pr_all = pct_ranks(prob[hb]) if nH and nB else None
            for cname, core_t, proto, bgsrc in CONFIGS:
                if core_t not in core_cache:
                    core_cache[core_t] = prediction_regions(prob, core_t)
                core, bhat, _ = core_cache[core_t]
                bg = bhat if bgsrc == 'pred' else B
                status = 'ok'
                if core.sum() < MIN_PX:
                    status = 'no_core'
                elif bg.sum() < MIN_PX:
                    status = 'no_bg'
                for lname in LEVEL_NAMES:
                    row = {'image_id': name, 'dataset': a.dataset, 'split': a.split, 'num_core_pixels': int(core.sum()),
                           'core_frac': float(core.mean()), 'num_H_pixels': nH, 'num_B_pixels': nB,
                           'num_B12_pixels': int(B12.sum()), 'num_Bhat_pixels': int(bg.sum()) if bgsrc == 'pred' else int(bhat.sum()),
                           'area_frac': area, 'size_group': grp, 'feature_level': lname, 'core_threshold': core_t,
                           'prototype_type': proto, 'bg_source': bgsrc, 'config': cname, 'AUC_probability': p_auc,
                           'AUC_probability_B12': p_auc12, 'baseline_dsc': C.dsc_from_counts(tp, fp, fn)}
                    st = status if status != 'ok' else ('few_H_or_B' if nH < MIN_PX or nB < MIN_PX else 'ok')
                    row['status'] = st
                    key = f'R:{cname}:{lname}'
                    if status != 'ok':
                        samples[key].append(np.full(len(sy), np.nan, np.float32))
                        if cname == 'main':
                            samples[f'AL:{lname}'].append(np.full(len(sy), np.nan, np.float32))
                            samples[f'AB:{lname}'].append(np.full(len(sy), np.nan, np.float32))
                        img_rows.append(row)
                        continue
                    a_l, a_b, R = relative_evidence(fm[lname], core, bg, proto)
                    samples[key].append(R[sy, sx].astype(np.float32))
                    if cname == 'main':
                        samples[f'AL:{lname}'].append(a_l[sy, sx].astype(np.float32))
                        samples[f'AB:{lname}'].append(a_b[sy, sx].astype(np.float32))
                    add_oracle(RANKERS.index(key), R, TAU_R)
                    rh, rb = R[H], R[B]
                    row['coverage_H'] = float(np.isfinite(rh).mean()) if nH else float('nan')
                    row['coverage_B'] = float(np.isfinite(rb).mean()) if nB else float('nan')
                    for sname, mask in (('H', H), ('B', B), ('B12', B12)):
                        hist_add(hists.setdefault((key, sname), np.zeros(NB, np.int64)), R[mask], 'R')
                        if cname == 'main':
                            hist_add(hists.setdefault((f'AL:{lname}', sname), np.zeros(NB, np.int64)), a_l[mask], 'A')
                            hist_add(hists.setdefault((f'AB:{lname}', sname), np.zeros(NB, np.int64)), a_b[mask], 'A')
                    if cname == 'main':
                        q = quad.setdefault(lname, np.zeros(6, np.int64))
                        q += [int((np.isfinite(rh) & (rh > 0)).sum()), int(np.isfinite(rh).sum()),
                              int((rb > 0)[prob[B] < 0.5].sum()), int((prob[B] < 0.5).sum()), int((rb > 0)[prob[B] >= 0.5].sum()),
                              int((prob[B] >= 0.5).sum())]
                    fh, fb = rh[np.isfinite(rh)], rb[np.isfinite(rb)]
                    if st == 'ok' and len(fh) >= MIN_PX and len(fb) >= MIN_PX:
                        row['AUC_relative_evidence'] = mw_auc(fh, fb)
                        b12 = R[B12][np.isfinite(R[B12])]
                        row['AUC_relative_evidence_B12'] = mw_auc(fh, b12) if len(b12) >= MIN_PX else float('nan')
                        row.update({'median_R_H': float(np.median(fh)), 'median_R_B': float(np.median(fb)),
                                    'mean_R_H': float(fh.mean()), 'mean_R_B': float(fb.mean()),
                                    'frac_H_R_pos': float((fh > 0).mean()), 'frac_B_R_pos': float((fb > 0).mean())})
                        if cname == 'main':
                            row['AUC_lesion_affinity'] = mw_auc(a_l[H], a_l[B])
                            row['AUC_background_affinity'] = mw_auc(a_b[H], a_b[B])
                        rr = R[hb]
                        if np.isfinite(rr).all():
                            pr = pct_ranks(rr)
                            hmask = H[hb]
                            row['pct_rank_H_R'], row['pct_rank_B_R'] = float(pr[hmask].mean()), float(pr[~hmask].mean())
                            row['pct_rank_H_P'], row['pct_rank_B_P'] = float(pr_all[hmask].mean()), float(pr_all[~hmask].mean())
                    img_rows.append(row)
            # fixed oracle reference settings in the csv: r = 16, k = 5 % and tau = 0
            j16 = CAND_R.index(16)
            for row in img_rows:
                ri = RANKERS.index(f'R:{row["config"]}:{row["feature_level"]}')
                if oracle[i, ri, j16, 0, 5]:
                    for tag, si in (('k5', TOPK.index(5)), ('tau0', len(TOPK) + TAU_R.index(0.0))):
                        cnt = oracle[i, ri, j16, si]
                        row[f'oracle_r16_{tag}_dsc'] = C.dsc_from_counts(tp + cnt[1], fp + cnt[0] - cnt[1], fn - cnt[1])
                        row[f'oracle_r16_{tag}_dH'] = -int(cnt[2])
                        row[f'oracle_r16_{tag}_newFP'] = int(cnt[0] - cnt[1])
            rows.extend(img_rows)
            status_all.append(img_rows[0]['status'] if img_rows else 'none')
            if (i + 1) % 50 == 0:
                print(f'  {i + 1}/{n}', flush=True)
    for h in hooks:
        h.remove()
    d = os.path.join(out_dir, a.dataset)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f'per_image_{a.split}.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, restval='')
        w.writeheader()
        w.writerows(rows)
    np.savez_compressed(os.path.join(d, f'raw_{a.split}.npz'), oracle=oracle, base=base, hcov=hcov, names=np.array(names),
                        samples_keys=np.array(list(samples)), **{f's_{k}': np.concatenate(v) for k, v in samples.items()},
                        hist_keys=np.array(['|'.join(k) for k in hists]), hists=np.stack(list(hists.values())),
                        quad_keys=np.array(list(quad)), quad=np.stack([quad[k] for k in quad]) if quad else np.zeros((0, 6)))
    print(f'checkpoint: {ckpt}; {n} images; wrote {d}/per_image_{a.split}.csv and raw_{a.split}.npz')


# ---------------------------------------------------------------------------------------------- summary
def load_raw(d, split):
    z = np.load(os.path.join(d, f'raw_{split}.npz'), allow_pickle=False)
    samples = {k: z[f's_{k}'] for k in z['samples_keys']}
    hists = {tuple(k.split('|')): h for k, h in zip(z['hist_keys'], z['hists'])}
    quad = {k: q for k, q in zip(z['quad_keys'], z['quad'])}
    return z['oracle'], z['base'], z['hcov'], z['names'], samples, hists, quad


def read_rows(d, split):
    rows = list(csv.DictReader(open(os.path.join(d, f'per_image_{split}.csv'))))
    for r in rows:
        for k, v in r.items():
            if k in ('image_id', 'dataset', 'split', 'status', 'feature_level', 'prototype_type', 'bg_source', 'config'):
                continue
            r[k] = float(v) if v != '' else float('nan')
    return rows


def hard_images(base):
    '''EXP-11 heavy-set rule on this model and split: large lesions, by |H| descending until 80 % of their H.'''
    large = np.flatnonzero(base[:, 6] == 2)
    order = large[np.argsort(-base[large, 3], kind='stable')]
    tot = base[order, 3].sum()
    keep, acc = [], 0
    for i in order:
        if acc >= 0.8 * tot or base[i, 3] == 0:
            break
        keep.append(int(i))
        acc += base[i, 3]
    return set(keep)


def per_image_stats(v):
    v = np.asarray([x for x in v if np.isfinite(x)])
    if len(v) == 0:
        return {'n': 0}
    return {'n': int(len(v)), 'mean': float(v.mean()), 'median': float(np.median(v)), 'p25': float(np.percentile(v, 25)),
            'frac_gt_0.7': float((v > 0.7).mean()), 'frac_gt_0.8': float((v > 0.8).mean())}


def capped_auc(samples, key, img_set=None):
    s, sets, img = samples[key], samples['set'], samples['img']
    ok = np.isfinite(s) & (np.ones(len(s), bool) if img_set is None else np.isin(img, list(img_set)))
    return mw_auc(s[ok & (sets == 0)], s[ok & (sets == 1)])


def oracle_table(oracle, base, ri, j, si, valid_only=None):
    '''Per-image metrics after adding the selection (ranker ri, radius index j, selection index si).'''
    tp, fp, fn, nH, ndm = base[:, 0], base[:, 1], base[:, 2], base[:, 3], base[:, 4]
    cnt = oracle[:, ri, j, si, :5].astype(np.float64)
    sel, sgt, sh, sfar, sdeep = cnt.T
    tp2, fp2, fn2 = tp + sgt, fp + sel - sgt, fn - sgt
    dsc = np.where(2 * tp2 + fp2 + fn2 > 0, 2 * tp2 / np.maximum(2 * tp2 + fp2 + fn2, 1), 1.0)
    return {'dsc': dsc, 'prec': np.where(tp2 + fp2 > 0, tp2 / np.maximum(tp2 + fp2, 1), np.nan),
            'rec': np.where(tp2 + fn2 > 0, tp2 / np.maximum(tp2 + fn2, 1), np.nan), 'sel': sel, 'sgt': sgt, 'sh': sh,
            'sfar': sfar, 'sdeep': sdeep, 'nH': nH, 'ndm': ndm, 'area0': tp + fp, 'area1': tp2 + fp2}


def oracle_summary(oracle, base, ri, j, si):
    t = oracle_table(oracle, base, ri, j, si)
    b = {'dsc': np.where(2 * base[:, 0] + base[:, 1] + base[:, 2] > 0, 2 * base[:, 0] / np.maximum(2 * base[:, 0] + base[:, 1] + base[:, 2], 1), 1.0)}
    large = base[:, 6] == 2
    sel = t['sel'].sum()
    p = stats.ttest_rel(t['dsc'], b['dsc']).pvalue if np.any(t['dsc'] != b['dsc']) else float('nan')
    return {'dsc': 100 * float(t['dsc'].mean()), 'd_dsc': 100 * float((t['dsc'] - b['dsc']).mean()), 'p_dsc': float(p),
            'precision': 100 * float(np.nanmean(t['prec'])), 'recall': 100 * float(np.nanmean(t['rec'])),
            'dsc_large': 100 * float(t['dsc'][large].mean()) if large.any() else float('nan'),
            'd_dsc_large': 100 * float((t['dsc'] - b['dsc'])[large].mean()) if large.any() else float('nan'),
            'H_before': int(t['nH'][large].sum()), 'H_fixed': int(t['sh'][large].sum()),
            'interior_fn_reduction_pct': 100 * float(t['sh'][large].sum() / max(t['nH'][large].sum(), 1)),
            'deep_miss_before': int(t['ndm'][large].sum()), 'deep_miss_fixed': int(t['sdeep'][large].sum()),
            'new_fp': int((t['sel'] - t['sgt']).sum()), 'new_fp_far3': int(t['sfar'].sum()),
            'selected': int(sel), 'expansion_precision': 100 * float(t['sgt'].sum() / sel) if sel else float('nan'),
            'expansion_recall_H': 100 * float(t['sh'].sum() / max(base[:, 3].sum(), 1)),
            'area_change_pct': 100 * float((t['area1'].sum() - t['area0'].sum()) / max(t['area0'].sum(), 1)),
            'fixed_H_per_new_fp': float(t['sh'].sum() / max((t['sel'] - t['sgt']).sum(), 1))}


def summarize(a, out_dir):
    d = os.path.join(out_dir, a.dataset)
    oracle, base, hcov, names, samples, hists, quad = load_raw(d, a.split)
    rows = read_rows(d, a.split)
    n = len(names)
    hard = hard_images(base)
    name_idx = {nm: i for i, nm in enumerate(names)}
    S = {'dataset': a.dataset, 'split': a.split, 'n_images': n, 'n_hard': len(hard),
         'n_large': int((base[:, 6] == 2).sum()), 'H_total': int(base[:, 3].sum()),
         'H_in_hard_pct': 100 * float(base[list(hard), 3].sum() / max(base[:, 3].sum(), 1)) if hard else 0.0}
    # core / status statistics (configuration main, one row per image)
    main_rows = [r for r in rows if r['config'] == 'main' and r['feature_level'] == 'E3']
    st = {}
    for r in main_rows:
        st[r['status']] = st.get(r['status'], 0) + 1
    core_rows = {ct: [r for r in rows if r['feature_level'] == 'E3' and r['config'] == cn]
                 for cn, ct in (('core0.7', 0.7), ('main', 0.8), ('core0.9', 0.9))}
    S['status_main'] = st
    S['core'] = {str(ct): {'mean_core_px': float(np.mean([r['num_core_pixels'] for r in rr])),
                           'mean_core_frac_pct': 100 * float(np.mean([r['core_frac'] for r in rr])),
                           'images_without_core': int(sum(r['num_core_pixels'] < MIN_PX for r in rr))}
                 for ct, rr in core_rows.items()}
    S['bhat'] = {'mean_px': float(np.mean([r['num_Bhat_pixels'] for r in main_rows])),
                 'images_without_bhat': int(sum(r['num_Bhat_pixels'] < MIN_PX for r in main_rows))}
    # AUC tables
    auc = {}
    for cname in CONFIG_NAMES:
        for lname in LEVEL_NAMES:
            rr = [r for r in rows if r['config'] == cname and r['feature_level'] == lname]
            key = f'R:{cname}:{lname}'
            ent = {'pooled_all_px': auc_hist(hists.get((key, 'H'), np.zeros(NB)), hists.get((key, 'B'), np.zeros(NB))),
                   'pooled_all_px_B12': auc_hist(hists.get((key, 'H'), np.zeros(NB)), hists.get((key, 'B12'), np.zeros(NB))),
                   'pooled_capped': capped_auc(samples, key), 'pooled_capped_hard': capped_auc(samples, key, hard),
                   'per_image': per_image_stats([r['AUC_relative_evidence'] for r in rr]),
                   'per_image_hard': per_image_stats([r['AUC_relative_evidence'] for r in rr if name_idx[r['image_id']] in hard]),
                   'per_image_B12': per_image_stats([r['AUC_relative_evidence_B12'] for r in rr]),
                   'coverage_H': float(np.nanmean([r['coverage_H'] for r in rr])) if rr else float('nan')}
            hh, hb_ = hists.get((key, 'H'), np.zeros(NB)), hists.get((key, 'B'), np.zeros(NB))
            ent['frac_H_R_pos'], ent['frac_B_R_pos'] = frac_above(hh, 'R', 0.0), frac_above(hb_, 'R', 0.0)
            ent['balacc_R_pos'] = 0.5 * (ent['frac_H_R_pos'] + 1 - ent['frac_B_R_pos'])
            ent['pct_rank_H'] = per_image_stats([r['pct_rank_H_R'] for r in rr])
            ent['pct_rank_B'] = per_image_stats([r['pct_rank_B_R'] for r in rr])
            if cname == 'main':
                for sig, kind in (('AL', 'A'), ('AB', 'A')):
                    k2 = f'{sig}:{lname}'
                    ent[f'{sig}_pooled_all_px'] = auc_hist(hists.get((k2, 'H'), np.zeros(NB)), hists.get((k2, 'B'), np.zeros(NB)))
                    ent[f'{sig}_pooled_capped'] = capped_auc(samples, k2)
                    col = 'AUC_lesion_affinity' if sig == 'AL' else 'AUC_background_affinity'
                    ent[f'{sig}_per_image'] = per_image_stats([r[col] for r in rr])
                    ent[f'{sig}_per_image_hard'] = per_image_stats([r[col] for r in rr if name_idx[r['image_id']] in hard])
            auc[key] = ent
    ok_main = [r for r in rows if r['config'] == 'main' and r['feature_level'] == 'E3']
    auc['P'] = {'pooled_all_px': auc_hist(hists[('P', 'H')], hists[('P', 'B')]),
                'pooled_all_px_B12': auc_hist(hists[('P', 'H')], hists[('P', 'B12')]),
                'pooled_capped': capped_auc(samples, 'P'), 'pooled_capped_hard': capped_auc(samples, 'P', hard),
                'per_image_all_with_H_B': per_image_stats([r['AUC_probability'] for r in ok_main]),
                'per_image': per_image_stats([r['AUC_probability'] for r in ok_main if r['status'] == 'ok']),
                'per_image_hard': per_image_stats([r['AUC_probability'] for r in ok_main if r['status'] == 'ok' and name_idx[r['image_id']] in hard]),
                'per_image_B12': per_image_stats([r['AUC_probability_B12'] for r in ok_main if r['status'] == 'ok']),
                'pct_rank_H': per_image_stats([r['pct_rank_H_P'] for r in ok_main]),
                'pct_rank_B': per_image_stats([r['pct_rank_B_P'] for r in ok_main])}
    # P threshold rule: balanced accuracy over the grid (best chosen on val)
    hp, hn = hists[('P', 'H')], hists[('P', 'B')]
    auc['P']['balacc_by_tau'] = {f'{t:.2f}': 0.5 * (frac_above(hp, 'P', t) + 1 - frac_above(hn, 'P', t)) for t in np.arange(0.02, 0.5, 0.02)}
    S['auc'] = auc
    S['quadrant_main'] = {k: {'H_R_pos_pct': 100 * q[0] / max(q[1], 1), 'B_lowP_R_pos_pct': 100 * q[2] / max(q[3], 1),
                              'B_highP_R_pos_pct': 100 * q[4] / max(q[5], 1), 'B_highP_pct': 100 * q[5] / max(q[3] + q[5], 1)}
                          for k, q in quad.items()}
    # oracle: full grid summary
    orc = {}
    for ri, rk in enumerate(RANKERS):
        if not oracle[:, ri, :, 0, 5].any():
            continue
        taus = TAU_P if rk == 'P' else ([] if rk == 'random' else TAU_R)
        for j, r in enumerate(CAND_R):
            for si, k in enumerate(TOPK):
                orc[f'{rk}|r{r}|k{k}'] = oracle_summary(oracle, base, ri, j, si)
            for tj, t in enumerate(taus):
                orc[f'{rk}|r{r}|tau{t:.2f}'] = oracle_summary(oracle, base, ri, j, len(TOPK) + tj)
    S['oracle'] = orc
    large = base[:, 6] == 2
    S['candidate_zone'] = {r: {'H_inside_pct': 100 * float(hcov[:, j, 0].sum() / max(base[:, 3].sum(), 1)),
                               'H_inside_pct_large': 100 * float(hcov[large, j, 0].sum() / max(base[large, 3].sum(), 1)),
                               'mean_candidates_px': float(hcov[:, j, 1].mean())} for j, r in enumerate(CAND_R)}
    S['oracle_baseline'] = {'dsc': 100 * float(np.mean([C.dsc_from_counts(*b[:3]) for b in base])), 'H_large': int(base[base[:, 6] == 2, 3].sum())}
    return S, rows, samples, oracle, base, hard


def select_on_val(S):
    auc = S['auc']
    cand = [(auc[f'R:{c}:{l}']['pooled_all_px'], c, l) for c in SELECTABLE for l in LEVEL_NAMES
            if np.isfinite(auc[f'R:{c}:{l}']['pooled_all_px'])]
    best_auc, c, l = max(cand)
    rk = f'R:{c}:{l}'
    orc = S['oracle']

    def best(prefix, kinds):
        keys = [k for k in orc if k.startswith(prefix + '|') and any(f'|{kd}' in k for kd in kinds)]
        return max(keys, key=lambda k: orc[k]['dsc']) if keys else None
    bal = auc['P']['balacc_by_tau']
    tau_p_rule = max(bal, key=bal.get)
    return {'config': c, 'level': l, 'ranker': rk, 'val_pooled_auc_R': best_auc,
            'tau_rule_R': best(rk, ['tau']), 'tau_rule_P': best('P', ['tau']),
            'topk_rule_R': best(rk, ['k']), 'topk_rule_P': best('P', ['k']), 'topk_rule_random': best('random', ['k']),
            'P_threshold_balacc': float(tau_p_rule)}


# ---------------------------------------------------------------------------------------------- report helpers
def vn(x, d=2, sign=False):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return '—'
    return (f'{x:+.{d}f}' if sign else f'{x:.{d}f}').replace('.', ',')


def pr_points(S, ranker, r, rule):
    pts = []
    for k, v in S['oracle'].items():
        rk, rr, sel = k.split('|')
        if rk == ranker and rr == f'r{r}' and sel.startswith(rule):
            pts.append((v['expansion_recall_H'], v['expansion_precision'], sel))
    return sorted(pts)


def precision_at_recall(pts, target):
    '''Highest expansion precision among operating points with recall >= target (NaN if none).'''
    ok = [p for rcl, p, _ in pts if rcl >= target and np.isfinite(p)]
    return max(ok) if ok else float('nan')


def write_markdown(S, sel, path):
    a = S['auc']
    L = [f'# EXP-16 {S["dataset"]} — {S["split"]} ({S["n_images"]} ảnh; ảnh khó {S["n_hard"]}, chứa {vn(S["H_in_hard_pct"], 1)} % pixel H)', '',
         f'Trạng thái ảnh (cấu hình chính): {S["status_main"]}. Core: ' +
         '; '.join(f'P > {k}: TB {vn(v["mean_core_px"], 0)} px ({vn(v["mean_core_frac_pct"], 1)} % ảnh), {v["images_without_core"]} ảnh không đủ core'
                   for k, v in S['core'].items()) + f'. B^: TB {vn(S["bhat"]["mean_px"], 0)} px, {S["bhat"]["images_without_bhat"]} ảnh không đủ.', '',
         '## AUC H so với B (cấu hình chính: core > 0,8, prototype toàn cục, μ_B từ prediction)', '',
         '| Tín hiệu | ' + ' | '.join(LEVEL_NAMES) + ' |', '|---|' + '---|' * len(LEVEL_NAMES)]
    pP = a['P']
    L.append('| P (gộp mọi pixel · gộp mẫu · trung vị từng ảnh) | ' + ' | '.join(
        f'{vn(pP["pooled_all_px"], 3)} · {vn(pP["pooled_capped"], 3)} · {vn(pP["per_image"].get("median"), 3)}' for _ in LEVEL_NAMES) + ' |')
    for sig, lab in (('AL', 'A_L'), ('AB', 'A_B')):
        L.append(f'| {lab} | ' + ' | '.join(
            f'{vn(a[f"R:main:{l}"][f"{sig}_pooled_all_px"], 3)} · {vn(a[f"R:main:{l}"][f"{sig}_pooled_capped"], 3)} · {vn(a[f"R:main:{l}"][f"{sig}_per_image"].get("median"), 3)}'
            for l in LEVEL_NAMES) + ' |')
    L.append('| **R = A_L − A_B** | ' + ' | '.join(
        f'**{vn(a[f"R:main:{l}"]["pooled_all_px"], 3)}** · {vn(a[f"R:main:{l}"]["pooled_capped"], 3)} · {vn(a[f"R:main:{l}"]["per_image"].get("median"), 3)}'
        for l in LEVEL_NAMES) + ' |')
    L += ['', '## AUC từng ảnh (cấu hình chính): trung bình / trung vị / phân vị 25 % / % ảnh > 0,7 / % ảnh > 0,8 (n ảnh)', '',
          '| Tín hiệu | Tầng | Mọi ảnh hợp lệ | Ảnh khó |', '|---|---|---|---|']

    def fmt(s):
        if not s or s.get('n', 0) == 0:
            return '—'
        return f'{vn(s["mean"], 3)} / {vn(s["median"], 3)} / {vn(s["p25"], 3)} / {vn(100 * s["frac_gt_0.7"], 0)} % / {vn(100 * s["frac_gt_0.8"], 0)} % (n {s["n"]})'
    L.append(f'| P | — | {fmt(pP["per_image"])} | {fmt(pP["per_image_hard"])} |')
    for l in LEVEL_NAMES:
        L.append(f'| R | {l} | {fmt(a[f"R:main:{l}"]["per_image"])} | {fmt(a[f"R:main:{l}"]["per_image_hard"])} |')
    L += ['', '## Độ bền của R (AUC gộp mọi pixel · trung vị từng ảnh)', '', '| Cấu hình | ' + ' | '.join(LEVEL_NAMES) + ' |',
          '|---|' + '---|' * len(LEVEL_NAMES)]
    for c in CONFIG_NAMES:
        L.append(f'| {c} | ' + ' | '.join(f'{vn(a[f"R:{c}:{l}"]["pooled_all_px"], 3)} · {vn(a[f"R:{c}:{l}"]["per_image"].get("median"), 3)}' for l in LEVEL_NAMES) + ' |')
    L.append(f'| main, tập âm B12 (10–30 px) | ' + ' | '.join(f'{vn(a[f"R:main:{l}"]["pooled_all_px_B12"], 3)} · {vn(a[f"R:main:{l}"]["per_image_B12"].get("median"), 3)}' for l in LEVEL_NAMES)
             + f' |  (P: {vn(pP["pooled_all_px_B12"], 3)} · {vn(pP["per_image_B12"].get("median"), 3)})')
    L += ['', '## Luật ngưỡng cố định và xếp hạng (cấu hình chính)', '',
          '| Tầng | % H có R > 0 | % B có R > 0 | balanced accuracy R > 0 | hạng % TB của H · B (theo R) |', '|---|---|---|---|---|']
    for l in LEVEL_NAMES:
        e = a[f'R:main:{l}']
        L.append(f'| {l} | {vn(100 * e["frac_H_R_pos"], 1)} | {vn(100 * e["frac_B_R_pos"], 1)} | {vn(e["balacc_R_pos"], 3)} | '
                 f'{vn(e["pct_rank_H"].get("mean"), 3)} · {vn(e["pct_rank_B"].get("mean"), 3)} |')
    bal = pP['balacc_by_tau']
    tb = f'{sel["P_threshold_balacc"]:.2f}' if (sel and S['split'] == 'test') else max(bal, key=bal.get)
    src = 'chọn trên val' if (sel and S['split'] == 'test') else 'tốt nhất trên split này'
    L.append(f'| P | ngưỡng P {src}: P > {tb} → balanced accuracy {vn(bal[tb], 3)}; '
             f'hạng % TB của H · B (theo P) {vn(pP["pct_rank_H"].get("mean"), 3)} · {vn(pP["pct_rank_B"].get("mean"), 3)} | | | |')
    L += ['', '## Góc phần tư P–R (cấu hình chính, mọi pixel)', '', '| Tầng | % H (đều có P < 0,5) có R > 0 | % B có P < 0,5 mà R > 0 | % B có P ≥ 0,5 mà R > 0 |', '|---|---|---|---|']
    for l, q in S['quadrant_main'].items():
        L.append(f'| {l} | {vn(q["H_R_pos_pct"], 1)} | {vn(q["B_lowP_R_pos_pct"], 1)} | {vn(q["B_highP_R_pos_pct"], 1)} |')
    L += ['', '## Vùng ứng viên (chưa tô, cách mask dự đoán ≤ r)', '', '| r | % pixel H nằm trong vùng (mọi ảnh · lesion to) | số ứng viên TB / ảnh |', '|---|---|---|']
    for r, v in S['candidate_zone'].items():
        L.append(f'| {r} | {vn(v["H_inside_pct"], 1)} · {vn(v["H_inside_pct_large"], 1)} | {vn(v["mean_candidates_px"], 0)} |')
    if sel:
        L += ['', f'## Oracle completion — cấu hình chọn trên val: {sel["ranker"]} (AUC gộp trên val {vn(sel["val_pooled_auc_R"], 3)})', '',
              f'Baseline: DSC TB theo ảnh {vn(S["oracle_baseline"]["dsc"])}; pixel H của lesion to {S["oracle_baseline"]["H_large"]:,}.', '',
              '| Luật (chọn trên val) | DSC TB (Δ, p) | precision · recall | DSC lesion to (Δ) | H lesion to được sửa | FP mới (cách GT > 3 px) | expansion precision · recall H | đổi diện tích |',
              '|---|---|---|---|---|---|---|---|']
        for lab, key in (('R, ngưỡng τ', sel['tau_rule_R']), ('P, ngưỡng τ', sel['tau_rule_P']), ('R, top-k %', sel['topk_rule_R']),
                         ('P, top-k %', sel['topk_rule_P']), ('ngẫu nhiên, top-k %', sel['topk_rule_random'])):
            if key is None or key not in S['oracle']:
                continue
            o = S['oracle'][key]
            L.append(f'| {lab}: `{key}` | {vn(o["dsc"])} ({vn(o["d_dsc"], 2, True)}, {vn(o["p_dsc"], 4)}) | {vn(o["precision"])} · {vn(o["recall"])} | '
                     f'{vn(o["dsc_large"])} ({vn(o["d_dsc_large"], 2, True)}) | {o["H_fixed"]:,} / {o["H_before"]:,} ({vn(o["interior_fn_reduction_pct"], 1)} %) | '
                     f'{o["new_fp"]:,} ({o["new_fp_far3"]:,}) | {vn(o["expansion_precision"], 1)} · {vn(o["expansion_recall_H"], 1)} | {vn(o["area_change_pct"], 1, True)} % |')
        L += ['', '### Expansion precision ở cùng recall (luật ngưỡng, r chọn trên val)', '']
        r_sel = int(sel['tau_rule_R'].split('|')[1][1:]) if sel['tau_rule_R'] else 16
        pts_r, pts_p = pr_points(S, sel['ranker'], r_sel, 'tau'), pr_points(S, 'P', r_sel, 'tau')
        pts_rk, pts_pk, pts_x = pr_points(S, sel['ranker'], r_sel, 'k'), pr_points(S, 'P', r_sel, 'k'), pr_points(S, 'random', r_sel, 'k')
        L += ['| recall H ≥ | R (ngưỡng) | P (ngưỡng) | R (top-k) | P (top-k) | ngẫu nhiên (top-k) |', '|---|---|---|---|---|---|']
        for tr in (5, 10, 20, 30, 40):
            L.append(f'| {tr} % | {vn(precision_at_recall(pts_r, tr), 1)} | {vn(precision_at_recall(pts_p, tr), 1)} | {vn(precision_at_recall(pts_rk, tr), 1)} | '
                     f'{vn(precision_at_recall(pts_pk, tr), 1)} | {vn(precision_at_recall(pts_x, tr), 1)} |')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n')


# ---------------------------------------------------------------------------------------------- plots / figures
PAL = {'P': '#2a78d6', 'R': '#eb6834', 'AL': '#1baf7a', 'AB': '#eda100', 'random': '#8a8984'}
LEVEL_COL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']


def make_plots(S, sel, samples, d, split):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    pd = os.path.join(d, 'plots')
    os.makedirs(pd, exist_ok=True)
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#52514e',
                         'axes.labelcolor': '#0b0b0b', 'xtick.color': '#52514e', 'ytick.color': '#52514e', 'figure.facecolor': '#fcfcfb',
                         'axes.facecolor': '#fcfcfb', 'grid.color': '#e6e5e1'})
    a = S['auc']
    # 1. AUC by level: pooled (all pixels) and per-image median, P vs A_L vs A_B vs R
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    x = np.arange(len(LEVEL_NAMES))
    for ax, (title, getter) in zip(axes, (('AUC gộp (mọi pixel)', lambda e, sig: e['pooled_all_px'] if sig == 'R' else e[f'{sig}_pooled_all_px']),
                                         ('AUC trung vị từng ảnh', lambda e, sig: e['per_image'].get('median', np.nan) if sig == 'R' else e[f'{sig}_per_image'].get('median', np.nan)))):
        for k, (sig, lab) in enumerate((('AL', 'A_L'), ('AB', 'A_B'), ('R', 'R'))):
            vals = [getter(a[f'R:main:{l}'], sig) for l in LEVEL_NAMES]
            ax.bar(x + (k - 1) * 0.26, vals, 0.24, color=PAL[sig], label=lab, edgecolor='#fcfcfb', linewidth=2)
        pv = a['P']['pooled_all_px'] if 'gộp' in title else a['P']['per_image'].get('median', np.nan)
        ax.axhline(pv, color=PAL['P'], lw=2, label='P')
        ax.text(len(LEVEL_NAMES) - 0.55, pv + 0.015, f'P {pv:.3f}'.replace('.', ','), color='#0b0b0b', ha='right', va='bottom', fontsize=8)
        ax.axhline(0.5, color='#a3a29d', lw=1, ls=':')
        ax.set_xticks(x, LEVEL_NAMES)
        ax.set_title(title)
        ax.set_ylim(0.0, 1.05)
        ax.grid(axis='y', lw=0.6)
    axes[0].set_ylabel('AUC (H so với B)')
    axes[1].legend(frameon=False, loc='lower right')
    fig.suptitle(f'{S["dataset"]} {split}: AUC của P, A_L, A_B, R theo tầng (cấu hình chính)')
    fig.tight_layout()
    fig.savefig(os.path.join(pd, f'auc_by_level_{split}.png'), dpi=150)
    plt.close(fig)
    # 2. P vs R density for H and B (capped samples), every level
    fig, axes = plt.subplots(2, len(LEVEL_NAMES), figsize=(2.6 * len(LEVEL_NAMES), 5), sharex=True, sharey=True)
    st, pv = samples['set'], samples['P']
    for j, l in enumerate(LEVEL_NAMES):
        rv = samples[f'R:main:{l}']
        for i, (s_id, lab) in enumerate(((0, 'H (lesion bị bỏ sót)'), (1, 'B (da 5–20 px)'))):
            ok = (st == s_id) & np.isfinite(rv)
            ax = axes[i, j]
            ax.hist2d(pv[ok], rv[ok], bins=[40, 40], range=[[0, 1], [-0.6, 0.6]], cmap='Blues', cmin=1)
            ax.axhline(0, color='#52514e', lw=0.8)
            ax.axvline(0.5, color='#52514e', lw=0.8, ls=':')
            if i == 0:
                ax.set_title(l)
            if j == 0:
                ax.set_ylabel(f'{lab}\nR')
            if i == 1:
                ax.set_xlabel('P')
    fig.suptitle(f'{S["dataset"]} {split}: P (ngang) và R (dọc) của pixel H và B')
    fig.tight_layout()
    fig.savefig(os.path.join(pd, f'density_P_R_{split}.png'), dpi=150)
    plt.close(fig)
    if not sel:
        return
    # 3. expansion precision-recall (selected R vs P vs random; R main per level) and fixed H vs new FP, per radius
    for r_sel in CAND_R:
        fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
        series = [(sel['ranker'], f'R ({sel["config"]}, {sel["level"]})', PAL['R']), ('P', 'P', PAL['P']), ('random', 'ngẫu nhiên', PAL['random'])]
        for rk, lab, col in series:
            for rule, ls in (('tau', '-'), ('k', '--')):
                pts = pr_points(S, rk, r_sel, rule)
                if not pts:
                    continue
                axes[0].plot([p[0] for p in pts], [p[1] for p in pts], ls, color=col, lw=2, marker='o', ms=4,
                             label=f'{lab}, {"ngưỡng" if rule == "tau" else "top-k %"}')
                keys = [f'{rk}|r{r_sel}|{p[2]}' for p in pts]
                axes[2].plot([S['oracle'][k]['new_fp'] / 1e3 for k in keys], [S['oracle'][k]['H_fixed'] / 1e3 for k in keys], ls, color=col, lw=2, marker='o', ms=4)
        for li, l in enumerate(LEVEL_NAMES):
            pts = pr_points(S, f'R:main:{l}', r_sel, 'tau')
            axes[1].plot([p[0] for p in pts], [p[1] for p in pts], '-', color=LEVEL_COL[li], lw=2, marker='o', ms=3, label=f'R main {l}')
        pts = pr_points(S, 'P', r_sel, 'tau')
        axes[1].plot([p[0] for p in pts], [p[1] for p in pts], '-', color='#0b0b0b', lw=1.5, ls=':', label='P')
        for ax in axes[:2]:
            ax.set_xlabel('expansion recall (% pixel H được thêm)')
            ax.set_ylabel('expansion precision (% pixel thêm thuộc GT)')
            ax.grid(lw=0.6)
            ax.legend(frameon=False, fontsize=7)
        axes[0].set_title('R chọn trên val, P, ngẫu nhiên')
        axes[1].set_title('R cấu hình chính theo tầng (luật ngưỡng)')
        axes[2].set_title('H lesion to được sửa so với FP mới')
        axes[2].set_xlabel('FP mới (nghìn pixel)')
        axes[2].set_ylabel('H lesion to được sửa (nghìn pixel)')
        axes[2].grid(lw=0.6)
        fig.suptitle(f'{S["dataset"]} {split}: oracle completion, vùng ứng viên r = {r_sel} px')
        fig.tight_layout()
        fig.savefig(os.path.join(pd, f'expansion_pr_r{r_sel}_{split}.png'), dpi=150)
        plt.close(fig)


def make_figures(a, out_dir, S, sel, rows, base, hard):
    '''8-panel figures for 9 hard images of the split: 3 highest / 3 median / 3 lowest per-image AUC of the selected R.'''
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    d = os.path.join(out_dir, a.dataset, 'figures')
    os.makedirs(d, exist_ok=True)
    rk_c, rk_l = sel['config'], sel['level']
    cand = [(r['AUC_relative_evidence'], r['image_id']) for r in rows if r['config'] == rk_c and r['feature_level'] == rk_l
            and np.isfinite(r['AUC_relative_evidence'])]
    names = [nm for nm in np.load(os.path.join(out_dir, a.dataset, f'raw_{a.split}.npz'))['names']]
    hard_names = {names[i] for i in hard}
    cand = sorted([c for c in cand if c[1] in hard_names])
    if not cand:
        return []
    m = len(cand) // 2
    picks = [('thấp nhất', c) for c in cand[:3]] + [('trung vị', c) for c in cand[max(0, m - 1):m + 2]] + [('cao nhất', c) for c in cand[-3:]]
    tau_key = sel['tau_rule_R']
    r_sel, tau = int(tau_key.split('|')[1][1:]), float(tau_key.split('|')[2][3:])
    run = a.checkpoint or RUN.format(ds=a.dataset)
    sd, _ = C.load_state_dict(C.resolve_checkpoint(run))
    model = C.build_model(sd, 'cpu', None, strict=True)[0]
    feats = {}
    lvl_mod = dict(LEVELS)[rk_l]
    h = getattr(model, lvl_mod).register_forward_pre_hook(lambda mod, inp: feats.__setitem__('f', inp[0].detach()))
    ds = split_dataset(DATA[a.dataset], C.make_config(a.dataset, 256), a.split)
    cfg = dict((c[0], c) for c in CONFIGS)[rk_c]
    out = []
    for tag, (auc_v, nm) in picks:
        i = names.index(nm)
        img, msk = ds[i]
        with torch.no_grad():
            prob = model(img.float()[None])[1][0, 0].numpy().astype(np.float64)
        gt = msk.float()[0].numpy() >= 0.5
        fm = F.interpolate(feats['f'], size=gt.shape, mode='bilinear', align_corners=True)[0]
        core, bhat, d_pred = prediction_regions(prob, cfg[1])
        ev = eval_regions(gt, prob)
        bg = bhat if cfg[3] == 'pred' else ev['B']
        _, _, R = relative_evidence(fm, core, bg, cfg[2])
        pred = prob >= 0.5
        selm = ~pred & (d_pred <= r_sel) & (R > tau)
        fig, axes = plt.subplots(1, 8, figsize=(16, 2.4))
        im = img.numpy().transpose(1, 2, 0)
        im = (im - im.min()) / max(im.max() - im.min(), 1e-8)
        panels = [(im, 'ảnh', None), (gt, 'GT', 'gray'), (pred, 'prediction', 'gray'), (prob, 'P', 'Blues'), (ev['H'], 'H', 'gray'),
                  (np.clip(R, -0.5, 0.5), f'R ({rk_l})', 'RdBu_r'), (selm, f'phần thêm (R > {tau}, r {r_sel})', 'gray'), (pred | selm, 'mask sau completion', 'gray')]
        for ax, (p, t, cm) in zip(axes, panels):
            ax.imshow(p, cmap=cm, vmin=(-0.5 if cm == 'RdBu_r' else None), vmax=(0.5 if cm == 'RdBu_r' else None))
            ax.set_title(t, fontsize=8)
            ax.axis('off')
        fig.suptitle(f'{a.dataset} {a.split} {nm}: AUC(R) {auc_v:.3f} ({tag} trong ảnh khó); |H| {int(ev["H"].sum())}, chọn {int(selm.sum())} px, '
                     f'trong đó thuộc GT {int((selm & gt).sum())}', fontsize=9)
        fig.tight_layout()
        fn = os.path.join(d, f'{a.split}_{tag.replace(" ", "_")}_{os.path.splitext(nm)[0]}.png')
        fig.savefig(fn, dpi=110)
        plt.close(fig)
        out.append({'image': nm, 'group': tag, 'auc_R': auc_v, 'file': fn})
    h.remove()
    return out


def main():
    a = parse_args()
    if not a.summarize_only:
        run_pass(a, a.out_dir)
    d = os.path.join(a.out_dir, a.dataset)
    S, rows, samples, oracle, base, hard = summarize(a, a.out_dir)
    sel = None
    if a.split == 'val':
        sel = select_on_val(S)
        with open(os.path.join(d, 'selection_val.json'), 'w') as f:
            json.dump(sel, f, indent=1)
    elif os.path.isfile(os.path.join(d, 'selection_val.json')):
        sel = json.load(open(os.path.join(d, 'selection_val.json')))
    S['selection'] = sel
    if sel:
        S['figures'] = make_figures(a, a.out_dir, S, sel, rows, base, hard) if a.split == 'test' else []
    make_plots(S, sel, samples, d, a.split)
    with open(os.path.join(d, f'summary_{a.split}.json'), 'w') as f:
        json.dump(S, f, indent=1, default=float)
    write_markdown(S, sel, os.path.join(d, f'summary_{a.split}.md'))
    print(f'wrote {d}/summary_{a.split}.json / .md, plots/')
    return 0


if __name__ == '__main__':
    sys.exit(main())
