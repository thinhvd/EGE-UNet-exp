'''
EXP-15 mechanism readouts and oracle routing on one split of the 70:10:20 protocol (test by default; the ground truth
is used only here, never by the models). Runs are given as LABEL=DIR, LABEL in E0..E5p (E0 original model, E1 plain
cross-stage fusion, E2..E5p lesion-state-aware fusion). The region and size definitions were fixed before the runs
(doc 15, step 0): lesion-area tertile edges from the train split (--area-edges), tau = 0.15 for the normalized depth,
the +-3 px contour ring, far background = more than 20 px outside the lesion.

Parts (--parts, default all):
 crit    every checkpoint of every run (selected = the val-loss checkpoint, SWA = checkpoints/swa-epoch200-300.pth, and
         the 11 checkpoints epoch200..300): per-image mean DSC / precision / recall; painted / true area of small and
         large lesions (geometric mean of the per-image ratio, in % above or below 1); interior-miss rate of large
         lesions (normalized depth >= tau, pooled); boundary F-score at 2 px (per-image mean). -> the LSA criterion
         table (selected and SWA; vs E1: small-lesion precision up and size ratio down AND large-lesion interior miss
         down, boundary F not lower; paired per-image t-tests) and the epoch-swing table (mean and SD over the 11
         epochs, difference of the means vs E0).
 gates   (LSA arms, --ckpt) mean gD / gC / gnull per stage and region; Spearman of gC with the lesion area (per image,
         inside the lesion) and of gC / gnull with the signed distance to the contour (pixels within 30 px of it);
         cosine between the two experts' contributions head_D,s(F_D) and head_C,s(F_C) per stage and region (they live
         in the same decoder channels; F_D and F_C themselves have unaligned channels); leverage = mean |f_s| / mean
         |D_s| per stage.
 force   (LSA arms) forced gates D (1,0,0) / C (0,1,0) / null (0,0,1) / equal (0.5,0.5,0) and knockouts of one stage /
         all stages; (E1) knockouts of the fusion heads. Per variant: per-image DSC, size ratios, interior miss, mean
         |change of the final logit at grid 128| in the ring.
 oracle  (LSA arms) variants {D only, C only, null, actual}. Oracle per pixel (a pixel is right when any variant gets
         it right: optimistic ceiling); per image x zone (zones: band = the ring, interior = lesion outside the ring,
         near background <= 20 px, far background; best of the 4^4 combinations per image); per image (best variant);
         one dataset-level rule (best variant per size tertile x zone, coordinate ascent; fit on this split, and
         2-fold cross-fit). Headroom = oracle - best of (D only, C only). The zone oracle is the stop-rule number
         (doc 15: headroom < 0.3 -> stop the LSA line). Control (E1): the same selection over {CSF on, CSF off, dec3
         off, dec5 off}, headroom vs the model itself: what zone-wise selection gains without any expert design (picking
         the best of 4^4 choices per image is biased upward even between near-identical variants). Caveat: a forced
         gate acts on every stage at once, and the pixel / zone oracles mix outputs of different forward passes.
 aux     (LSA arms) per-image DSC of the aux heads: detail seg, completion seg, detail ring channel vs the +-3 px ring
         (probe arms: what each expert encodes; supervise arms: how well each learned its task).

  python analysis/exp15_mechanism.py --dataset isic17 --data-path data/data_isic1718_701020/isic2017 \
      --run E0=results/egeunet_isic17_learnable_s42 --run E3=... --area-edges 0.10,0.25 --out-dir results/exp15_isic17/mechanism
'''
import os
import sys
import json
import argparse
import itertools

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from scipy.ndimage import distance_transform_edt, binary_erosion

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis import common as C                                   # noqa: E402
from datasets.dataset import NPY_datasets, split_dataset           # noqa: E402
from models.lsa_fusion import _resize                              # noqa: E402
import contour_losses as CL                                        # noqa: E402

STAGES = ('dec3', 'dec4', 'dec5')
FORCED = {'D': (1.0, 0.0, 0.0), 'C': (0.0, 1.0, 0.0), 'null': (0.0, 0.0, 1.0), 'equal': (0.5, 0.5, 0.0)}
VARIANTS = ('D', 'C', 'null', 'actual')
ZONES = ('band', 'interior', 'near_bg', 'far_bg')
TERT = ('small', 'mid', 'large')
SWA_NAME = 'swa-epoch200-300.pth'
EPOCHS = list(range(200, 301, 10))
PAIRS = [('E3', 'E2'), ('E4', 'E3'), ('E4', 'E5p'), ('E5p', 'E2')]   # router / expert supervision contrasts (doc 15, 2d)
SWING_KEYS = ('dsc', 'precision_small', 'size_small_pct', 'size_large_pct', 'interior_miss_large', 'bf2')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', required=True, choices=['isic17', 'isic18'])
    p.add_argument('--data-path', required=True)
    p.add_argument('--split', default='test')
    p.add_argument('--run', action='append', required=True, metavar='LABEL=DIR')
    p.add_argument('--area-edges', required=True, help='two lesion-area tertile edges (fraction of the image), step 0 S0-4')
    p.add_argument('--tau', type=float, default=0.15)
    p.add_argument('--ckpt', default='swa', choices=['swa', 'selected'], help='checkpoint of gates / force / oracle / aux')
    p.add_argument('--parts', default='crit,gates,force,oracle,aux')
    p.add_argument('--epochs', default='200,300,10', help='first,last,every of the saved checkpoints (SWA = their average)')
    p.add_argument('--device', default='cpu')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--max-images', type=int, default=None)
    p.add_argument('--out-dir', required=True)
    return p.parse_args()


# ---------------------------------------------------------------------------------------------------------- data
class Data:
    def __init__(self, ds, n, edges):
        imgs, gts = [], []
        for i in range(n):
            img, m = ds[i]
            imgs.append(img.float())
            gts.append(m[0].numpy() >= 0.5)
        self.img = torch.stack(imgs)
        self.gt = np.stack(gts)
        self.names = [os.path.basename(ds.data[i][0]) for i in range(n)]
        mt = torch.from_numpy(self.gt[:, None].astype(np.float32))
        self.depth = CL.normalized_depth(mt)[:, 0].numpy()
        self.ring = CL.boundary_ring(mt, 3)[:, 0].numpy() > 0.5
        self.ring128 = CL.band_at_grid(CL.boundary_ring(mt, 3), (128, 128))[:, 0].numpy() > 0.5
        big = float(sum(self.gt.shape[1:]))
        self.dout = np.stack([distance_transform_edt(~g) if g.any() else np.full(g.shape, big) for g in self.gt])
        self.din = np.stack([distance_transform_edt(g) for g in self.gt])
        self.area = self.gt.mean((1, 2))
        self.tert = np.digitize(self.area, edges)
        self.zone = np.full(self.gt.shape, 3, np.int8)
        self.zone[(~self.gt) & (self.dout <= 20)] = 2
        self.zone[self.gt] = 1
        self.zone[self.ring] = 0
        self.gt_edge = [edge(g) for g in self.gt]
        self.gt_edge_dist = [distance_transform_edt(~e) if e.any() else None for e in self.gt_edge]
        self.deep = (self.depth >= TAU[0]) & self.gt
        self.pred_e0 = None

    def __len__(self):
        return len(self.gt)

    def regions(self, sl):
        g, t, d, o = self.gt[sl], self.tert[sl][:, None, None], self.depth[sl], self.dout[sl]
        large = np.broadcast_to(t == 2, g.shape)
        small = np.broadcast_to(t == 0, g.shape)
        r = {'lesion': g, 'background': ~g, 'small_lesion': small & g, 'small_lesion_halo': small & (o <= 8),
             'band3': self.ring[sl], 'large_depth_0_.15': large & g & (d < 0.15), 'large_depth_.15_1': large & g & (d >= 0.15),
             'large_depth_.15_.5': large & g & (d >= 0.15) & (d < 0.5), 'large_depth_.5_1': large & g & (d >= 0.5),
             'far_bg_20px': o > 20}
        if self.pred_e0 is not None:
            r['large_missed_by_E0'] = large & g & ~self.pred_e0[sl]
        return r


TAU = [0.15]


def edge(b):
    '''Contour pixels of a mask (in the mask, with a 4-neighbour outside); the image frame is not a contour.'''
    return b & ~binary_erosion(b, border_value=1)


def bf_score(pred, gt_edge, gt_edge_dist, tol=2.0):
    pe = edge(pred)
    if not pe.any() and not gt_edge.any():
        return 1.0
    if not pe.any() or not gt_edge.any():
        return 0.0
    prec = float((gt_edge_dist[pe] <= tol).mean())
    rec = float((distance_transform_edt(~pe)[gt_edge] <= tol).mean())
    return 2 * prec * rec / (prec + rec) if prec + rec else 0.0


# ---------------------------------------------------------------------------------------------------------- models
def load_model(run, which, device):
    if which == 'selected':
        path = C.resolve_checkpoint(run)
    elif which == 'swa':
        path = os.path.join(run, 'checkpoints', SWA_NAME)
    else:
        path = os.path.join(run, 'checkpoints', f'epoch{which:03d}.pth')
    sd, _ = C.load_state_dict(path)
    return C.build_model(sd, device, None, strict=True)[0]


def is_lsa(model):
    return getattr(model, 'lsa_mode', 'none') != 'none'


def infer(model, data, device, bs, on_batch=None):
    '''Final probability of every image; on_batch(slice, gt_pre, prob) sees every batch.'''
    probs = np.empty(data.gt.shape, np.float32)
    with torch.no_grad():
        for i in range(0, len(data), bs):
            sl = slice(i, min(i + bs, len(data)))
            gp, out = model(data.img[sl].to(device))
            probs[sl] = out[:, 0].cpu().numpy()
            if on_batch is not None:
                on_batch(sl, gp, out)
    return probs


def counts(pred, gt):
    tp = (pred & gt).sum((1, 2)).astype(np.float64)
    fp = (pred & ~gt).sum((1, 2)).astype(np.float64)
    fn = (~pred & gt).sum((1, 2)).astype(np.float64)
    return tp, fp, fn


def dsc_arr(tp, fp, fn):
    den = 2 * tp + fp + fn
    return np.where(den > 0, 2 * tp / np.maximum(den, 1e-12), 1.0)


def per_image(pred, data, with_bf=True):
    g = data.gt
    tp, fp, fn = counts(pred, g)
    miss = (data.deep & ~pred).sum((1, 2)).astype(np.float64)
    deep = data.deep.sum((1, 2)).astype(np.float64)
    out = {'dsc': dsc_arr(tp, fp, fn), 'precision': np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), np.nan),
           'recall': np.where(tp + fn > 0, tp / np.maximum(tp + fn, 1), np.nan),
           'log_ratio': np.log(np.maximum(tp + fp, 1) / np.maximum(tp + fn, 1)), 'miss_deep': miss, 'deep': deep,
           'interior_miss_frac': np.where(deep > 0, miss / np.maximum(deep, 1), np.nan)}
    if with_bf:
        out['bf2'] = np.array([bf_score(pred[i], data.gt_edge[i], data.gt_edge_dist[i]) for i in range(len(g))])
    return out


def summarize(pi, data):
    s, l = data.tert == 0, data.tert == 2
    out = {'dsc': 100 * pi['dsc'].mean(), 'precision': 100 * np.nanmean(pi['precision']), 'recall': 100 * np.nanmean(pi['recall']),
           'dsc_small': 100 * pi['dsc'][s].mean(), 'dsc_large': 100 * pi['dsc'][l].mean(),
           'precision_small': 100 * np.nanmean(pi['precision'][s]), 'recall_large': 100 * np.nanmean(pi['recall'][l]),
           'size_small_pct': 100 * (np.exp(pi['log_ratio'][s].mean()) - 1),
           'size_large_pct': 100 * (np.exp(pi['log_ratio'][l].mean()) - 1),
           'interior_miss_large': 100 * pi['miss_deep'][l].sum() / max(pi['deep'][l].sum(), 1)}
    if 'bf2' in pi:
        out['bf2'] = 100 * pi['bf2'].mean()
    return {k: float(v) for k, v in out.items()}


def paired(v, r, mask=None, scale=100):
    ok = np.isfinite(v) & np.isfinite(r) & (np.ones(len(v), bool) if mask is None else mask)
    if ok.sum() < 3:
        return float('nan'), float('nan')
    return float(scale * (v[ok] - r[ok]).mean()), float(stats.ttest_rel(v[ok], r[ok]).pvalue)


# ---------------------------------------------------------------------------------------------------------- parts
def part_crit(runs, data, a, out):
    res, pis = {}, {}
    for lab, run in runs.items():
        res[lab] = {}
        for which in ['selected', 'swa'] + EPOCHS:
            name = str(which)
            try:
                model = load_model(run, which, a.device)
            except FileNotFoundError as e:
                print(f'  crit {lab} {name}: skipped ({e})')
                continue
            pred = infer(model, data, a.device, a.batch_size) >= 0.5
            pi = per_image(pred, data)
            res[lab][name] = summarize(pi, data)
            if which in ('selected', 'swa'):
                pis[(lab, name)] = pi
            if lab == 'E0' and which == a.ckpt:
                data.pred_e0 = pred
        ep = [res[lab][str(e)] for e in EPOCHS if str(e) in res[lab]]
        if ep:
            res[lab]['epoch_mean'] = {k: float(np.mean([m[k] for m in ep])) for k in ep[0]}
            res[lab]['epoch_sd'] = {k: float(np.std([m[k] for m in ep], ddof=1)) if len(ep) > 1 else float('nan') for k in ep[0]}
            res[lab]['n_epochs'] = len(ep)
        r = res[lab]
        print(f'  crit {lab}: DSC selected {r.get("selected", {}).get("dsc", float("nan")):.2f} | SWA '
              f'{r.get("swa", {}).get("dsc", float("nan")):.2f} | epoch mean {r.get("epoch_mean", {}).get("dsc", float("nan")):.2f} '
              f'(SD {r.get("epoch_sd", {}).get("dsc", float("nan")):.2f})', flush=True)
    tests = []
    small, large = data.tert == 0, data.tert == 2
    for which in ('selected', 'swa'):
        pairs = [(lab, ref) for ref in ('E1', 'E0') for lab in runs if lab not in (ref, 'E0')] + PAIRS
        for lab, ref in pairs:
            if (lab, which) not in pis or (ref, which) not in pis:
                continue
            v, r = pis[(lab, which)], pis[(ref, which)]
            row = {'checkpoint': which, 'variant': lab, 'reference': ref}
            row['d_precision_small'], row['p_precision_small'] = paired(v['precision'], r['precision'], small)
            row['d_log_ratio_small'], row['p_log_ratio_small'] = paired(v['log_ratio'], r['log_ratio'], small)
            row['d_interior_miss_large'], row['p_interior_miss_large'] = paired(v['interior_miss_frac'], r['interior_miss_frac'], large)
            row['d_bf2'], row['p_bf2'] = paired(v['bf2'], r['bf2'])
            row['d_dsc'], row['p_dsc'] = paired(v['dsc'], r['dsc'])
            row['lsa_criterion'] = bool(row['d_precision_small'] > 0 and row['d_log_ratio_small'] < 0
                                        and row['d_interior_miss_large'] < 0 and row['d_bf2'] >= 0)
            tests.append(row)
    swing = {}
    if 'E0' in res and 'epoch_mean' in res['E0']:
        for lab in res:
            if 'epoch_mean' in res[lab]:
                swing[lab] = {k: {'mean': res[lab]['epoch_mean'][k], 'sd': res[lab]['epoch_sd'][k],
                                  'mean_minus_E0': res[lab]['epoch_mean'][k] - res['E0']['epoch_mean'][k]} for k in SWING_KEYS}
    out['crit'] = {'runs': res, 'tests': tests, 'epoch_swing': swing}


class Recorder:
    '''Wraps model.lsa.stage to keep, per stage, the gates, both experts' contributions, the added feature and D_s.'''
    def __init__(self, lsa):
        self.lsa, self.store, self.orig = lsa, {}, lsa.stage

        def stage(s, fd, fc, d, q):
            f, g = self.orig(s, fd, fc, d, q)
            size = d.shape[2:4]
            self.store[s] = {'g': g, 'cd': lsa.head_d[s](_resize(fd, size)), 'cc': lsa.head_c[s](_resize(fc, size)), 'f': f, 'd': d}
            return f, g
        lsa.stage = stage

    def remove(self):
        del self.lsa.stage


def up256(x, size):
    return F.interpolate(x, size=size, mode='bilinear', align_corners=True)


def part_gates_aux(lab, model, data, a, out, do_gates, do_aux):
    rec = Recorder(model.lsa)
    acc = {s: {} for s in STAGES}
    lev = {s: np.zeros(4) for s in STAGES}            # sum |f| lesion, sum |d| lesion, sum |f| bg, sum |d| bg
    area_gc = {s: [] for s in STAGES}
    dist_samples = {s: ([], [], []) for s in STAGES}  # signed distance, gC, gnull
    aux = {'d_seg': [], 'c_seg': [], 'd_ring': []}
    rng = np.random.default_rng(0)
    size = data.gt.shape[1:]

    def on_batch(sl, gp, prob):
        regs = data.regions(sl)
        g_les = data.gt[sl]
        if do_gates:
            for s in STAGES:
                st = rec.store[s]
                g = up256(st['g'], size).cpu().numpy()
                cd, cc = st['cd'], st['cc']
                cos = (cd * cc).sum(1, keepdim=True) / (cd.norm(dim=1, keepdim=True) * cc.norm(dim=1, keepdim=True) + 1e-12)
                cos = up256(cos, size)[:, 0].cpu().numpy()
                fn_, dn_ = up256(st['f'].norm(dim=1, keepdim=True), size)[:, 0].cpu().numpy(), up256(st['d'].norm(dim=1, keepdim=True), size)[:, 0].cpu().numpy()
                lev[s] += [fn_[g_les].sum(), dn_[g_les].sum(), fn_[~g_les].sum(), dn_[~g_les].sum()]
                for rn, m in regs.items():
                    a_ = acc[s].setdefault(rn, np.zeros(5))
                    a_ += [g[:, 0][m].sum(), g[:, 1][m].sum(), g[:, 2][m].sum(), cos[m].sum(), m.sum()]
                for j, i in enumerate(range(sl.start, sl.stop)):
                    if data.gt[i].any():
                        area_gc[s].append((data.area[i], float(g[j, 1][data.gt[i]].mean())))
                    sd_ = data.din[i] - data.dout[i]
                    idx = np.flatnonzero(np.abs(sd_).ravel() <= 30)
                    if len(idx):
                        idx = rng.choice(idx, min(500, len(idx)), replace=False)
                        dist_samples[s][0].append(sd_.ravel()[idx])
                        dist_samples[s][1].append(g[j, 1].ravel()[idx])
                        dist_samples[s][2].append(g[j, 2].ravel()[idx])
        if do_aux:
            bnd = gp.boundary
            d = torch.sigmoid(up256(bnd['lsa_d'], size)).cpu().numpy() >= 0.5
            c = torch.sigmoid(up256(bnd['lsa_c'], size)).cpu().numpy() >= 0.5
            gt, ring = data.gt[sl], data.ring[sl]
            aux['d_seg'].append(dsc_arr(*counts(d[:, 0], gt)))
            aux['c_seg'].append(dsc_arr(*counts(c[:, 0], gt)))
            aux['d_ring'].append(dsc_arr(*counts(d[:, 1], ring)))

    infer(model, data, a.device, a.batch_size, on_batch)
    rec.remove()
    if do_gates:
        res = {}
        for s in STAGES:
            r = {rn: {'gD': v[0] / max(v[4], 1), 'gC': v[1] / max(v[4], 1), 'gnull': v[2] / max(v[4], 1),
                      'cos_contrib': v[3] / max(v[4], 1), 'n_px': int(v[4])} for rn, v in acc[s].items()}
            ag = np.array(area_gc[s])
            sdist, gc, gn = (np.concatenate(x) if x else np.zeros(0) for x in dist_samples[s])
            r['_spearman_gC_vs_area'] = float(stats.spearmanr(ag[:, 0], ag[:, 1])[0]) if len(ag) > 2 else float('nan')
            r['_spearman_gC_vs_signed_dist'] = float(stats.spearmanr(sdist, gc)[0]) if len(sdist) > 2 else float('nan')
            r['_spearman_gnull_vs_signed_dist'] = float(stats.spearmanr(sdist, gn)[0]) if len(sdist) > 2 else float('nan')
            L = lev[s]
            r['_leverage_lesion'] = float(L[0] / max(L[1], 1e-12))
            r['_leverage_bg'] = float(L[2] / max(L[3], 1e-12))
            res[s] = r
            print(f'  gates {lab} {s}: lesion gD/gC/gnull {r["lesion"]["gD"]:.3f}/{r["lesion"]["gC"]:.3f}/{r["lesion"]["gnull"]:.3f}; '
                  f'far bg {r["far_bg_20px"]["gD"]:.3f}/{r["far_bg_20px"]["gC"]:.3f}/{r["far_bg_20px"]["gnull"]:.3f}; '
                  f'leverage {r["_leverage_lesion"]:.3f}/{r["_leverage_bg"]:.3f}; rho(gC, area) {r["_spearman_gC_vs_area"]:+.2f}', flush=True)
        out.setdefault('gates', {})[lab] = res
    if do_aux:
        r = {k: 100 * float(np.concatenate(v).mean()) for k, v in aux.items()}
        out.setdefault('aux', {})[lab] = r
        print(f'  aux {lab}: DSC detail seg {r["d_seg"]:.2f}, completion seg {r["c_seg"]:.2f}, detail ring {r["d_ring"]:.2f}', flush=True)


def final_logits(model, data, a):
    zs = np.empty((len(data), 128, 128), np.float32)
    box = {}
    h = model.final.register_forward_hook(lambda m, i, o: box.__setitem__('z', o.detach()))

    def on_batch(sl, gp, prob):
        zs[sl] = box['z'][:, 0].cpu().numpy()
    probs = infer(model, data, a.device, a.batch_size, on_batch)
    h.remove()
    return probs, zs


def part_force(lab, model, data, a, out):
    variants = []
    if is_lsa(model):
        variants = [('force_' + k, ('force', v)) for k, v in FORCED.items() if not (k == 'equal' and model.lsa.mode == 'fixed')]
        variants += [(f'off_{s}', ('off', {s})) for s in STAGES] + [('off_all', ('off', set(STAGES)))]
    elif hasattr(model, 'fusion') and hasattr(model.fusion, 'heads'):
        variants = [(f'off_{s}', ('zero', [s])) for s in model.fusion.heads] + [('off_all', ('zero', list(model.fusion.heads)))]
    if not variants:
        return
    p0, z0 = final_logits(model, data, a)
    base = summarize(per_image(p0 >= 0.5, data, with_bf=False), data)
    res = {'actual': base}
    for name, (kind, arg) in variants:
        saved = {}
        if kind == 'force':
            model.lsa.force_gates = arg
        elif kind == 'off':
            model.lsa.stage_off = set(arg)
        else:
            for s in arg:
                h = model.fusion.heads[s]
                saved[s] = (h.weight.data.clone(), h.bias.data.clone())
                h.weight.data.zero_()
                h.bias.data.zero_()
        p, z = final_logits(model, data, a)
        if kind == 'force':
            model.lsa.force_gates = None
        elif kind == 'off':
            model.lsa.stage_off = set()
        else:
            for s, (w, b) in saved.items():
                model.fusion.heads[s].weight.data.copy_(w)
                model.fusion.heads[s].bias.data.copy_(b)
        m = summarize(per_image(p >= 0.5, data, with_bf=False), data)
        m['mean_abs_dz_ring'] = float(np.abs(z - z0)[data.ring128].mean())
        m['delta_dsc'] = m['dsc'] - base['dsc']
        res[name] = m
        print(f'  force {lab} {name}: DSC {m["dsc"]:.2f} ({m["delta_dsc"]:+.2f}), small size {m["size_small_pct"]:+.1f} %, large size '
              f'{m["size_large_pct"]:+.1f} %, |dz| ring {m["mean_abs_dz_ring"]:.3f}', flush=True)
    res['_mean_abs_z_ring'] = float(np.abs(z0)[data.ring128].mean())
    out.setdefault('force', {})[lab] = res


def zone_counts(pred, data, sl):
    '''(B, 4 zones, 3) tp / fp / fn.'''
    g, z = data.gt[sl], data.zone[sl]
    res = np.zeros((len(g), len(ZONES), 3))
    for k in range(len(ZONES)):
        m = z == k
        res[:, k, 0] = (pred & g & m).sum((1, 2))
        res[:, k, 1] = (pred & ~g & m).sum((1, 2))
        res[:, k, 2] = (~pred & g & m).sum((1, 2))
    return res


def dsc3(c):
    return dsc_arr(c[..., 0], c[..., 1], c[..., 2])


def rule_score(cnt, tert, rule):
    '''cnt (V, N, Z, 3); rule (3 tertiles, Z) variant index -> per-image DSC.'''
    n = cnt.shape[1]
    pick = rule[tert]                                               # (N, Z)
    c = cnt[pick, np.arange(n)[:, None], np.arange(cnt.shape[2])[None]].sum(1)
    return dsc3(c)


def fit_rule(cnt, tert, start):
    rule = np.full((3, cnt.shape[2]), start, int)
    best = rule_score(cnt, tert, rule).mean()
    for _ in range(10):
        changed = False
        for t in range(3):
            for z in range(cnt.shape[2]):
                for v in range(cnt.shape[0]):
                    if v == rule[t, z]:
                        continue
                    r2 = rule.copy()
                    r2[t, z] = v
                    s = rule_score(cnt, tert, r2).mean()
                    if s > best + 1e-12:
                        rule, best, changed = r2, s, True
        if not changed:
            break
    return rule


CSF_VARIANTS = ('actual', 'off_all', 'off_dec3', 'off_dec5')


def set_variant(model, v):
    """Apply oracle variant v; returns a function that undoes it. LSA: forced gates; plain CSF (control): zeroed heads."""
    if is_lsa(model):
        model.lsa.force_gates = None if v == 'actual' else FORCED[v]

        def undo():
            model.lsa.force_gates = None
        return undo
    stages = [] if v == 'actual' else (list(model.fusion.heads) if v == 'off_all' else [v[4:]])
    saved = {s: (model.fusion.heads[s].weight.data.clone(), model.fusion.heads[s].bias.data.clone()) for s in stages}
    for s in stages:
        model.fusion.heads[s].weight.data.zero_()
        model.fusion.heads[s].bias.data.zero_()

    def undo():
        for s, (w, b) in saved.items():
            model.fusion.heads[s].weight.data.copy_(w)
            model.fusion.heads[s].bias.data.copy_(b)
    return undo


def part_oracle(lab, model, data, a, out):
    """LSA arms: variants D / C / null / actual, headroom vs the best of D only / C only. E1 (control): the same
    selection over {CSF on, CSF off, dec3 off, dec5 off} (4 variants, so the same 4^4 choices per image), headroom vs
    the model itself: how much zone-wise selection gains without any expert design (the floor the LSA headroom is
    read against)."""
    variants = VARIANTS if is_lsa(model) else CSF_VARIANTS
    n = len(data)
    cnt = np.zeros((len(variants), n, len(ZONES), 3))
    pix = np.zeros((n, 3))
    with torch.no_grad():
        for i in range(0, n, a.batch_size):
            sl = slice(i, min(i + a.batch_size, n))
            x = data.img[sl].to(a.device)
            preds = []
            for vi, v in enumerate(variants):
                undo = set_variant(model, v)
                p = model(x)[1][:, 0].cpu().numpy() >= 0.5
                undo()
                preds.append(p)
                cnt[vi, sl] = zone_counts(p, data, sl)
            g = data.gt[sl]
            anyp, allp = np.logical_or.reduce(preds), np.logical_and.reduce(preds)
            pix[sl, 0] = (g & anyp).sum((1, 2))
            pix[sl, 1] = (~g & allp).sum((1, 2))
            pix[sl, 2] = (g & ~anyp).sum((1, 2))
    single = {v: dsc3(cnt[vi].sum(1)) for vi, v in enumerate(variants)}
    combos = np.array(list(itertools.product(range(len(variants)), repeat=len(ZONES))))      # (256, Z)
    zone_best = np.empty(n)
    for i in range(n):
        c = cnt[combos, i, np.arange(len(ZONES))[None]].sum(1)                                  # (256, 3)
        zone_best[i] = dsc3(c).max()
    image_best = np.max(np.stack([single[v] for v in variants]), 0)
    pixel = dsc3(pix)
    act = variants.index('actual')
    rule = fit_rule(cnt, data.tert, act)
    rule_fit = rule_score(cnt, data.tert, rule)
    half = np.arange(n) % 2
    cross = np.empty(n)
    for f in (0, 1):
        fi, ti = half == f, half != f
        r2 = fit_rule(cnt[:, fi], data.tert[fi], act)
        cross[ti] = rule_score(cnt[:, ti], data.tert[ti], r2)
    m = lambda x: 100 * float(np.mean(x))
    ref_names = ('D', 'C') if is_lsa(model) else ('actual',)
    best_single = max(m(single[v]) for v in ref_names)
    res = {'variants': list(variants), 'reference': '+'.join(ref_names), 'single': {v: m(single[v]) for v in variants},
           'best_single': best_single, 'oracle_pixel': m(pixel), 'oracle_zone': m(zone_best), 'oracle_image': m(image_best),
           'rule_fit_on_split': m(rule_fit), 'rule_2fold': m(cross),
           'rule': {TERT[t]: {ZONES[z]: variants[rule[t, z]] for z in range(len(ZONES))} for t in range(3)}}
    for k in ('oracle_pixel', 'oracle_zone', 'oracle_image', 'rule_fit_on_split', 'rule_2fold'):
        res[f'headroom_{k}'] = res[k] - best_single
    res['actual_minus_best_single'] = res['single']['actual'] - best_single
    res['headroom_oracle_zone_by_tertile'] = {TERT[t]: 100 * float(zone_best[data.tert == t].mean() - max(
        single[v][data.tert == t].mean() for v in ref_names)) for t in range(3) if (data.tert == t).any()}
    out.setdefault('oracle', {})[lab] = res
    print(f'  oracle {lab}: ' + ' '.join(f'{v} {res["single"][v]:.2f}' for v in variants)
          + f' | headroom zone {res["headroom_oracle_zone"]:+.2f}, image {res["headroom_oracle_image"]:+.2f}, '
          f'pixel {res["headroom_oracle_pixel"]:+.2f}, rule {res["headroom_rule_fit_on_split"]:+.2f} (2-fold {res["headroom_rule_2fold"]:+.2f})', flush=True)


# ---------------------------------------------------------------------------------------------------------- report
def vn(x, d=2, sign=False):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return '—'
    return (f'{x:+.{d}f}' if sign else f'{x:.{d}f}').replace('.', ',')


def report(out, a):
    st = lambda p: '' if not np.isfinite(p) else ('***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else '')
    md = [f'# EXP-15 cơ chế ({a.dataset}, tập {a.split}, {out["n_images"]} ảnh; model cho cổng / ép cổng / oracle: {a.ckpt})', '',
          f'Nhóm kích thước theo diện tích lesion (mốc từ train): nhỏ < {out["area_edges"][0]:.4f} ≤ vừa < {out["area_edges"][1]:.4f} ≤ to. '
          'Tỉ lệ tô / thật: trung bình nhân của (diện tích tô / diện tích thật) theo ảnh, ghi bằng % lệch khỏi 1. '
          f'Bỏ sót interior: pixel lesion to có độ sâu chuẩn hóa ≥ {a.tau} mà model không tô. BF 2 px: boundary F-score với dung sai 2 px.', '']
    if 'crit' in out:
        c = out['crit']
        for which in ('selected', 'swa', 'epoch_mean'):
            md += [f'## Tiêu chí LSA — {which}', '',
                   '| Run | DSC TB | precision lesion nhỏ | tô/thật nhỏ | tô/thật to | bỏ sót interior to | recall to | BF 2 px |', '|---|---|---|---|---|---|---|---|']
            for lab, r in c['runs'].items():
                if which in r:
                    s = r[which]
                    md.append(f'| {lab} | {vn(s["dsc"])} | {vn(s["precision_small"])} | {vn(s["size_small_pct"], 1, True)} % | '
                              f'{vn(s["size_large_pct"], 1, True)} % | {vn(s["interior_miss_large"])} % | {vn(s["recall_large"])} | {vn(s.get("bf2"))} |')
            md.append('')
        if c['tests']:
            md += ['## Paired t-test theo ảnh cho tiêu chí LSA (Δ = run − mốc)', '',
                   '| Checkpoint | Run | Mốc | Δ DSC | Δ precision nhỏ | Δ log(tô/thật) nhỏ ×100 | Δ bỏ sót interior to (điểm %) | Δ BF 2 px | đạt tiêu chí |',
                   '|---|---|---|---|---|---|---|---|---|']
            for t in c['tests']:
                md.append(f'| {t["checkpoint"]} | {t["variant"]} | {t["reference"]} | {vn(t["d_dsc"], 2, True)}{st(t["p_dsc"])} | '
                          f'{vn(t["d_precision_small"], 2, True)}{st(t["p_precision_small"])} | {vn(t["d_log_ratio_small"], 2, True)}{st(t["p_log_ratio_small"])} | '
                          f'{vn(t["d_interior_miss_large"], 2, True)}{st(t["p_interior_miss_large"])} | {vn(t["d_bf2"], 2, True)}{st(t["p_bf2"])} | '
                          f'{"có" if t["lsa_criterion"] else "không"} |')
            md.append('')
        if c['epoch_swing']:
            md += [f'## Dao động qua {len(EPOCHS)} checkpoint epoch {EPOCHS[0]}–{EPOCHS[-1]} (TB ± SD; Δ TB so với E0)', '',
                   '| Run | ' + ' | '.join(SWING_KEYS) + ' |', '|---|' + '---|' * len(SWING_KEYS)]
            for lab, r in c['epoch_swing'].items():
                md.append(f'| {lab} | ' + ' | '.join(f'{vn(r[k]["mean"])} ± {vn(r[k]["sd"])} ({vn(r[k]["mean_minus_E0"], 2, True)})' for k in SWING_KEYS) + ' |')
            md.append('')
    if 'gates' in out:
        md += ['## Cổng trung bình theo vùng (gD / gC / gnull; cos = cosine giữa phần đóng góp của hai expert)', '']
        for lab, r in out['gates'].items():
            md += [f'### {lab}', '', '| Vùng | ' + ' | '.join(STAGES) + ' |', '|---|' + '---|' * len(STAGES)]
            for rn in [k for k in r['dec5'] if not k.startswith('_')]:
                cell = lambda x: '—' if not x['n_px'] else f'{vn(x["gD"])} / {vn(x["gC"])} / {vn(x["gnull"])} (cos {vn(x["cos_contrib"], 2, True)})'
                md.append(f'| {rn} | ' + ' | '.join(cell(r[s][rn]) for s in STAGES) + ' |')
            for k in ('_spearman_gC_vs_area', '_spearman_gC_vs_signed_dist', '_spearman_gnull_vs_signed_dist', '_leverage_lesion', '_leverage_bg'):
                md.append(f'| {k[1:]} | ' + ' | '.join(vn(r[s][k], 3, True) for s in STAGES) + ' |')
            md.append('')
    if 'force' in out:
        md += ['## Ép cổng / tắt tầng (Δ DSC so với model thật; |Δ logit| ở dải ±3 px, lưới 128)', '']
        for lab, r in out['force'].items():
            md += [f'### {lab} (|logit| TB ở dải: {vn(r["_mean_abs_z_ring"])})', '',
                   '| Biến thể | DSC TB | Δ DSC | tô/thật nhỏ | tô/thật to | bỏ sót interior to | |Δ logit| dải |', '|---|---|---|---|---|---|---|']
            for name, m in r.items():
                if name.startswith('_'):
                    continue
                md.append(f'| {name} | {vn(m["dsc"])} | {vn(m.get("delta_dsc", 0.0), 2, True)} | {vn(m["size_small_pct"], 1, True)} % | '
                          f'{vn(m["size_large_pct"], 1, True)} % | {vn(m["interior_miss_large"])} % | {vn(m.get("mean_abs_dz_ring", 0.0), 3)} |')
            md.append('')
    if 'oracle' in out:
        lsa_rows = {k: v for k, v in out['oracle'].items() if v['reference'] == 'D+C'}
        ctl_rows = {k: v for k, v in out['oracle'].items() if v['reference'] != 'D+C'}
        hr = lambda r, k: f'{vn(r[k])} ({vn(r["headroom_" + k], 2, True)})'
        md += ['## Oracle routing (dùng GT chỉ trong phân tích; headroom = oracle − tốt nhất của chỉ-D / chỉ-C)', '',
               '| Run | chỉ D | chỉ C | null | cổng thật | oracle pixel | oracle vùng (luật dừng) | oracle ảnh | luật theo nhóm × vùng (fit / 2-fold) |',
               '|---|---|---|---|---|---|---|---|---|']
        for lab, r in lsa_rows.items():
            s_ = r['single']
            md.append(f'| {lab} | {vn(s_["D"])} | {vn(s_["C"])} | {vn(s_["null"])} | {vn(s_["actual"])} | {hr(r, "oracle_pixel")} | '
                      f'**{hr(r, "oracle_zone")}** | {hr(r, "oracle_image")} | {hr(r, "rule_fit_on_split")} / {hr(r, "rule_2fold")} |')
        for lab, r in ctl_rows.items():
            s_ = r['single']
            md += ['', f'Đối chứng {lab} (CSF thường, không có expert): cùng cách chọn trên 4 biến thể '
                   + ', '.join(f'{v} {vn(s_[v])}' for v in r['variants']) + f'; headroom so với chính model: oracle pixel '
                   f'{vn(r["headroom_oracle_pixel"], 2, True)}, oracle vùng **{vn(r["headroom_oracle_zone"], 2, True)}**, oracle ảnh '
                   f'{vn(r["headroom_oracle_image"], 2, True)}, luật {vn(r["headroom_rule_fit_on_split"], 2, True)} / 2-fold '
                   f'{vn(r["headroom_rule_2fold"], 2, True)}. Headroom của LSA chỉ có nghĩa khi lớn hơn mức này.']
        md.append('')
        for lab, r in out['oracle'].items():
            md.append(f'- {lab}: luật chọn theo nhóm × vùng = ' + '; '.join(f'{t}: ' + ', '.join(f'{z} {v}' for z, v in zz.items()) for t, zz in r['rule'].items()))
        md.append('')
    if 'aux' in out:
        md += ['## Aux head của expert (DSC TB theo ảnh)', '', '| Run | detail seg | completion seg | detail ring ±3 px |', '|---|---|---|---|']
        for lab, r in out['aux'].items():
            md.append(f'| {lab} | {vn(r["d_seg"])} | {vn(r["c_seg"])} | {vn(r["d_ring"])} |')
        md.append('')
    return '\n'.join(md) + '\n'


def main():
    a = parse_args()
    TAU[0] = a.tau
    first, last, every = (int(x) for x in a.epochs.split(','))
    EPOCHS[:] = list(range(first, last + 1, every))
    global SWA_NAME
    SWA_NAME = f'swa-epoch{first}-{last}.pth'
    parts = set(a.parts.split(','))
    edges = [float(x) for x in a.area_edges.split(',')]
    config = C.make_config(a.dataset, 256)
    ds = NPY_datasets(a.data_path, config, train=False) if a.split == 'val' else split_dataset(a.data_path, config, a.split)
    n = len(ds) if a.max_images is None else min(a.max_images, len(ds))
    data = Data(ds, n, edges)
    runs = dict(s.split('=', 1) for s in a.run)
    out = {'dataset': a.dataset, 'split': a.split, 'n_images': n, 'area_edges': edges, 'tau': a.tau, 'ckpt': a.ckpt,
           'tertile_counts': np.bincount(data.tert, minlength=3).tolist(), 'runs': runs}
    print(f'{a.dataset} {a.split}: {n} images, tertiles {out["tertile_counts"]}')
    if 'crit' in parts:
        part_crit(runs, data, a, out)
    elif 'E0' in runs and 'gates' in parts:
        data.pred_e0 = infer(load_model(runs['E0'], a.ckpt, a.device), data, a.device, a.batch_size) >= 0.5
    for lab, run in runs.items():
        if not ({'gates', 'force', 'oracle', 'aux'} & parts) or lab == 'E0':
            continue
        try:
            model = load_model(run, a.ckpt, a.device)
        except FileNotFoundError as e:
            print(f'  {lab}: skipped ({e})')
            continue
        if is_lsa(model) and ({'gates', 'aux'} & parts):
            part_gates_aux(lab, model, data, a, out, 'gates' in parts, 'aux' in parts)
        if 'force' in parts:
            part_force(lab, model, data, a, out)
        if 'oracle' in parts and (is_lsa(model) or lab == 'E1'):
            part_oracle(lab, model, data, a, out)
    os.makedirs(a.out_dir, exist_ok=True)
    tag = f'{a.split}_{a.ckpt}'
    with open(os.path.join(a.out_dir, f'mechanism_{tag}.json'), 'w') as f:
        json.dump(out, f, indent=1, default=float)
    with open(os.path.join(a.out_dir, f'mechanism_{tag}.md'), 'w', encoding='utf-8') as f:
        f.write(report(out, a))
    print(f'wrote {a.out_dir}/mechanism_{tag}.json / .md')
    return 0


if __name__ == '__main__':
    sys.exit(main())
