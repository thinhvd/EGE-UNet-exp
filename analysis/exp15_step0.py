'''
EXP-15 step 0 (CPU; train / VAL splits of the 70:10:20 protocol only, the test split is not read), on the EXP-14 runs.

S0-1  Selected checkpoint vs SWA (uniform average of epochs 200..300) vs the mean of the metrics over those 11
      checkpoints, for every EXP-14 run: per-image DSC / precision / recall, painted / true area of small and large
      lesions. Does SWA reproduce the late-epoch direction of CSF vs the original model ("CSF tightens every mask")?
S0-2  Per-stage knockout of the plain fusion (EXP-14 A1, selected checkpoint and SWA): zero the fusion head of dec3 /
      dec4 / dec5 one at a time and all together; change of the dec5 logit inside the +-3 px contour ring (cells at the
      128 grid), of the per-image DSC and of the painted / true area per size tertile.
S0-3  A zero-GPU competitor: a threshold that depends on the predicted lesion size. The images are binned by the area
      predicted at 0.5 (quintiles); per bin the threshold in {0.1 .. 0.9} maximizing the per-image mean DSC is chosen
      (a) on the train split, (b) on one half of val and scored on the other (2-fold). Reported for EXP-14 A0 and A1.
S0-4  Region definitions fixed from train: lesion-area tertile edges (fraction of the image).

  python analysis/exp15_step0.py --dataset isic17 --out results/exp15_step0/isic17.json
'''
import os
import sys
import json
import argparse

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis import common as C                                   # noqa: E402
from analysis.swa_average import average_state_dicts               # noqa: E402
from datasets.dataset import split_dataset                         # noqa: E402
import contour_losses as CL                                        # noqa: E402

SUB = {'isic17': 'isic2017', 'isic18': 'isic2018'}
ARMS = [('A0', 'learnable'), ('A1', 'learnable_fuse-sum-shallow3-d8'), ('A2', 'learnable_fuse-sum-shallow3-d8_loss-ihp'),
        ('A3', 'learnable_fuse-sum-shallow3-d8_loss-focal'), ('A4', 'learnable_fuse-sum-shallow3-d8_loss-tversky')]
THRS = [round(0.1 * k, 1) for k in range(1, 10)]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', required=True, choices=['isic17', 'isic18'])
    p.add_argument('--results', default='results/EGE-UNet-results-exp14')
    p.add_argument('--parts', default='1,2,3,4')
    p.add_argument('--max-images', type=int, default=None)
    p.add_argument('--out', required=True)
    return p.parse_args()


def load_sd(path):
    sd = torch.load(path, map_location='cpu', weights_only=False)
    return sd['model_state_dict'] if 'model_state_dict' in sd else sd


def model_from(sd):
    return C.build_model(sd, 'cpu', None, strict=True)[0]


def probs(model, data):
    with torch.no_grad():
        return [model(img.float()[None])[1][0, 0].numpy() for img, _ in data]


def metrics(prob_list, gts, tert, thr=0.5):
    dsc, prec, rec, ratio = [], [], [], []
    for p, g in zip(prob_list, gts):
        b = p >= thr
        tp, fp, fn = (b & g).sum(), (b & ~g).sum(), (~b & g).sum()
        dsc.append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 1.0)
        prec.append(tp / (tp + fp) if tp + fp else np.nan)
        rec.append(tp / (tp + fn) if tp + fn else np.nan)
        ratio.append(np.log(max(tp + fp, 1) / max(tp + fn, 1)))
    dsc, ratio = np.array(dsc), np.array(ratio)
    return {'dsc': 100 * dsc.mean(), 'precision': 100 * np.nanmean(prec), 'recall': 100 * np.nanmean(rec),
            'size_small_pct': 100 * (np.exp(ratio[tert == 0].mean()) - 1), 'size_large_pct': 100 * (np.exp(ratio[tert == 2].mean()) - 1)}


def main():
    a = parse_args()
    parts = set(a.parts.split(','))
    config = C.make_config(a.dataset, 256)
    root = os.path.join('data/data_isic1718_701020', SUB[a.dataset])
    ds_tr = split_dataset(root, config, 'train')
    ds_va = split_dataset(root, config, 'val')
    n_tr = len(ds_tr) if a.max_images is None else min(a.max_images, len(ds_tr))
    n_va = len(ds_va) if a.max_images is None else min(a.max_images, len(ds_va))
    gt_area_tr = np.array([float((ds_tr[i][1][0].numpy() >= 0.5).mean()) for i in range(n_tr)])
    edges = np.quantile(gt_area_tr, [1 / 3, 2 / 3]).tolist()
    va = [ds_va[i] for i in range(n_va)]
    gts = [m[0].numpy() >= 0.5 for _, m in va]
    tert = np.digitize(np.array([g.mean() for g in gts]), edges)
    out = {'dataset': a.dataset, 'S0-4': {'area_tertile_edges_from_train': edges, 'n_train': n_tr,
                                          'val_tertile_counts': np.bincount(tert, minlength=3).tolist()}}
    print(f'S0-4 {a.dataset}: lesion-area tertile edges from train = {edges[0]:.4f} / {edges[1]:.4f} of the image; '
          f'val images per tertile {out["S0-4"]["val_tertile_counts"]}')
    runs = {lab: os.path.join(a.results, f'egeunet_{a.dataset}_{arm}_s42') for lab, arm in ARMS}
    cache = {}

    if '1' in parts:
        res = {}
        for lab, run in runs.items():
            ck = os.path.join(run, 'checkpoints')
            epochs = [load_sd(os.path.join(ck, f'epoch{e:03d}.pth')) for e in range(200, 301, 10)]
            sel = load_sd(C.resolve_checkpoint(run))
            swa = average_state_dicts(epochs)
            p_sel, p_swa = probs(model_from(sel), va), probs(model_from(swa), va)
            per_epoch = [metrics(probs(model_from(sd), va), gts, tert) for sd in epochs]
            mean_ep = {k: float(np.mean([m[k] for m in per_epoch])) for k in per_epoch[0]}
            sd_ep = {k: float(np.std([m[k] for m in per_epoch], ddof=1)) for k in per_epoch[0]}
            res[lab] = {'selected': metrics(p_sel, gts, tert), 'swa': metrics(p_swa, gts, tert), 'epoch_mean': mean_ep, 'epoch_sd': sd_ep}
            cache[(lab, 'selected')], cache[(lab, 'swa')] = (sel, p_sel), (swa, p_swa)
            r = res[lab]
            print(f'S0-1 {a.dataset} {lab}: DSC selected {r["selected"]["dsc"]:.2f} | SWA {r["swa"]["dsc"]:.2f} | epoch mean '
                  f'{mean_ep["dsc"]:.2f} (SD {sd_ep["dsc"]:.2f}); small size {r["selected"]["size_small_pct"]:+.1f} / '
                  f'{r["swa"]["size_small_pct"]:+.1f} / {mean_ep["size_small_pct"]:+.1f} %; large size {r["selected"]["size_large_pct"]:+.1f} / '
                  f'{r["swa"]["size_large_pct"]:+.1f} / {mean_ep["size_large_pct"]:+.1f} %', flush=True)
        diff = {how: {k: res['A1'][how][k] - res['A0'][how][k] for k in res['A0'][how]} for how in ('selected', 'swa', 'epoch_mean')}
        out['S0-1'] = {'runs': res, 'A1_minus_A0': diff}
        for how, d in diff.items():
            print(f'S0-1 {a.dataset} CSF - original ({how}): ' + ', '.join(f'{k} {v:+.2f}' for k, v in d.items()))

    if '2' in parts:
        res2 = {}
        for how in ('selected', 'swa'):
            sd = cache[('A1', how)][0] if ('A1', how) in cache else (load_sd(C.resolve_checkpoint(runs['A1'])) if how == 'selected'
                                                                       else average_state_dicts([load_sd(os.path.join(runs['A1'], 'checkpoints', f'epoch{e:03d}.pth')) for e in range(200, 301, 10)]))
            model = model_from(sd)
            rec = {}
            model.final.register_forward_hook(lambda m, i, o: rec.__setitem__('z', o.detach()))
            base_z, base_p = [], []
            with torch.no_grad():
                for img, _ in va:
                    base_p.append(model(img.float()[None])[1][0, 0].numpy())
                    base_z.append(rec['z'][0, 0].numpy())
            base_m = metrics(base_p, gts, tert)
            rings = [CL.band_at_grid(CL.boundary_ring(m.float()[None], 3), (128, 128))[0, 0].numpy() > 0.5 for _, m in va]
            rows = {}
            for name, stages in (('dec3', ['dec3']), ('dec4', ['dec4']), ('dec5', ['dec5']), ('all', ['dec3', 'dec4', 'dec5'])):
                saved = {s: (model.fusion.heads[s].weight.data.clone(), model.fusion.heads[s].bias.data.clone()) for s in stages}
                for s in stages:
                    model.fusion.heads[s].weight.data.zero_()
                    model.fusion.heads[s].bias.data.zero_()
                kz, kp = [], []
                with torch.no_grad():
                    for img, _ in va:
                        kp.append(model(img.float()[None])[1][0, 0].numpy())
                        kz.append(rec['z'][0, 0].numpy())
                for s in stages:
                    model.fusion.heads[s].weight.data.copy_(saved[s][0])
                    model.fusion.heads[s].bias.data.copy_(saved[s][1])
                dz_ring = float(np.mean(np.concatenate([np.abs(b - k)[r] for b, k, r in zip(base_z, kz, rings)])))
                z_ring = float(np.mean(np.concatenate([np.abs(b)[r] for b, r in zip(base_z, rings)])))
                km = metrics(kp, gts, tert)
                rows[name] = {'mean_abs_dz_ring': dz_ring, 'mean_abs_z_ring': z_ring, **{f'delta_{k}': km[k] - base_m[k] for k in km}}
                print(f'S0-2 {a.dataset} CSF {how}, fusion off at {name}: |dz| in ring {dz_ring:.3f} (|z| {z_ring:.3f}); dDSC '
                      f'{rows[name]["delta_dsc"]:+.2f}, d small size {rows[name]["delta_size_small_pct"]:+.1f}, d large size '
                      f'{rows[name]["delta_size_large_pct"]:+.1f}', flush=True)
            res2[how] = rows
        out['S0-2'] = res2

    if '3' in parts:
        tr = [ds_tr[i] for i in range(n_tr)]
        gts_tr = [m[0].numpy() >= 0.5 for _, m in tr]
        res3 = {}
        for lab in ('A0', 'A1'):
            sel = cache[(lab, 'selected')][0] if (lab, 'selected') in cache else load_sd(C.resolve_checkpoint(runs[lab]))
            model = model_from(sel)
            p_tr, p_va = probs(model, tr), probs(model, va)

            def dsc_at(p, g, t):
                b = p >= t
                tp, fp, fn = (b & g).sum(), (b & ~g).sum(), (~b & g).sum()
                return 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 1.0

            def fit(pl, gl):
                area = np.array([(p >= 0.5).mean() for p in pl])
                cuts = np.quantile(area, [0.2, 0.4, 0.6, 0.8])
                b = np.digitize(area, cuts)
                table = {}
                for k in range(5):
                    idx = np.flatnonzero(b == k)
                    table[k] = max(THRS, key=lambda t: np.mean([dsc_at(pl[i], gl[i], t) for i in idx])) if len(idx) else 0.5
                return cuts, table

            def apply(pl, gl, cuts, table):
                area = np.array([(p >= 0.5).mean() for p in pl])
                b = np.digitize(area, cuts)
                return 100 * float(np.mean([dsc_at(p, g, table[k]) for p, g, k in zip(pl, gl, b)]))
            base = 100 * float(np.mean([dsc_at(p, g, 0.5) for p, g in zip(p_va, gts)]))
            cuts, table = fit(p_tr, gts_tr)
            a_tr = apply(p_va, gts, cuts, table)
            half = np.arange(len(p_va)) % 2
            cv = []
            for f in (0, 1):
                fi, ti = np.flatnonzero(half == f), np.flatnonzero(half != f)
                c2, t2 = fit([p_va[i] for i in fi], [gts[i] for i in fi])
                cv.append((apply([p_va[i] for i in ti], [gts[i] for i in ti], c2, t2), 100 * float(np.mean([dsc_at(p_va[i], gts[i], 0.5) for i in ti]))))
            cv_gain = float(np.mean([x - y for x, y in cv]))
            res3[lab] = {'val_dsc_at_0.5': base, 'fit_on_train': {'val_dsc': a_tr, 'gain': a_tr - base, 'thresholds_by_size_bin': table},
                         'val_2fold': {'gain': cv_gain}}
            print(f'S0-3 {a.dataset} {lab}: val per-image DSC at 0.5 {base:.2f}; size-binned threshold fit on train -> {a_tr:.2f} '
                  f'({a_tr - base:+.2f}), thresholds small->large {[table[k] for k in range(5)]}; 2-fold on val gain {cv_gain:+.2f}', flush=True)
        out['S0-3'] = res3

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, 'w') as f:
        json.dump(out, f, indent=1, default=float)
    print(f'wrote {a.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
