'''
EXP-7: what does a boundary-type loss do to small-lesion spill and to large lesions? (diagnostics
D1-D11 of exp_docs/07, as one reusable tool). CPU is fine; nothing here touches training.

  sets   Build two FIXED pixel sets from a reference pair of EXP-6 runs (baseline, E3a):
           spill  = small-tertile pixels outside the lesion that the baseline drew and E3a removed
           chunk  = large-tertile lesion pixels that the baseline drew and E3a dropped
         Saved per dataset as a compressed npz. Later runs are read on these same pixels, so the
         comparison is not biased by selecting pixels on the runs being compared.

  probe  For each run, against a reference run (the baseline of the same batch):
           pooled DSC, per-image DSC per size tertile;
           small tertile: net spill pixels vs the reference;
           large tertile, split by whether the lesion touches the image border: net missed pixels
             vs the reference, area ratio (predicted / true), share of the newly missed pixels
             lying > 5 px from the run's own mask (dropped chunks rather than a thinner rim);
           mean probability on the fixed spill / chunk sets (low on spill and high on chunk is the
             goal);
           area ratio of the last deep-supervision head on non-touching large lesions;
           colour of the dropped lesion tissue on a core (0) -> healthy skin (1) axis.

  python analysis/e3a_mechanism.py sets --dataset isic17 --data-path data/data_isic1718/isic2017 \
      --base results/EGE-UNet-results-exp6/egeunet_isic17_learnable_s42 \
      --e3a results/EGE-UNet-results-exp6/egeunet_isic17_learnable_loss-bl_s42 --out <sets.npz>
  python analysis/e3a_mechanism.py probe --dataset isic17 --data-path data/data_isic1718/isic2017 \
      --sets <sets.npz> --ref base=<run> --run e3a=<run> --run x1=<run> ... --out <table.md>
'''
import os
import sys
import csv
import argparse

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FAR_PX = 5


def forward_all(run, dataset, data_path):
    '''Final probability map and last deep-supervision head for every test image of `dataset`.'''
    from analysis import common as C
    from datasets.dataset import NPY_datasets
    config = C.make_config(dataset, 256)
    ds = NPY_datasets(data_path, config, train=False)
    sd, _ = C.load_state_dict(C.resolve_checkpoint(run))
    model, _ = C.build_model(sd, 'cpu', None, strict=True)
    model.eval()
    probs, heads, gts, imgs = [], [], [], []
    with torch.no_grad():
        for i in range(len(ds)):
            img, msk = ds[i]
            gt_pre, out = model(img.float()[None])
            probs.append(out[0, 0].numpy().astype(np.float32))
            heads.append(gt_pre[4][0, 0].numpy() >= 0.5)
            gts.append(msk.float()[0].numpy() >= 0.5)
            imgs.append(img.float().numpy())
    return np.array(probs), np.array(heads), np.array(gts), np.array(imgs)


def tertiles(gts):
    area = gts.reshape(len(gts), -1).sum(1)
    return np.digitize(area, np.quantile(area, [1 / 3, 2 / 3]))


def touches_border(g):
    return bool(g[0, :].any() or g[-1, :].any() or g[:, 0].any() or g[:, -1].any())


def colour_axis(x, region, g):
    '''Position of `region` on the axis lesion core (0) -> far healthy skin (1); None if undefined.'''
    from scipy.ndimage import distance_transform_edt as edt
    d_in, d_out = edt(g), edt(~g)
    core, skin = g & (d_in >= max(3, 0.5 * d_in.max())), d_out > 40
    if core.sum() < 20 or skin.sum() < 200 or region.sum() < 30:
        return None
    c, s, r = x[:, core].mean(1), x[:, skin].mean(1), x[:, region].mean(1)
    return float(np.dot(r - c, s - c) / max(np.dot(s - c, s - c), 1e-6))


def cmd_sets(a):
    ub, _, g, _ = forward_all(a.base, a.dataset, a.data_path)
    ue, _, _, _ = forward_all(a.e3a, a.dataset, a.data_path)
    t = tertiles(g)
    pb, pe = ub >= 0.5, ue >= 0.5
    spill = pb & ~pe & ~g & (t == 0)[:, None, None]
    chunk = pb & ~pe & g & (t == 2)[:, None, None]
    np.savez_compressed(a.out, spill=np.packbits(spill), chunk=np.packbits(chunk), shape=np.array(g.shape))
    print(f'{a.dataset}: spill set {spill.sum()} px in {int(spill.any((1, 2)).sum())} small-lesion images, '
          f'chunk set {chunk.sum()} px in {int(chunk.any((1, 2)).sum())} large-lesion images -> {a.out}')


def load_sets(path):
    z = np.load(path)
    shape = tuple(z['shape'])
    n = int(np.prod(shape))
    return (np.unpackbits(z['spill'])[:n].reshape(shape).astype(bool),
            np.unpackbits(z['chunk'])[:n].reshape(shape).astype(bool))


def cmd_probe(a):
    from scipy.ndimage import distance_transform_edt as edt
    spill, chunk = load_sets(a.sets) if a.sets else (None, None)
    ref_label, ref_run = a.ref.split('=', 1)
    runs = [(ref_label, ref_run)] + [tuple(r.split('=', 1)) for r in a.run]
    fwd = {lab: forward_all(run, a.dataset, a.data_path) for lab, run in runs}
    _, _, g, x = fwd[ref_label]
    t = tertiles(g)
    touch = np.array([touches_border(gi) for gi in g])
    small = np.where(t == 0)[0]
    large_nt = np.where((t == 2) & ~touch)[0]
    large_t = np.where((t == 2) & touch)[0]
    ref_p = fwd[ref_label][0] >= 0.5

    def lr(p, idx):
        return float(np.mean([np.log((p[i].sum() + 1) / (g[i].sum() + 1)) for i in idx])) if len(idx) else float('nan')

    rows = []
    for lab, _ in runs:
        u, head, _, _ = fwd[lab]
        p = u >= 0.5
        tp, fp, fn = (p & g).reshape(len(g), -1).sum(1), (p & ~g).reshape(len(g), -1).sum(1), (~p & g).reshape(len(g), -1).sum(1)
        dsc = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
        row = {'run': lab, 'pooled_dsc': 100 * 2 * tp.sum() / (2 * tp.sum() + fp.sum() + fn.sum())}
        for k, n in enumerate(['small', 'mid', 'large']):
            row[f'dsc_{n}'] = 100 * dsc[t == k].mean()
        row['net_spill_small_kpx'] = ((p & ~g)[small].sum() - (ref_p & ~g)[small].sum()) / 1e3
        for tag, idx in (('large_nontouch', large_nt), ('large_touch', large_t)):
            row[f'net_missed_{tag}_kpx'] = ((~p & g)[idx].sum() - (~ref_p & g)[idx].sum()) / 1e3
            row[f'area_ratio_{tag}_pct'] = 100 * (np.exp(lr(p, idx)) - 1)
        newly, far, cols = 0, 0, []
        for i in large_nt:
            xm = g[i] & ~p[i] & ref_p[i]
            if xm.any():
                newly += xm.sum()
                if p[i].any():
                    far += (edt(~p[i])[xm] > FAR_PX).sum()
                c = colour_axis(x[i], xm, g[i])
                if c is not None:
                    cols.append(c)
        row['newly_missed_large_nontouch_kpx'] = newly / 1e3
        row['newly_missed_far_share_pct'] = 100 * far / max(newly, 1)
        row['dropped_tissue_colour'] = float(np.median(cols)) if cols else float('nan')
        row['ds_head_area_ratio_large_nontouch_pct'] = 100 * (np.exp(lr(head, large_nt)) - 1)
        if spill is not None:
            row['mean_u_on_spill_set'] = float(u[spill].mean())
            row['mean_u_on_chunk_set'] = float(u[chunk].mean())
        rows.append(row)

    keys = list(rows[0].keys())
    lines = [f'# e3a_mechanism probe: {a.dataset} (reference {ref_label}; large lesions: {len(large_nt)} not touching '
             f'the border, {len(large_t)} touching)', '', '| ' + ' | '.join(keys) + ' |', '|' + '---|' * len(keys)]
    for r in rows:
        lines.append('| ' + ' | '.join(r['run'] if k == 'run' else f'{r[k]:.3f}' if isinstance(r[k], float) else str(r[k])
                                       for k in keys) + ' |')
    text = '\n'.join(lines)
    print(text)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, 'w') as f:
            f.write(text + '\n')
        with open(os.path.splitext(a.out)[0] + '.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    ps = sub.add_parser('sets')
    pp = sub.add_parser('probe')
    for q in (ps, pp):
        q.add_argument('--dataset', required=True, choices=['isic17', 'isic18'])
        q.add_argument('--data-path', required=True)
    ps.add_argument('--base', required=True)
    ps.add_argument('--e3a', required=True)
    ps.add_argument('--out', required=True)
    pp.add_argument('--ref', required=True, help='label=run_dir (the baseline of the same batch)')
    pp.add_argument('--run', action='append', default=[], help='label=run_dir')
    pp.add_argument('--sets', default=None, help='npz written by `sets`')
    pp.add_argument('--out', default=None, help='markdown path; a .csv with the same stem is written too')
    a = p.parse_args()
    return cmd_sets(a) if a.cmd == 'sets' else cmd_probe(a)


if __name__ == '__main__':
    sys.exit(main())
