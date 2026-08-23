'''
Visualize and quantify the static GHPA gates of a trained EGE-UNet checkpoint.

In GHPA the feature groups are multiplied by gates conv_xy(BI(P_xy)), conv_zx(BI(P_zx)),
conv_zy(BI(P_zy)) that do NOT depend on the input image. This script captures those gates for all
six GHPA modules (enc4/5/6, dec1/2/3), plots them, and measures how spatially structured and how
center-biased they are, compared with (a) an untrained model (P = ones, where the only structure is
the 1-px border band of the zero-padded depthwise conv) and (b) the dataset lesion prior (mean train
mask at the module's resolution).

Usage:
  python analysis/visualize_hpa.py --checkpoint <work_dir or .pth> --data-path data/data_isic1718/isic2017
Outputs (default <work_dir>/analysis/hpa/): hpa_stats.json, hpa_stats_per_module.csv,
  hpa_stats_per_channel.csv, gates_<S>.npz, lesion_prior_<S>.npy and fig_*.png
'''
import os
import sys
import csv
import json
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis import common as C
from models.egeunet import EGEUNet


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', required=True, help='.pth file or experiment work_dir')
    p.add_argument('--out-dir', default=None, help='default: <work_dir>/analysis/hpa')
    p.add_argument('--input-size', type=int, default=256)
    p.add_argument('--data-path', default=None, help='dataset root (train/masks used for the lesion prior)')
    p.add_argument('--dataset', default='isic17', choices=['isic17', 'isic18'])
    p.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    p.add_argument('--c-list', type=int, nargs=6, default=None, help='override the channel list inferred from the checkpoint')
    p.add_argument('--hpa-mode', default='auto', choices=['auto', 'learnable', 'none'])
    p.add_argument('--untrained-seed', type=int, default=0)
    p.add_argument('--no-untrained', action='store_true', help='skip the untrained reference model')
    p.add_argument('--non-strict', action='store_true')
    p.add_argument('--dpi', type=int, default=150)
    return p.parse_args()


# ----------------------------------------------------------------------------- figures

def _grid_shape(n, max_cols=8):
    cols = min(max_cols, n)
    rows = int(np.ceil(n / cols))
    return rows, cols


def fig_channels(g, title, path, dpi, cmap=C.DIVERGING, center=0.0):
    n = g.shape[0]
    rows, cols = _grid_shape(n)
    d = float(np.abs(g - center).max())
    d = d if d > 1e-9 else 1e-9
    fig, axes = plt.subplots(rows, cols, figsize=(1.6 * cols + 1.2, 1.6 * rows + 0.8), squeeze=False)
    im = None
    for i, ax in enumerate(axes.ravel()):
        ax.set_xticks([]); ax.set_yticks([])
        if i < n:
            im = ax.imshow(g[i], cmap=cmap, vmin=center - d, vmax=center + d, interpolation='nearest')
            ax.set_title(f'ch {i}', fontsize=8)
        else:
            ax.axis('off')
    fig.suptitle(title, fontsize=9)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, pad=0.02)
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def fig_mean_maps(rec, st, prior_hw, title, path, dpi):
    g = rec['g_xy']
    mean_g, mean_abs = g.mean(axis=0), np.abs(g).mean(axis=0)
    ref = prior_hw if prior_hw is not None else C.radial_map(*mean_g.shape)
    ref_name = 'lesion prior (mean train mask)' if prior_hw is not None else 'radial distance (no --data-path)'
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    d = float(np.abs(mean_g).max()) or 1e-9
    im0 = axes[0].imshow(mean_g, cmap=C.DIVERGING, vmin=-d, vmax=d, interpolation='nearest')
    axes[0].set_title('channel-mean g_xy (signed)', fontsize=9)
    im1 = axes[1].imshow(mean_abs, cmap=C.SEQ_GATE, interpolation='nearest')
    mm = st['meanmap']
    axes[1].set_title(f'channel-mean |g_xy|\ncv_int={C.fmt(mm["cv_interior"])}  r_prior_int={C.fmt(mm["r_prior_interior"])}  '
                      f'r_radial_int={C.fmt(mm["r_radial_interior"])}', fontsize=8)
    im2 = axes[2].imshow(ref, cmap=C.SEQ_PRIOR, interpolation='nearest')
    axes[2].set_title(ref_name, fontsize=9)
    for ax, im in zip(axes, (im0, im1, im2)):
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(title, fontsize=9)
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def fig_profiles(rec, st_zx, st_zy, prior_hw, title, path, dpi):
    gzx, gzy = rec['g_zx'], rec['g_zy']
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5))
    for ax, g, name in ((axes[0, 0], gzx, 'g_zx (channel x row)'), (axes[0, 1], gzy, 'g_zy (channel x column)')):
        d = float(np.abs(g).max()) or 1e-9
        im = ax.imshow(g, cmap=C.DIVERGING, vmin=-d, vmax=d, aspect='auto', interpolation='nearest')
        ax.set_title(name, fontsize=9)
        ax.set_xlabel('position'); ax.set_ylabel('channel')
        fig.colorbar(im, ax=ax, shrink=0.8)
    for ax, g, st, axis_name, marg in ((axes[1, 0], gzx, st_zx, 'row', None if prior_hw is None else prior_hw.mean(axis=1)),
                                       (axes[1, 1], gzy, st_zy, 'column', None if prior_hw is None else prior_hw.mean(axis=0))):
        a = np.abs(g).mean(axis=0)
        pos = np.arange(a.shape[0])
        ax.plot(pos, a / (a.max() + C.EPS), color=C.CATEGORICAL[0], lw=2, label='channel-mean |g| (normalized)')
        if marg is not None:
            ax.plot(pos, marg / (marg.max() + C.EPS), color=C.CATEGORICAL[1], lw=2, ls='--', label='lesion prior marginal (normalized)')
        mm = st['meanmap']
        ax.set_title(f'{axis_name} profile   cv_int={C.fmt(mm["cv_interior"])}  r_prior_int={C.fmt(mm["r_prior_interior"])}', fontsize=8)
        ax.set_xlabel(axis_name); ax.set_ylabel('normalized value')
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, frameon=False)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def fig_summary(gates_tr, gates_un, stats, priors, ckpt_name, path, dpi, interior_only=False):
    mods = list(gates_tr.keys())
    rows = [('trained |g_xy| (channel mean)', gates_tr, C.SEQ_GATE),
            ('untrained |g_xy| (P = ones)', gates_un, C.SEQ_GATE),
            ('lesion prior' if priors else 'radial distance', None, C.SEQ_PRIOR)]
    crop = C.interior if interior_only else (lambda a: a)
    fig, axes = plt.subplots(3, len(mods), figsize=(2.2 * len(mods) + 1, 6.8), squeeze=False)
    for j, m in enumerate(mods):
        H, W = gates_tr[m]['H'], gates_tr[m]['W']
        for i, (label, src, cmap) in enumerate(rows):
            ax = axes[i, j]
            ax.set_xticks([]); ax.set_yticks([])
            if src is None:
                img = priors[m] if priors else C.radial_map(H, W)
            elif src and m in src:
                img = np.abs(src[m]['g_xy']).mean(axis=0)
            else:
                ax.axis('off'); continue
            ax.imshow(crop(img), cmap=cmap, interpolation='nearest')
            if i == 0:
                r = stats[m]['trained']['xy']['meanmap']['r_prior_interior']
                ax.set_title(f'{m}  {H}x{W}\nr_prior_int={C.fmt(r)}', fontsize=8)
            if j == 0:
                ax.set_ylabel(label, fontsize=8)
    scope = 'interior only (1-px border removed)' if interior_only else 'full maps'
    fig.suptitle(f'GHPA gates vs lesion prior, {scope}  [{ckpt_name}]', fontsize=9)
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def fig_summary_stats(stats, have_prior, path, dpi):
    mods = list(stats.keys())
    keys = [('cv_interior_mean', 'spatial non-uniformity cv (interior), mean over channels')]
    if have_prior:
        keys.append(('r_prior_interior', 'corr(|g_xy| channel-mean, lesion prior) interior'))
    keys.append(('r_radial_interior', 'corr(|g_xy| channel-mean, radial distance) interior  (neg = center emphasis)'))
    fig, axes = plt.subplots(1, len(keys), figsize=(4.2 * len(keys) + 1, 3.4), squeeze=False)
    x = np.arange(len(mods)); w = 0.36
    for ax, (k, label) in zip(axes[0], keys):
        for s, (which, col) in enumerate((('trained', C.CATEGORICAL[0]), ('untrained', C.CATEGORICAL[1]))):
            vals = []
            for m in mods:
                st = stats[m].get(which)
                if st is None:
                    vals.append(np.nan); continue
                vals.append(st['xy']['summary'][k] if k.endswith('_mean') else st['xy']['meanmap'][k])
            ax.bar(x + (s - 0.5) * w, vals, width=w * 0.92, color=col, label=which)
        ax.axhline(0, color='0.5', lw=0.8)
        ax.set_xticks(x); ax.set_xticklabels(mods, fontsize=8)
        ax.set_title(label, fontsize=8)
        ax.grid(axis='y', alpha=0.25)
        ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


# ----------------------------------------------------------------------------- main

def main():
    args = parse_args()
    device = C.pick_device(args.device)
    ckpt = C.resolve_checkpoint(args.checkpoint)
    work_dir = C.infer_work_dir(ckpt)
    out_dir = args.out_dir or os.path.join(work_dir, 'analysis', 'hpa')
    os.makedirs(out_dir, exist_ok=True)
    S = args.input_size
    ckpt_name = os.path.basename(ckpt)
    print(f'checkpoint: {ckpt}\nout_dir:    {out_dir}')

    sd, meta = C.load_state_dict(ckpt)
    overrides = {'c_list': args.c_list, 'hpa_mode': None if args.hpa_mode == 'auto' else args.hpa_mode}
    model, cfg = C.build_model(sd, device, overrides, strict=not args.non_strict)
    print(f'model config: {cfg}')

    gates_tr = C.capture_gates(model, S, device, cfg['input_channels'])
    if not gates_tr:
        info = {'checkpoint': ckpt, 'hpa_mode': 'none', 'note': 'model has no HPA gates; nothing to visualize'}
        json.dump(info, open(os.path.join(out_dir, 'hpa_stats.json'), 'w'), indent=2)
        print('model has hpa_mode=none (no HPA gates) -> wrote hpa_stats.json and exit')
        return 0

    gates_un = {}
    if not args.no_untrained:
        torch.manual_seed(args.untrained_seed)
        um = EGEUNet(**{**cfg, 'hpa_mode': 'learnable'}).eval().to(device)
        gates_un = C.capture_gates(um, S, device, cfg['input_channels'])

    prior, priors, n_masks = None, {}, 0
    if args.data_path:
        prior, n_masks = C.lesion_prior(args.data_path, S, os.path.join(out_dir, f'lesion_prior_{S}.npy'))
        priors = {m: C.downsample_prior(prior, (r['H'], r['W'])) for m, r in gates_tr.items()}
        print(f'lesion prior from {args.data_path} ({"cached" if n_masks < 0 else n_masks} masks)')
    else:
        print('no --data-path: prior-based statistics are skipped (nan)')

    stats = {}
    for m, rec in gates_tr.items():
        stats[m] = {'H': rec['H'], 'W': rec['W'], 'C': rec['C']}
        pr = priors.get(m)
        for which, src in (('trained', gates_tr), ('untrained', gates_un)):
            if m not in src:
                continue
            r = src[m]
            stats[m][which] = {
                'xy': C.gate_stats(r['g_xy'], r['p_xy'], pr),
                'zx': C.gate_stats(r['g_zx'], r['p_zx'], None if pr is None else pr.mean(axis=1)),
                'zy': C.gate_stats(r['g_zy'], r['p_zy'], None if pr is None else pr.mean(axis=0)),
            }

    # ---- tables
    info = {'checkpoint': ckpt, 'work_dir': work_dir, 'checkpoint_meta': meta, 'input_size': S,
            'model_config': cfg, 'data_path': args.data_path, 'n_train_masks': n_masks,
            'untrained_seed': None if args.no_untrained else args.untrained_seed,
            'torch': torch.__version__, 'modules': stats}
    json.dump(info, open(os.path.join(out_dir, 'hpa_stats.json'), 'w'), indent=1)

    mod_fields = ['module', 'which', 'gate', 'H', 'W', 'C'] + [k + '_meanmap' for k in C.STAT_KEYS] + \
                 [k for k in next(iter(next(iter(stats.values()))['trained'].values()))['summary'].keys()]
    with open(os.path.join(out_dir, 'hpa_stats_per_module.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=mod_fields, restval='')
        w.writeheader()
        for m, st in stats.items():
            for which in ('trained', 'untrained'):
                if which not in st:
                    continue
                for gate in ('xy', 'zx', 'zy'):
                    g = st[which][gate]
                    row = {'module': m, 'which': which, 'gate': gate, 'H': st['H'], 'W': st['W'], 'C': st['C']}
                    row.update({k + '_meanmap': g['meanmap'][k] for k in C.STAT_KEYS})
                    row.update(g['summary'])
                    w.writerow(row)
    ch_fields = ['module', 'which', 'gate', 'channel'] + C.STAT_KEYS + ['p_mean_abs_dev_from_one', 'p_cv']
    with open(os.path.join(out_dir, 'hpa_stats_per_channel.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=ch_fields, restval='')
        w.writeheader()
        for m, st in stats.items():
            for which in ('trained', 'untrained'):
                if which not in st:
                    continue
                for gate in ('xy', 'zx', 'zy'):
                    for d in st[which][gate]['per_channel']:
                        w.writerow({'module': m, 'which': which, 'gate': gate, **d})

    npz = {}
    for which, src in (('trained', gates_tr), ('untrained', gates_un)):
        for m, r in src.items():
            for k in ('g_xy', 'g_zx', 'g_zy', 'p_xy', 'p_zx', 'p_zy'):
                npz[f'{which}/{m}/{k}'] = r[k]
    for m, pr in priors.items():
        npz[f'prior/{m}'] = pr
    np.savez_compressed(os.path.join(out_dir, f'gates_{S}.npz'), **npz)

    # ---- figures
    for m, rec in gates_tr.items():
        st = stats[m]['trained']
        ttl = f'{m}  {rec["H"]}x{rec["W"]}  C={rec["C"]}  [{ckpt_name}]'
        fig_channels(rec['g_xy'], f'g_xy = conv_xy(BI(P_xy)) per channel   {ttl}',
                     os.path.join(out_dir, f'fig_{m}_gxy_channels.png'), args.dpi)
        fig_mean_maps(rec, st['xy'], priors.get(m), ttl, os.path.join(out_dir, f'fig_{m}_gxy_mean.png'), args.dpi)
        fig_channels(rec['p_xy'], f'raw P_xy (8x8, init = 1) per channel   {ttl}',
                     os.path.join(out_dir, f'fig_{m}_pxy_raw.png'), args.dpi, center=1.0)
        fig_profiles(rec, st['zx'], st['zy'], priors.get(m), ttl, os.path.join(out_dir, f'fig_{m}_gzx_gzy.png'), args.dpi)
    fig_summary(gates_tr, gates_un, stats, priors, ckpt_name, os.path.join(out_dir, 'fig_summary_gxy_vs_prior.png'), args.dpi)
    fig_summary(gates_tr, gates_un, stats, priors, ckpt_name, os.path.join(out_dir, 'fig_summary_gxy_vs_prior_interior.png'), args.dpi,
                interior_only=True)
    fig_summary_stats(stats, bool(priors), os.path.join(out_dir, 'fig_summary_stats.png'), args.dpi)

    # ---- stdout table
    hdr = f'{"module":7s} {"C":>3s} {"HxW":>7s} | {"cv_int tr":>9s} {"cv_int un":>9s} | {"r_prior tr":>10s} {"r_radial tr":>11s} {"ctr_ratio tr":>12s} | {"P dev":>7s}'
    print('\nchannel-mean |g_xy| statistics (interior = 1-px border removed)')
    print(hdr); print('-' * len(hdr))
    for m, st in stats.items():
        tr = st['trained']['xy']; un = st.get('untrained', {}).get('xy')
        print(f'{m:7s} {st["C"]:3d} {st["H"]:3d}x{st["W"]:<3d} | '
              f'{C.fmt(tr["summary"]["cv_interior_mean"]):>9s} {C.fmt(un["summary"]["cv_interior_mean"]) if un else "-":>9s} | '
              f'{C.fmt(tr["meanmap"]["r_prior_interior"]):>10s} {C.fmt(tr["meanmap"]["r_radial_interior"]):>11s} '
              f'{C.fmt(tr["meanmap"]["center_ratio_interior"]):>12s} | {C.fmt(tr["summary"]["p_mean_abs_dev_from_one_mean"]):>7s}')
    print(f'\nwrote {out_dir}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
