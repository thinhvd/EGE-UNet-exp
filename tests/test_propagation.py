'''
Checks for the EXP-13 core-to-region propagation (models/propagation.py) and its losses
(contour_losses.edge_masks / affinity_terms / completion_term / WithPropagationAux). CPU, about a minute:

    python tests/test_propagation.py

1. parameter counts (D1 54,965 -> 55,634 for every mode), no propagation keys in older configurations, refusals;
2. same global RNG and trunk as D1; enabled = False gives D1 bit for bit; at init the grown part is <= 0.12 next to
   the seed and <= 0.015 two cells away;
3. the operator on toy graphs: w^k at Chebyshev distance k, a closed ring of cut edges (diagonals included) stops
   the flood, a ring with one gap floods without the cap and decays with it, edges are symmetric, OR fusion;
4. gradients: the affinity loss reaches the edge head and never the trunk; decoupled modes get no segmentation
   gradient into the edge head, the coupled mode does;
5. provenance buffers refuse mismatched loads; analysis.common recovers the configuration;
6. edge groups on a half plane and the hard-edge selection; image_relative on a three-level map;
7. criterion values, log columns, describe() strings; train.py / config keys; older checkpoints still load.
'''
import os
import sys
import json
import subprocess

import numpy as np
import torch
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import contour_losses as CL                                           # noqa: E402
from models.egeunet import EGEUNet                                    # noqa: E402
from models.fusion import FUSION_STAGE_SETS, DeepSupervisionOutputs   # noqa: E402
from models.propagation import (CorePropagation, propagate, full_weights, image_relative, shift,  # noqa: E402
                                OFFS_HALF)
from utils import GT_BceDiceLoss, set_seed                            # noqa: E402

FAILED = []
H = W = 256
CSF = dict(fusion_mode='sum', fusion_stages=FUSION_STAGE_SETS['shallow3'], fusion_dim=8)
PROP = dict(propagate_tau=0.3, propagate_kappa=10.0, propagate_beta=1.0)


def check(name, ok, detail=''):
    print(f'  [{"ok" if ok else "FAIL"}] {name}' + (f'  ({detail})' if detail else ''))
    if not ok:
        FAILED.append(name)


def build(**kw):
    set_seed(42)
    return EGEUNet(**kw)


def nparams(m):
    return sum(p.numel() for p in m.parameters())


def rect_target(b=2):
    t = torch.zeros(b, 1, H, W)
    t[0, :, 40:220, 50:200] = 1
    if b > 1:
        t[1, :, 100:160, 20:90] = 1
    return t


def main():
    torch.manual_seed(0)
    x = torch.rand(2, 3, H, W) * 255
    target = rect_target()

    print('1. parameter counts and refusals')
    d1 = build(**CSF)
    models = {m: build(propagate_mode=m, **PROP, **CSF) for m in ('pair', 'pair_coupled', 'pair_nocap')}
    check('D1 54965, every propagation mode 55634 (+669)', nparams(d1) == 54965 and all(nparams(m) == 55634 for m in models.values()),
          f'{nparams(d1)} {[nparams(m) for m in models.values()]}')
    for label, m in (('baseline', build()), ('sum-shallow3-d8', d1), ('complete full', build(complete_mode='full', **CSF)),
                     ('refine gate', build(refine_mode='gate', **CSF))):
        check(f'{label}: no propagation keys', not any(k.startswith('propagate.') for k in m.state_dict()))
    for bad, kw in (('without fusion', dict(propagate_mode='pair', **PROP)),
                    ('with deep3', dict(propagate_mode='pair', fusion_mode='sum', fusion_stages=FUSION_STAGE_SETS['deep3'], **PROP)),
                    ('with complete', dict(propagate_mode='pair', complete_mode='full', **PROP, **CSF)),
                    ('with refine', dict(propagate_mode='pair', refine_mode='plain', **PROP, **CSF)),
                    ('without gt_ds', dict(propagate_mode='pair', gt_ds=False, **PROP, **CSF)),
                    ('unknown mode', dict(propagate_mode='grow', **PROP, **CSF)),
                    ('kappa 0', dict(propagate_mode='pair', propagate_tau=0.3, propagate_kappa=0.0, propagate_beta=1.0, **CSF))):
        try:
            build(**kw)
            check(f'propagate {bad} is refused', False)
        except ValueError:
            check(f'propagate {bad} is refused', True)

    print('2. same trunk and RNG as D1, start close to D1')
    draws = {}
    for label, kw in (('D1', CSF), ('pair', dict(propagate_mode='pair', **PROP, **CSF)),
                      ('coupled', dict(propagate_mode='pair_coupled', **PROP, **CSF)), ('nocap T8', dict(propagate_mode='pair_nocap',
                                                                                                         propagate_steps=8, **PROP, **CSF))):
        build(**kw)
        draws[label] = torch.rand(4)
    check('every mode leaves the global RNG where D1 leaves it', all(torch.equal(draws['D1'], v) for v in draws.values()))
    s1 = d1.state_dict()
    check('trunk tensors identical to D1', all(all(torch.equal(s1[k], m.state_dict()[k]) for k in s1) for m in models.values()))
    m = models['pair'].eval()
    d1.eval()
    with torch.no_grad():
        g1, o1 = d1(x)
        m.propagate.enabled = False
        g0, o0 = m(x)
        m.propagate.enabled = True
        g2, o2 = m(x)
    check('enabled = False: output == D1 (torch.equal) and a plain tuple', torch.equal(o0, o1) and type(g0) is tuple)
    check('propagation returns DeepSupervisionOutputs with aff / prop / seed / contrast',
          isinstance(g2, DeepSupervisionOutputs) and {'aff', 'prop', 'seed', 'contrast'} <= set(g2.boundary)
          and tuple(g2.boundary['aff'].shape) == (2, 4, 128, 128))
    seed, r = g2.boundary['seed'], g2.boundary['prop']
    grown = r - seed
    dist = torch.from_numpy(np.stack([__import__('scipy.ndimage', fromlist=['x']).distance_transform_cdt(
        1 - seed[b, 0].numpy().astype(np.int64), metric='chessboard') for b in range(2)]))[:, None].float()
    g_d1 = float(grown[dist == 1].max()) if (dist == 1).any() else 0.0
    g_d2 = float(grown[dist >= 2].max()) if (dist >= 2).any() else 0.0
    check('init: grown part <= 0.12 next to the seed, <= 0.015 from two cells on', g_d1 <= 0.12 and g_d2 <= 0.015,
          f'{g_d1:.4f} / {g_d2:.5f}, seed cells {int(seed.sum())}')
    check('init: |p - p_D1| <= 0.12 everywhere', float((o2 - o1).abs().max()) <= 0.12 + 1e-6, f'{float((o2 - o1).abs().max()):.4f}')

    print('3. operator on toy graphs')
    hh = ww = 40
    sd_ = torch.zeros(1, 1, hh, ww)
    sd_[..., 20, 20] = 1
    w8 = torch.full((1, 8, hh, ww), 0.5)
    rr = propagate(sd_, w8, 6)
    vals = [float(rr[0, 0, 20, 20 + k]) for k in range(8)]
    check('uniform w = 0.5: R = 0.5^k at Chebyshev distance k (<= T), 0 beyond', np.allclose(vals[:7], 0.5 ** np.arange(7)) and vals[7] == 0)
    lab = torch.zeros(1, 1, hh, ww)
    lab[..., 10:30, 10:30] = 1
    s_half = torch.cat([(lab == shift(lab, dy, dx, fill=-1)).float() for dy, dx in OFFS_HALF], 1)
    rr = propagate(sd_, full_weights(s_half), 60)
    check('closed ring of cut edges (diagonals included): no flood outside', float((rr * (1 - lab)).max()) == 0.0 and float(rr[..., 10:30, 10:30].min()) == 1.0)
    gap = s_half.clone()
    gap[0, 0, 20, 29] = 1.0           # the edge (20,29)-(20,30) across the right wall
    rr = propagate(sd_, full_weights(gap), 60)
    check('one gap, no cap: the outside floods', float(rr[0, 0, 2, 2]) == 1.0)
    u = torch.where(lab > 0.5, torch.ones_like(lab), torch.full_like(lab, 0.3))
    rr = propagate(sd_, full_weights(gap, u), 60)
    out_vals = [float(rr[0, 0, 20, 30 + k]) for k in range(3)]
    check('one gap, cap 0.3 outside: leak decays as 0.3^k', np.allclose(out_vals, [0.3, 0.09, 0.027], atol=1e-6), f'{out_vals}')
    s_rand = torch.rand(1, 4, hh, ww)
    wf = full_weights(s_rand)
    sym = all(abs(float(wf[0, i, y, xx]) - float(wf[0, 4 + i, y + dy, xx + dx])) < 1e-7
              for i, (dy, dx) in enumerate(OFFS_HALF) for y, xx in ((5, 5), (17, 22), (30, 8)))
    check('edge symmetry: weight into x from x+o == weight into x+o from x', sym)
    pb, gg = torch.tensor([0.1]), torch.tensor([0.5])
    check('OR fusion: 0.1 OR 0.5 = 0.55; G = 0 leaves p_base (1e-7)', abs(float(1 - (1 - pb) * (1 - gg)) - 0.55) < 1e-7
          and abs(float(1 - (1 - pb) * 1) - 0.1) < 1e-7)

    print('4. gradients')
    for mode in ('pair', 'pair_coupled', 'pair_nocap'):
        mdl = build(propagate_mode=mode, **PROP, **CSF)
        mdl.train()
        gp, o = mdl(x)
        GT_BceDiceLoss(1, 1)(gp, o, target).backward()     # segmentation loss only
        seg_to_edge = any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in mdl.propagate.parameters())
        check(f'{mode}: segmentation loss reaches the edge head' + (' (coupled)' if mode == 'pair_coupled' else ': no'),
              seg_to_edge == (mode == 'pair_coupled'))
        mdl.zero_grad()
        gp, o = mdl(x)
        crit = CL.build_criterion(aff_weight=1.0)
        aff, _ = CL.affinity_terms(gp.boundary['aff'], CL.edge_masks(target, (128, 128), gp.boundary['contrast']))
        aff.backward()
        trunk = any(p.grad is not None and float(p.grad.abs().sum()) > 0 for n, p in mdl.named_parameters() if not n.startswith('propagate.'))
        edge = all(p.grad is not None and float(p.grad.abs().sum()) > 0 for n, p in mdl.propagate.named_parameters()
                   if n.startswith(('edge.2', 'off_bias')))
        check(f'{mode}: affinity loss reaches edge.2 / off_bias and never the trunk', edge and not trunk)
    mdl = build(propagate_mode='pair', **PROP, **CSF)
    mdl.train()
    crit = CL.build_criterion(aff_weight=1.0)
    opt = torch.optim.AdamW(mdl.parameters(), lr=1e-3, weight_decay=1e-2)
    before = {n: p.detach().clone() for n, p in mdl.propagate.named_parameters()}
    for _ in range(3):
        opt.zero_grad()
        gp, o = mdl(x)
        crit(gp, o, target).backward()
        opt.step()
    check('every edge-head tensor moved after 3 AdamW steps', all(not torch.equal(before[n], p.detach()) for n, p in mdl.propagate.named_parameters()))

    print('5. provenance buffers and analysis.common')
    sd2 = models['pair'].state_dict()
    for label, kw in (('pair checkpoint into coupled', dict(propagate_mode='pair_coupled', **PROP)),
                      ('tau 0.3 into tau 0.4', dict(propagate_mode='pair', propagate_tau=0.4, propagate_kappa=10.0, propagate_beta=1.0)),
                      ('T=32 into T=16', dict(propagate_mode='pair', propagate_steps=16, **PROP))):
        try:
            build(**kw, **CSF).load_state_dict(sd2, strict=True)
            check(f'{label} is refused', False)
        except RuntimeError:
            check(f'{label} is refused', True)
    try:
        from analysis import common as AC
        for mode in ('pair', 'pair_coupled', 'pair_nocap'):
            mm = build(propagate_mode=mode, propagate_tau=0.2, propagate_kappa=5.0, propagate_beta=2.0, **CSF)
            g = torch.Generator().manual_seed(1)
            with torch.no_grad():
                for p in mm.propagate.parameters():
                    p.add_(torch.randn(p.shape, generator=g) * 0.3)
            sd = mm.state_dict()
            cfg = AC.infer_model_config(sd)
            rebuilt, _ = AC.build_model(sd, strict=True)
            with torch.no_grad():
                same = torch.equal(rebuilt(x)[1], mm.eval()(x)[1])
            check(f'{mode}: configuration recovered (tau 0.2, kappa 5, beta 2), strict load, same output',
                  cfg['propagate_mode'] == mode and cfg['propagate_tau'] == 0.2 and cfg['propagate_kappa'] == 5.0
                  and cfg['propagate_beta'] == 2.0 and same)
        check('older configurations infer no propagate keys', 'propagate_mode' not in AC.infer_model_config(s1))
    except ImportError as e:
        print(f'  [skip] analysis.common needs {e.name}')

    print('6. edge groups, hard edges, image-relative coordinates')
    half = torch.zeros(1, 1, H, W)
    half[..., :128, :] = 1
    mk = CL.edge_masks(half, (128, 128))
    rows = torch.nonzero(mk['X'][0].any(0).any(1)).flatten().tolist()
    check('half plane: cut edges only on cell row 63 (down and diagonal offsets), none for the right offset',
          rows == [63] and not mk['X'][0, 0].any() and mk['X'][0, 1].sum() == 128, f'rows {rows}')
    check('LL above, BB below (last column has no right neighbour), groups disjoint',
          bool(mk['LL'][0, 0, :63, :127].all()) and bool(mk['BB'][0, 0, 65:, :127].all()) and not mk['LL'][0, 0, :, 127].any()
          and not (mk['X'] & mk['LL']).any() and not (mk['LL'] & mk['BB']).any())
    con = torch.rand(1, 4, 128, 128)
    mk = CL.edge_masks(half, (128, 128), con, 0.2)
    frac = float(mk['LL_hard'].sum()) / float(mk['LL'].sum())
    check('hard edges: ~20 % of LL edges, all inside LL, highest contrast', abs(frac - 0.2) < 0.01 and not (mk['LL_hard'] & ~mk['LL']).any()
          and float(con[mk['LL_hard']].min()) >= float(con[mk['LL'] & ~mk['LL_hard']].max()), f'{100 * frac:.1f} %')
    z = torch.full((1, 1, 16, 16), -5.0)
    z[..., :, 10:] = 5.0
    z[..., :, 5:10] = -5.0 + 10.0 / 3
    rgb = torch.zeros(1, 3, 16, 16)
    rgb[:, 0, :, 10:] = 1.0
    rgb[:, 0, :, 5:10] = 0.5
    tz, trgb, valid = image_relative(z, rgb, 0.9)
    check('image_relative: t_z = 0 / 1/3 / 1 on background / middle / core; t_rgb = 0 / 0.5 / 1',
          bool(valid[0]) and abs(float(tz[0, 0, 0, 0])) < 1e-6 and abs(float(tz[0, 0, 0, 7]) - 1 / 3) < 1e-5
          and abs(float(tz[0, 0, 0, 12]) - 1) < 1e-6 and abs(float(trgb[0, 0, 0, 7]) - 0.5) < 1e-6)
    _, _, v2 = image_relative(torch.full((1, 1, 8, 8), -5.0), torch.zeros(1, 3, 8, 8), 0.9)
    check('image_relative: no core -> invalid, t = 0', not bool(v2[0]))

    print('7. criterion, describe(), config, older checkpoints')
    check('defaults still build the bare GT_BceDiceLoss', type(CL.build_criterion()) is GT_BceDiceLoss)
    check('EXP-11 describe() unchanged', CL.describe(boundary_weight=0.3, boundary_stages={'ring': 1.0}, ring_radius=3)
          == 'bcedice + 0.3 * ring (BceDice(0.5,1) on the +-3px contour ring, barrier head)')
    d_e1 = CL.describe(aff_weight=1.0, hard_frac=0.2)
    d_e2 = CL.describe(aff_weight=1.0, comp_weight=1.0, hard_frac=0.2)
    check('E1 / E2 describe() ASCII', d_e1 == 'bcedice + 1 * aff (edge BCE same-label: 0.5 cut / 0.25 LL+BB / 0.25 hard 20% contrast)'
          and d_e2 == d_e1 + ' + 1 * comp (BCE of propagated R vs GT, GT dilated 10 cells)' and d_e2.isascii(), d_e2)
    c1, c2 = CL.build_criterion(aff_weight=1.0), CL.build_criterion(aff_weight=1.0, comp_weight=1.0)
    check('log names: aff_X / aff_LL / aff_BB / aff_hard / rank_acc (+ comp)',
          c1.names == ['aff_X', 'aff_LL', 'aff_BB', 'aff_hard', 'rank_acc'] and c2.names == c1.names + ['comp'])
    s_logit = torch.randn(2, 4, 128, 128)
    contrast = torch.rand(2, 4, 128, 128)
    r_ = torch.rand(2, 1, 128, 128)
    seed_ = (torch.rand(2, 1, 128, 128) > 0.9).float()
    gp = DeepSupervisionOutputs(tuple(torch.sigmoid(torch.randn(2, 1, H, W)) for _ in range(5)),
                                boundary={'aff': s_logit, 'prop': r_, 'seed': seed_, 'contrast': contrast})
    out = torch.sigmoid(torch.randn(2, 1, H, W))
    mk = CL.edge_masks(target, (128, 128), contrast, 0.2)
    sp = lambda t, m: float((t * m).sum() / m.sum())
    hand = 0.5 * sp(F.softplus(s_logit), mk['X']) + 0.125 * (sp(F.softplus(-s_logit), mk['LL']) + sp(F.softplus(-s_logit), mk['BB'])) \
        + 0.25 * sp(F.softplus(-s_logit), mk['LL_hard'])
    region = (F.max_pool2d(mk['lab'].float(), 21, 1, 10) > 0.5) & (seed_ < 0.5)
    comp = float((F.binary_cross_entropy(r_, mk['lab'].float(), reduction='none') * region).sum() / region.sum())
    base = float(GT_BceDiceLoss(1, 1)(gp, out, target))
    check('E1 loss = BceDice + affinity (hand formula)', abs(float(c1(gp, out, target)) - (base + hand)) < 1e-4)
    check('E2 loss = E1 + completion (hand formula)', abs(float(c2(gp, out, target)) - (base + hand + comp)) < 1e-4)
    try:
        CL.build_criterion(aff_weight=1.0, boundary_weight=0.3)
        check('affinity + boundary loss refused', False)
    except ValueError:
        check('affinity + boundary loss refused', True)

    py = sys.executable
    probe = ('import sys, json; sys.argv = ["train.py"] + json.loads(sys.argv[1]); import train; '
             'from configs.config_setting import get_config; c = get_config(train.parse_args()); '
             'print("CFG" + json.dumps({"model": {k: v for k, v in c.model_config.items() if k.startswith(("propagate", "complete"))}, '
             '"loss": sorted(getattr(c, "loss_config", {}).keys())}))')

    def cfg(flags):
        p = subprocess.run([py, '-c', probe, json.dumps(flags)], cwd=REPO, capture_output=True, text=True)
        line = [ln for ln in p.stdout.splitlines() if ln.startswith('CFG')]
        return p.returncode, (json.loads(line[0][3:]) if line else None), p.stderr
    base_f = ['--work-dir', '/tmp/x', '--device', 'cpu']
    csf = ['--fusion', 'sum', '--fusion-stages', 'shallow3', '--fusion-dim', '8']
    pf = ['--propagate-tau', '0.3', '--propagate-kappa', '10', '--propagate-beta', '1', '--aff-weight', '1']
    rc, c_old, _ = cfg(base_f + csf)
    check('D1 flags: no propagate keys, no loss_config', rc == 0 and c_old == {'model': {}, 'loss': []}, str(c_old))
    rc, c_e11, _ = cfg(base_f + csf + ['--complete', 'full', '--boundary-weight', '0.3'])
    check('EXP-11 D2 flags: no propagate / aff keys', rc == 0 and 'aff_weight' not in c_e11['loss']
          and not any(k.startswith('propagate') for k in c_e11['model']), str(c_e11))
    rc, c_e2, _ = cfg(base_f + csf + ['--propagate', 'pair_coupled'] + pf + ['--comp-weight', '1'])
    check('E2 flags: propagate_* keys, aff/comp/hard_frac loss keys', rc == 0 and c_e2['model'].get('propagate_mode') == 'pair_coupled'
          and {'aff_weight', 'comp_weight', 'hard_frac'} <= set(c_e2['loss']), str(c_e2))
    for label, flags in (('--propagate without tau/kappa/beta/aff', csf + ['--propagate', 'pair']),
                         ('--propagate with deep3', ['--fusion', 'sum', '--fusion-stages', 'deep3', '--propagate', 'pair'] + pf),
                         ('--propagate with --complete', csf + ['--propagate', 'pair', '--complete', 'full', '--boundary-weight', '0.3'] + pf),
                         ('--comp-weight with pair', csf + ['--propagate', 'pair', '--comp-weight', '1'] + pf),
                         ('--aff-weight without --propagate', csf + ['--aff-weight', '1'])):
        rc, _, err = cfg(base_f + flags)
        check(f'{label} refused by train.py', rc == 2, err.strip().splitlines()[-1] if err.strip() else '')
    from analysis import common as AC
    ok, seen = True, 0
    for e in ('exp10', 'exp11'):
        root = os.path.join(REPO, 'results', f'EGE-UNet-results-{e}')
        if not os.path.isdir(root):
            continue
        for dname in sorted(os.listdir(root)):
            if dname.startswith('egeunet_'):
                try:
                    AC.build_model(AC.load_state_dict(AC.resolve_checkpoint(os.path.join(root, dname)))[0], strict=True)
                    seen += 1
                except Exception as ex:  # noqa: BLE001
                    ok = False
                    print(f'    {dname}: {ex}')
    check(f'EXP-10 / EXP-11 checkpoints still load strictly ({seen})', ok)

    print()
    if FAILED:
        print(f'{len(FAILED)} check(s) FAILED: {FAILED}')
        return 1
    print('all checks passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
