'''
Checks for the EXP-10 dec5 boundary residual refinement (models/refine.py) and its contour-zone loss
(contour_losses.boundary_zone). CPU, under a minute, no dataset needed:

    python tests/test_refine.py

1. parameter counts (B1 54,965 / B2 55,263 / B3 55,254) and no refine keys in the older configurations;
2. every refine model computes exactly the fusion-only model (= the baseline) at init, and the fusion-only
   and refine models leave the global RNG in the same state (same data order later on);
3. the boundary logit rides on gt_pre only in the gated mode; the two inference switches (enabled,
   gate_override) do what they say;
4. the contour zone equals scipy's Euclidean distance test on both sides of the contour, is empty for
   empty / full masks and along the image frame, matches the GPU formula, and is at least 5 cells wide
   on each side at the 128 grid;
5. the criterion: inner + w * BceDice(0.5,1) on the zone at the head's grid, one logged term named
   bnd_dec5, weight 0 = inner with no gradient, EXP-9 configurations unchanged;
6. gradient flow: step 1 moves only the fusion heads and refine.delta (+ bnd_head through the zone
   loss); after one AdamW step the shared weights of B3 (and of B2 with zone weight 0) still equal B1's,
   while B2's zone loss already reaches the backbone; after a few steps every refine tensor moved;
7. analysis.common recovers the configuration from a checkpoint and loads it strictly.
'''
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import contour_losses as CL                                           # noqa: E402
from models.egeunet import EGEUNet                                    # noqa: E402
from models.fusion import FUSION_STAGE_SETS, DeepSupervisionOutputs   # noqa: E402
from utils import GT_BceDiceLoss, BceDiceLoss, set_seed               # noqa: E402

FAILED = []
H = W = 256
CSF = dict(fusion_mode='sum', fusion_stages=FUSION_STAGE_SETS['shallow3'], fusion_dim=8)


def check(name, ok, detail=''):
    print(f'  [{"ok" if ok else "FAIL"}] {name}' + (f'  ({detail})' if detail else ''))
    if not ok:
        FAILED.append(name)


def build(**kw):
    set_seed(42)
    return EGEUNet(**kw)


def nparams(m):
    return sum(p.numel() for p in m.parameters())


def disc(r, cy=H / 2, cx=W / 2):
    yy, xx = torch.meshgrid(torch.arange(H, dtype=torch.float32), torch.arange(W, dtype=torch.float32),
                            indexing='ij')
    return (((yy - cy) ** 2 + (xx - cx) ** 2) <= r * r).float().view(1, 1, H, W)


def zone_ref(g, r):
    from scipy.ndimage import distance_transform_edt
    m = g[0, 0].numpy() > 0.5
    d = np.where(m, distance_transform_edt(m), distance_transform_edt(~m))
    return (d <= r).astype(np.float32)


def randomize(module, seed=0, scale=0.3):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in module.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * scale)


def main():
    torch.manual_seed(0)
    x = torch.randn(2, 3, H, W)
    target = torch.cat([disc(40), disc(15, 60, 190)])

    print('1. parameter counts')
    b0, b1 = build(), build(**CSF)
    b2, b3 = build(refine_mode='gate', **CSF), build(refine_mode='plain', **CSF)
    check('B0 53374 / B1 54965 / B2 55263 / B3 55254', (nparams(b0), nparams(b1), nparams(b2), nparams(b3)) ==
          (53374, 54965, 55263, 55254), f'{nparams(b0)} {nparams(b1)} {nparams(b2)} {nparams(b3)}')
    parts = {n: p.numel() for n, p in b2.refine.named_parameters()}
    check('B2 refine = bnd_head 9 + mix 200 + GN 16 + delta 73', sum(parts.values()) == 298 and parts['delta.weight'] == 72)
    for label, m in (('baseline', b0), ('sum-shallow3-d8', b1), ('bg_stage', build(fusion_mode='bg_stage', fusion_stages=FUSION_STAGE_SETS['shallow3'], fusion_dim=8))):
        check(f'{label}: no refine keys', not any(k.startswith('refine.') for k in m.state_dict()))
    for bad, kw in (('refine without fusion', dict(refine_mode='gate')),
                    ('refine with deep3', dict(refine_mode='gate', fusion_mode='sum', fusion_stages=FUSION_STAGE_SETS['deep3'])),
                    ('refine with bg_stage', dict(refine_mode='gate', fusion_mode='bg_stage', fusion_stages=FUSION_STAGE_SETS['shallow3'])),
                    ('refine without gt_ds', dict(refine_mode='plain', gt_ds=False, **CSF))):
        try:
            build(**kw)
            check(f'{bad} is refused', False)
        except ValueError:
            check(f'{bad} is refused', True)

    print('2. same function at init, same RNG state after construction')
    draws = {}
    for label, kw in (('B1', CSF), ('B2', dict(refine_mode='gate', **CSF)), ('B3', dict(refine_mode='plain', **CSF))):
        build(**kw)
        draws[label] = torch.rand(4)
    check('B1, B2, B3 leave the global RNG in the same state (fork_rng)',
          torch.equal(draws['B1'], draws['B2']) and torch.equal(draws['B1'], draws['B3']))
    for train_mode in (True, False):
        for m in (b0, b1, b2, b3):
            m.train(train_mode)
        with torch.no_grad():
            o = [m(x) for m in (b0, b1, b2, b3)]
        worst = max(float((o[k][1] - o[0][1]).abs().max()) for k in (1, 2, 3))
        worst = max([worst] + [float((a - b).abs().max()) for k in (1, 2, 3) for a, b in zip(o[k][0], o[0][0])])
        check(f'B1 / B2 / B3 output == baseline at init ({"train" if train_mode else "eval"})', worst == 0.0, f'{worst}')
    s1, s2, s3 = b1.state_dict(), b2.state_dict(), b3.state_dict()
    check('shared tensors of B2 and B3 identical to B1', all(torch.equal(s1[k], s2[k]) and torch.equal(s1[k], s3[k]) for k in s1))
    for m in (b0, b1, b2, b3):
        m.eval()

    print('3. outputs and inference switches')
    with torch.no_grad():
        g1, _ = b1(x)
        g2, _ = b2(x)
        g3, _ = b3(x)
    check('B1 and B3 return a plain tuple, B2 a DeepSupervisionOutputs with dec5 only',
          type(g1) is tuple and type(g3) is tuple and isinstance(g2, DeepSupervisionOutputs) and list(g2.boundary) == ['dec5']
          and tuple(g2.boundary['dec5'].shape) == (2, 1, 128, 128))
    randomize(b2.refine, 1)
    randomize(b3.refine, 1)
    b3.refine.mix.load_state_dict(b2.refine.mix.state_dict())
    b3.refine.delta.load_state_dict(b2.refine.delta.state_dict())
    shared = {k: v for k, v in b2.state_dict().items() if not k.startswith('refine.')}
    b1.load_state_dict(shared, strict=True)
    with torch.no_grad():
        _, o1 = b1(x)
        b2.refine.enabled = False
        _, o2off = b2(x)
        b2.refine.enabled = True
        _, o2 = b2(x)
        _, o3 = b3(x)
        b2.refine.gate_override = 'ones'
        _, o2ones = b2(x)
        zone = CL.band_at_grid(CL.boundary_zone(target, 10), (128, 128))
        b2.refine.gate_override = zone
        _, o2gt = b2(x)
        b2.refine.gate_override = None
        b3.refine.gate_override = zone
        _, o3z = b3(x)
        b3.refine.gate_override = None
        logit = b2.refine.bnd_head(torch.zeros(2, 8, 128, 128))
    check('enabled=False: B2 == B1 with the same shared weights (torch.equal)', torch.equal(o2off, o1))
    check('residual is not a no-op once trained', float((o2 - o2off).abs().max()) > 1e-3)
    check("gate_override='ones' on B2 == plain B3 with the same residual weights", torch.allclose(o2ones, o3, atol=1e-6))
    check('gate_override=zone on B2 differs from the learned gate', not torch.equal(o2gt, o2))
    with torch.no_grad():
        d5 = torch.randn(2, 8, 128, 128)
        p1 = torch.randn(2, 8, 128, 128)
        p2 = torch.randn(2, 8, 64, 64)
        gdz, dz, lb = b2.refine(d5, p1, p2)
        b3.refine.gate_override = zone
        gdz3, dz3, lb3 = b3.refine(d5, p1, p2)
        b3.refine.gate_override = None
    check('gate: gdz == dz * sigmoid(logit), logit = bnd_head(d5)',
          torch.allclose(gdz, dz * torch.sigmoid(lb), atol=1e-6) and torch.equal(lb, b2.refine.bnd_head(d5)))
    check('plain with a zone override: gdz == dz * zone, no logit', torch.equal(gdz3, dz3 * zone) and lb3 is None
          and torch.equal(dz3, dz))

    print('4. contour zone')
    from scipy.ndimage import distance_transform_edt
    worst = 0
    for g in (disc(40), disc(3, 20, 200), disc(60, 30, 30), disc(1, 128, 250), target[:1], target[1:]):
        for r in (3, 10):
            worst = max(worst, int((CL.boundary_zone(g, r)[0, 0].numpy() != zone_ref(g, r)).sum()))
    check('zone == scipy (d_in for lesion px, d_out for background px) <= r, r = 3 / 10', worst == 0, f'{worst} px differ')
    m = disc(40)[0, 0] > 0.5
    gpu_formula = torch.where(m, CL.edt_torch(~m), CL.edt_torch(m))
    cpu_formula = np.where(m.numpy(), distance_transform_edt(m.numpy()), distance_transform_edt(~m.numpy()))
    check('GPU formula (edt_torch) == CPU formula (scipy)', float(np.abs(gpu_formula.numpy() - cpu_formula).max()) < 1e-4)
    z, o = torch.zeros(1, 1, H, W), torch.ones(1, 1, H, W)
    check('empty / full mask -> empty zone', float(CL.boundary_zone(z, 10).sum()) == 0.0 and float(CL.boundary_zone(o, 10).sum()) == 0.0)
    half = torch.zeros(1, 1, H, W)
    half[..., :128, :] = 1
    rows = torch.nonzero(CL.boundary_zone(half, 10)[0, 0].sum(1)).flatten()
    check('half plane: zone = 10 rows inside + 10 rows outside, nothing along the frame',
          int(rows.min()) == 118 and int(rows.max()) == 137, f'rows {int(rows.min())}..{int(rows.max())}')
    z10 = CL.band_at_grid(CL.boundary_zone(disc(60), 10), (128, 128))[0, 0]
    line = z10[64].numpy()
    runs = np.diff(np.flatnonzero(np.diff(np.concatenate([[0], line, [0]]))))[::2] if line.any() else []
    check('zone at the 128 grid: two runs of >= 10 cells on the centre row (5 each side of the contour)',
          len(runs) == 2 and all(r >= 10 for r in runs), f'runs {list(runs)}')
    cov = float(CL.boundary_zone(target, 10).mean())
    check('zone coverage is a fraction of the image, not all of it', 0.02 < cov < 0.5, f'{100 * cov:.1f} %')

    print('5. criterion')
    check('defaults still build the bare GT_BceDiceLoss', type(CL.build_criterion()) is GT_BceDiceLoss)
    crit9 = CL.build_criterion(extra_term='bl', extra_weight='0.095', boundary_weight=1.0)
    check('EXP-9 criterion unchanged (3 band terms, 2-px band)', crit9.names == ['bl', 'bnd_dec3', 'bnd_dec4', 'bnd_dec5']
          and crit9.target_fn is CL.boundary_band)
    check('EXP-9 describe() unchanged', CL.describe('bcedice', 'bl', '0.095', boundary_weight=1.0)
          == 'bcedice + 0.095 * bl + 1 * bnd (0.1*dec3 + 0.2*dec4 + 0.3*dec5, BceDice(0.5,1) on the contour band per head grid)')
    crit = CL.build_criterion(boundary_weight=0.3, boundary_stages={'dec5': 1.0}, boundary_radius=10)
    check('B2 criterion: one term bnd_dec5 with the r=10 zone', crit.names == ['bnd_dec5'] and crit.stage_weights == {'dec5': 1.0}
          and crit.target_fn.keywords == {'radius': 10.0})
    d10 = CL.describe(boundary_weight=0.3, boundary_stages={'dec5': 1.0}, boundary_radius=10)
    check('describe() is ASCII and names the zone', d10 == 'bcedice + 0.3 * zone (BceDice(0.5,1) on the r=10px contour zone, 1*dec5)'
          and d10.isascii(), d10)
    logit = torch.randn(2, 1, 128, 128)
    gt_pre = DeepSupervisionOutputs(tuple(torch.sigmoid(torch.randn(2, 1, H, W)) for _ in range(5)), boundary={'dec5': logit})
    out = torch.sigmoid(torch.randn(2, 1, H, W))
    hand = BceDiceLoss(0.5, 1)(torch.sigmoid(logit), CL.band_at_grid(CL.boundary_zone(target, 10), (128, 128)))
    total = crit(gt_pre, out, target)
    expect = GT_BceDiceLoss(1, 1)(gt_pre, out, target) + 0.3 * hand
    check('loss = inner + 0.3 * BceDice(sigmoid(logit), zone@128)', abs(float(total - expect)) < 1e-6, f'{float(total):.6f} vs {float(expect):.6f}')
    check('running mean recorded under bnd_dec5', abs(crit.epoch_means()['bnd_dec5'] - float(hand)) < 1e-6)
    crit0 = CL.build_criterion(boundary_weight=0.0, boundary_stages={'dec5': 1.0}, boundary_radius=10)
    lg = logit.clone().requires_grad_(True)
    out_g = out.clone().requires_grad_(True)
    v0 = crit0(DeepSupervisionOutputs(tuple(gt_pre), boundary={'dec5': lg}), out_g, target)
    v0.backward()
    check('weight 0: value == inner, no gradient to the logit',
          torch.equal(v0.detach(), GT_BceDiceLoss(1, 1)(gt_pre, out, target)) and lg.grad is None and out_g.grad is not None)
    try:
        crit(tuple(gt_pre), out, target)
        check('plain tuple refused', False)
    except ValueError:
        check('plain tuple refused', True)
    for bad in (dict(boundary_weight=0.3, boundary_radius=0), dict(boundary_weight=-1.0)):
        try:
            CL.build_criterion(**bad)
            check(f'{bad} refused', False)
        except ValueError:
            check(f'{bad} refused', True)

    print('6. gradient flow and one-step equality with B1')
    for label, mode, weight in (('B2', 'gate', 0.3), ('B3', 'plain', None)):
        m = build(refine_mode=mode, **CSF)
        m.train()
        crit_m = CL.build_criterion(boundary_weight=weight, boundary_stages={'dec5': 1.0}, boundary_radius=10) if weight is not None \
            else CL.build_criterion()
        gp, o = m(x)
        crit_m(gp, o, target).backward()
        grads = {n: (p.grad is not None and float(p.grad.abs().sum()) > 0) for n, p in m.named_parameters()}
        check(f'{label} step 1: refine.delta gets gradient, refine.mix does not (delta starts at 0)',
              grads['refine.delta.weight'] and grads['refine.delta.bias'] and not any(v for n, v in grads.items() if n.startswith('refine.mix.')))
        if mode == 'gate':
            check('B2 step 1: bnd_head gets gradient from the zone loss', grads['refine.bnd_head.weight'])
        check(f'{label} step 1: fusion heads get gradient, proj / sum_w do not', all(v for n, v in grads.items() if n.startswith('fusion.heads.'))
              and not any(v for n, v in grads.items() if n.startswith(('fusion.proj.', 'fusion.sum_w.'))))
    # one AdamW step with the same batch. The residual starts at 0, so through the residual path the
    # shared weights get exactly 0 gradient at step 1: B3 (no zone loss) and B2 with zone weight 0
    # must still equal B1. With zone weight 0.3 the zone loss reaches D5 through the boundary head,
    # i.e. it supervises the backbone from step 1 on (intended; the B2 - B3 contrast includes it).
    def one_step(mode, weight):
        mdl = build(**CSF) if mode is None else build(refine_mode=mode, **CSF)
        mdl.train()
        cr = CL.build_criterion() if weight is None else \
            CL.build_criterion(boundary_weight=weight, boundary_stages={'dec5': 1.0}, boundary_radius=10)
        opt = torch.optim.AdamW(mdl.parameters(), lr=1e-3, weight_decay=1e-2)
        opt.zero_grad()
        gp, o = mdl(x)
        cr(gp, o, target).backward()
        opt.step()
        return mdl, cr, opt
    ref, _, _ = one_step(None, None)
    s1 = ref.state_dict()
    same = lambda m: all(torch.equal(s1[k], m.state_dict()[k]) for k in s1)
    m3, _, _ = one_step('plain', None)
    m2w0, _, _ = one_step('gate', 0.0)
    m2, c2, o2 = one_step('gate', 0.3)
    check('after 1 AdamW step: B3 shared weights == B1 (residual gradient is exactly 0 at step 1)', same(m3))
    check('after 1 AdamW step: B2 with zone weight 0 == B1', same(m2w0))
    check('after 1 AdamW step: B2 with zone weight 0.3 != B1 (the zone loss supervises D5 through the head)', not same(m2))
    before = {n: p.detach().clone() for n, p in m2.refine.named_parameters()}
    for _ in range(3):
        o2.zero_grad()
        gp, o = m2(x)
        c2(gp, o, target).backward()
        o2.step()
    check('every refine tensor moved after 3 more steps', all(not torch.equal(before[n], p.detach()) for n, p in m2.refine.named_parameters()))

    print('7. checkpoint round trip (analysis.common)')
    try:
        from analysis import common as AC
        for label, model, mode in (('B1', build(**CSF), 'none'), ('B2', build(refine_mode='gate', **CSF), 'gate'),
                                   ('B3', build(refine_mode='plain', **CSF), 'plain')):
            if mode != 'none':
                randomize(model.refine, 2)
            sd = model.state_dict()
            cfg = AC.infer_model_config(sd)
            rebuilt, _ = AC.build_model(sd, strict=True)
            with torch.no_grad():
                same = torch.equal(rebuilt(x)[1], model.eval()(x)[1])
            check(f'{label}: refine_mode {mode!r} recovered, strict load, same output',
                  cfg['refine_mode'] == mode and cfg['fusion_mode'] == 'sum' and cfg['fusion_dim'] == 8 and same)
    except ImportError as e:
        print(f'  [skip] analysis.common needs {e.name}')

    print()
    if FAILED:
        print(f'{len(FAILED)} check(s) FAILED: {FAILED}')
        return 1
    print('all checks passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
