'''
Checks for the EXP-9 boundary-guided cross-stage fusion (models/fusion.py 'bg_stage') and its
contour-band loss (contour_losses.WithBoundaryAux). CPU, well under a minute, no dataset needed:

    python tests/test_bg_fusion.py

1. parameter counts of every EXP-4/5/8 configuration are unchanged, the EXP-9 arms have the planned
   counts, and the old modes carry no boundary-guided keys;
2. every boundary-guided model computes exactly the baseline function at init (heads are zero);
3. the boundary logits ride on gt_pre only in that mode, survive copy / pickle, and do not change
   what GT_BceDiceLoss computes;
4. the fusion arithmetic: with one alpha shared by all sources it is exactly a (1 + alpha * B) gate on
   the fused output; with per-source alphas it matches the formula written out by hand;
5. the contour band: no band on empty / full masks or along the image frame, equal to scipy's
   dilation minus erosion, soft targets thresholded at 0.5, max-pooled band at least one cell wide;
6. WithBoundaryAux adds exactly weight * (0.1 BD3 + 0.2 BD4 + 0.3 BD5); weight 0 changes nothing and
   sends no gradient; the criteria and log strings of the older experiments are unchanged;
7. gradient flow at the first steps (only the fusion heads, plus the boundary heads through the band
   loss) and every fusion tensor moving after a few AdamW steps;
8. analysis.common recovers the configuration from a checkpoint and loads it strictly.
'''
import copy
import os
import pickle
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


def check(name, ok, detail=''):
    print(f'  [{"ok" if ok else "FAIL"}] {name}' + (f'  ({detail})' if detail else ''))
    if not ok:
        FAILED.append(name)


def build(**kw):
    set_seed(42)
    return EGEUNet(**kw)


def nparams(m):
    return sum(p.numel() for p in m.parameters())


def bg(stages='shallow3', dim=8, **kw):
    return build(fusion_mode='bg_stage', fusion_stages=FUSION_STAGE_SETS[stages], fusion_dim=dim, **kw)


def disc(r, cy=H / 2, cx=W / 2):
    yy, xx = torch.meshgrid(torch.arange(H, dtype=torch.float32), torch.arange(W, dtype=torch.float32),
                            indexing='ij')
    return (((yy - cy) ** 2 + (xx - cx) ** 2) <= r * r).float().view(1, 1, H, W)


def randomize_fusion(model, seed=0):
    '''Non-zero heads / weights so the fusion arithmetic is actually exercised.'''
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.fusion.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.3)


def main():
    torch.manual_seed(0)
    x = torch.randn(2, 3, H, W)

    print('1. parameter counts')
    expect_old = {('learnable', 'none', None): 53374,
                  ('learnable', 'sum', 'deep3'): 57445, ('learnable', 'sum', 'all5'): 57863,
                  ('learnable', 'concat', 'deep3'): 64086, ('learnable', 'concat', 'all5'): 66030,
                  ('learnable', 'csaa', 'deep3'): 65718, ('learnable', 'csaa', 'all5'): 67662,
                  ('learnable', 'sum_attn', 'deep3'): 59077, ('learnable', 'sum_attn', 'all5'): 59495,
                  ('none', 'sum', 'deep3'): 49059, ('none', 'sum_attn', 'all5'): 51109}
    for (hpa, mode, st), want in expect_old.items():
        m = build(hpa_mode=hpa, fusion_mode=mode, fusion_stages=FUSION_STAGE_SETS[st] if st else None)
        keys = m.state_dict().keys()
        clean = not any('bnd_heads' in k or 'bg_alpha' in k for k in keys)
        check(f'{hpa}+{mode}-{st}: {want} params, no boundary-guided keys', nparams(m) == want and clean,
              f'{nparams(m)}')
    a1 = build(fusion_mode='sum', fusion_stages=FUSION_STAGE_SETS['shallow3'], fusion_dim=8)
    check('A1e sum-shallow3-d8: 54965 params', nparams(a1) == 54965, f'{nparams(a1)}')
    a3 = bg()
    check('A3 bg_stage-shallow3-d8: 55031 params', nparams(a3) == 55031, f'{nparams(a3)}')
    parts = {'proj': nparams(a3.fusion.proj), 'heads': nparams(a3.fusion.heads),
             'w+alpha': sum(p.numel() for p in a3.fusion.sum_w.parameters()) +
             sum(p.numel() for p in a3.fusion.bg_alpha.parameters()),
             'bnd_heads': nparams(a3.fusion.bnd_heads)}
    check('A3 branch = proj 1144 + heads 432 + w/alpha 30 + boundary heads 51',
          parts == {'proj': 1144, 'heads': 432, 'w+alpha': 30, 'bnd_heads': 51}, str(parts))

    print('2. same function as the baseline at init')
    base = build()
    for label, model in (('bg_stage shallow3 d8', bg()), ('bg_stage all5 d16', bg('all5', 16))):
        for train_mode in (True, False):
            base.train(train_mode)
            model.train(train_mode)
            with torch.no_grad():
                gb, ob = base(x)
                gm, om = model(x)
            diff = max([float((om - ob).abs().max())] + [float((a - b).abs().max()) for a, b in zip(gm, gb)])
            check(f'{label} ({"train" if train_mode else "eval"}): max |out - baseline| = 0', diff == 0.0,
                  f'{diff}')
        sb, sm = base.state_dict(), model.state_dict()
        same = all(torch.equal(sb[k], sm[k]) for k in sb)
        check(f'{label}: every backbone tensor identical to the baseline', same)
    base.eval()

    print('3. boundary logits travel on gt_pre only in the boundary-guided mode')
    with torch.no_grad():
        gb, _ = base(x)
        ga1, _ = a1.eval()(x)
        g3, o3 = a3.eval()(x)
    check('baseline and sum-shallow3 return a plain tuple', type(gb) is tuple and type(ga1) is tuple)
    check('bg_stage returns DeepSupervisionOutputs', isinstance(g3, DeepSupervisionOutputs) and len(g3) == 5)
    shapes = {k: tuple(v.shape) for k, v in g3.boundary.items()}
    check('boundary logits at 32 / 64 / 128', shapes == {'dec3': (2, 1, 32, 32), 'dec4': (2, 1, 64, 64),
                                                         'dec5': (2, 1, 128, 128)}, str(shapes))
    t = (torch.rand(2, 1, H, W) > 0.7).float()
    check('GT_BceDiceLoss(subclass) == GT_BceDiceLoss(plain tuple)',
          torch.equal(GT_BceDiceLoss(1, 1)(g3, o3, t), GT_BceDiceLoss(1, 1)(tuple(g3), o3, t)))
    for how, clone in (('copy', copy.copy(g3)), ('pickle', pickle.loads(pickle.dumps(g3)))):
        ok = isinstance(clone, DeepSupervisionOutputs) and set(clone.boundary) == set(g3.boundary) and \
            all(torch.equal(a, b) for a, b in zip(clone, g3)) and \
            all(torch.equal(clone.boundary[k], g3.boundary[k]) for k in g3.boundary)
        check(f'{how} keeps the tuple and .boundary', ok)

    print('4. fusion arithmetic')
    m = bg()
    randomize_fusion(m, 1)
    f = m.fusion
    feats = [torch.randn(2, c, H // s, W // s) for c, s in zip([8, 16, 24, 32, 48], [2, 4, 8, 16, 32])]
    p = f.project(feats)
    worst_gate, worst_formula = 0.0, 0.0
    for s, c, g in (('dec3', 24, 32), ('dec4', 16, 64), ('dec5', 8, 128)):
        feat = torch.randn(2, c, g, g)
        w = f.sum_w[s]
        resized = [torch.nn.functional.interpolate(q, size=(g, g), mode='bilinear', align_corners=True)
                   if q.shape[2:] != (g, g) else q for q in p]
        with torch.no_grad():
            B = torch.sigmoid(f.bnd_heads[s](feat))
            a_shared = torch.randn(1).item()
            f.bg_alpha[s].fill_(a_shared)
            gated, _ = f.guided_stage(s, p, feat)
            plain = sum(w[i] * resized[i] for i in range(5))
            worst_gate = max(worst_gate, float((gated - f.heads[s]((1 + a_shared * B) * plain)).abs().max()))
            f.bg_alpha[s].copy_(torch.randn(5))
            fused, logit = f.guided_stage(s, p, feat)
            a = f.bg_alpha[s]
            by_hand = f.heads[s](sum(w[i] * (1 + a[i] * B) * resized[i] for i in range(5)))
            worst_formula = max(worst_formula, float((fused - by_hand).abs().max()))
            check(f'{s}: logit = boundary head on the given feature',
                  torch.equal(logit, f.bnd_heads[s](feat)))
    check('shared alpha == (1 + alpha * B) gate on the summed output', worst_gate < 1e-5, f'{worst_gate:.2e}')
    check('per-source alphas == sum_j w_j (1 + a_j B) P_j by hand', worst_formula < 1e-5, f'{worst_formula:.2e}')
    try:
        f(feats)
        check('forward() refuses the boundary-guided mode', False)
    except RuntimeError:
        check('forward() refuses the boundary-guided mode', True)
    try:
        bg(gt_ds=False)
        check('boundary-guided fusion without deep supervision is refused', False)
    except ValueError:
        check('boundary-guided fusion without deep supervision is refused', True)

    print('5. contour band')
    from scipy.ndimage import binary_dilation, binary_erosion
    z, o = torch.zeros(1, 1, H, W), torch.ones(1, 1, H, W)
    check('empty mask -> no band', float(CL.boundary_band(z).sum()) == 0.0)
    check('full mask -> no band (the image frame is not a contour)', float(CL.boundary_band(o).sum()) == 0.0)
    worst = 0
    for g in (disc(40), disc(3, 20, 200), disc(60, 30, 30), disc(1, 128, 250)):
        gb_ = g[0, 0].numpy() > 0.5
        ref = binary_dilation(gb_, np.ones((3, 3))) & ~binary_erosion(gb_, np.ones((3, 3)), border_value=1)
        worst = max(worst, int((CL.boundary_band(g)[0, 0].numpy().astype(bool) != ref).sum()))
    check('band = scipy dilation - erosion (border_value=1), incl. lesions touching the frame', worst == 0,
          f'{worst} differing px')
    half = torch.zeros(1, 1, H, W)
    half[..., :128, :] = 1
    rows = torch.nonzero(CL.boundary_band(half)[0, 0].sum(1)).flatten().tolist()
    check('half plane touching the frame: band only on its inner edge (rows 127, 128)', rows == [127, 128],
          str(rows))
    soft = disc(40) * 0.6 + (disc(50) - disc(40)) * 0.4
    check('soft target thresholded at 0.5', torch.equal(CL.boundary_band(soft), CL.boundary_band(disc(40))))
    band = CL.boundary_band(disc(60))
    ok = True
    for g in (32, 64, 128):
        bg_ = CL.band_at_grid(band, (g, g))
        s = H // g
        blocks = band.view(1, 1, g, s, g, s).amax(dim=(3, 5))
        ok &= torch.equal(bg_, blocks) and float(bg_[0, 0, g // 2, g // 2]) == 0.0 and float(bg_.sum()) > 0
        # a horizontal line through the centre crosses the ring on both sides
        line = bg_[0, 0, g // 2].numpy()
        ok &= int(((line[1:] > 0) & (line[:-1] == 0)).sum()) == 2
    check('band max-pooled to 32/64/128: closed ring, >= 1 cell, empty centre', bool(ok))
    check('band_at_grid at full size is the identity', CL.band_at_grid(band, (H, W)) is band)

    print('6. WithBoundaryAux and the criteria')
    check('defaults still build the bare GT_BceDiceLoss', type(CL.build_criterion()) is GT_BceDiceLoss)
    single = CL.build_criterion(extra_term='bl', extra_weight='0.095')
    check('bl alone still builds EXP-6 WithExtraTerm (one name)',
          type(single) is CL.WithExtraTerm and single.names == ['bl'])
    check('describe() of EXP-6/7/8 configs unchanged',
          CL.describe('bcedice', 'bl', '0.095', snbl_tau=3.0, area_delta=0.05) == 'bcedice + 0.095 * bl'
          and CL.describe('bcedice', 'bl', '0.0475') == 'bcedice + 0.0475 * bl')
    d1 = CL.describe('bcedice', 'bl', '0.095', boundary_weight=1.0)
    d0 = CL.describe('bcedice', 'bl', '0.095', boundary_weight=0.0)
    check('describe() tells the band loss on / logged-only apart',
          d1.startswith('bcedice + 0.095 * bl + 1 * bnd') and 'logged only' in d0, f'{d1} | {d0}')
    target = torch.cat([disc(40), disc(15, 60, 190)])
    bnd = {'dec3': torch.randn(2, 1, 32, 32), 'dec4': torch.randn(2, 1, 64, 64), 'dec5': torch.randn(2, 1, 128, 128)}
    gt_pre = DeepSupervisionOutputs(tuple(torch.sigmoid(torch.randn(2, 1, H, W)) for _ in range(5)), boundary=bnd)
    out = torch.sigmoid(torch.randn(2, 1, H, W))
    bd = BceDiceLoss(wb=0.5, wd=1)
    band = CL.boundary_band(target)
    hand = {s: bd(torch.sigmoid(bnd[s]), CL.band_at_grid(band, bnd[s].shape[-2:])) for s in bnd}
    for inner_label, kw, inner_value in (
            ('GT_BceDice', {}, GT_BceDiceLoss(1, 1)(gt_pre, out, target)),
            ('GT_BceDice + 0.095 bl', {'extra_term': 'bl', 'extra_weight': '0.095'},
             GT_BceDiceLoss(1, 1)(gt_pre, out, target) + 0.095 * CL.BoundaryLoss()(out, target))):
        crit = CL.build_criterion(boundary_weight=1.0, **kw)
        total = crit(gt_pre, out, target)
        expect = inner_value + 1.0 * (0.1 * hand['dec3'] + 0.2 * hand['dec4'] + 0.3 * hand['dec5'])
        check(f'{inner_label}: loss = inner + (0.1 BD3 + 0.2 BD4 + 0.3 BD5)', abs(float(total - expect)) < 1e-5,
              f'{float(total):.6f} vs {float(expect):.6f}')
        means = crit.epoch_means()
        want = (['bl'] if kw else []) + ['bnd_dec3', 'bnd_dec4', 'bnd_dec5']
        check(f'{inner_label}: names / running means {want}', crit.names == want and list(means) == want
              and all(abs(means[f'bnd_{s}'] - float(hand[s])) < 1e-6 for s in bnd))
    crit0 = CL.build_criterion(extra_term='bl', extra_weight='0.095', boundary_weight=0.0)
    inner0 = CL.build_criterion(extra_term='bl', extra_weight='0.095')
    check('weight 0: value identical to the inner criterion',
          torch.equal(crit0(gt_pre, out, target), inner0(gt_pre, out, target)))
    lg = {k: v.clone().requires_grad_(True) for k, v in bnd.items()}
    out_g = out.clone().requires_grad_(True)
    crit0(DeepSupervisionOutputs(tuple(gt_pre), boundary=lg), out_g, target).backward()
    check('weight 0: no gradient reaches the boundary logits', all(v.grad is None for v in lg.values())
          and out_g.grad is not None)
    check('weight 0: band terms still logged', all(np.isfinite(v) for v in crit0.epoch_means().values()))
    try:
        CL.build_criterion(boundary_weight=1.0)(tuple(gt_pre), out, target)
        check('a plain tuple (no boundary logits) is refused', False)
    except ValueError:
        check('a plain tuple (no boundary logits) is refused', True)
    try:
        CL.build_criterion(boundary_weight=-1.0)
        check('negative boundary weight is refused', False)
    except ValueError:
        check('negative boundary weight is refused', True)

    print('7. gradient flow')
    target = torch.cat([disc(40), disc(15, 60, 190)])
    for label, weight in (('A3 (band loss on)', 1.0), ('A4e (band loss logged only)', 0.0)):
        m = bg()
        m.train()
        crit = CL.build_criterion(extra_term='bl', extra_weight='0.095', boundary_weight=weight)
        gp, o = m(x)
        crit(gp, o, target).backward()
        grads = {n: (p.grad is not None and float(p.grad.abs().sum()) > 0) for n, p in m.fusion.named_parameters()}
        heads_ok = all(v for n, v in grads.items() if n.startswith('heads.'))
        bnd_ok = all(v == (weight > 0) for n, v in grads.items() if n.startswith('bnd_heads.'))
        rest_zero = not any(v for n, v in grads.items() if n.startswith(('proj.', 'sum_w.', 'bg_alpha.')))
        check(f'{label}, step 1: fusion heads get gradient', heads_ok)
        check(f'{label}, step 1: boundary heads get gradient {"from the band loss" if weight else "= 0"}', bnd_ok)
        check(f'{label}, step 1: proj / w / alpha still 0 (heads start at zero)', rest_zero)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-2)
        before = {n: p.detach().clone() for n, p in m.fusion.named_parameters()}
        for _ in range(3):
            opt.zero_grad()
            gp, o = m(x)
            crit(gp, o, target).backward()
            opt.step()
        moved = all(not torch.equal(before[n], p.detach()) for n, p in m.fusion.named_parameters())
        check(f'{label}: every fusion tensor moved after 3 AdamW steps', moved)

    print('8. checkpoint round trip (analysis.common)')
    try:
        from analysis import common as AC
        for label, model, mode, dim in (('A3', bg(), 'bg_stage', 8), ('A1e', a1, 'sum', 8),
                                        ('bg all5 d16', bg('all5', 16), 'bg_stage', 16)):
            sd = model.state_dict()
            cfg = AC.infer_model_config(sd)
            ok = cfg['fusion_mode'] == mode and cfg['fusion_stages'] == model.fusion_stages and cfg['fusion_dim'] == dim
            rebuilt, _ = AC.build_model(sd, strict=True)
            with torch.no_grad():
                ok &= torch.equal(rebuilt(x)[1], model.eval()(x)[1])
            check(f'{label}: config {mode}/{model.fusion_stages}/d{dim} recovered, strict load, same output', ok)
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
