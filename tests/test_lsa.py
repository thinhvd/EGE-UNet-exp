'''
Checks for the EXP-15 lesion-state-aware selective fusion (models/lsa_fusion.py) and its aux losses
(contour_losses.InteriorHardPositiveRegion / WithExpertAux). CPU, about a minute:

    python tests/test_lsa.py [MINI_DATA_ROOT]

1. parameters (CSF 54,965 -> LSA 56,510 in every mode), no lsa keys in older configurations, refusals;
2. same global RNG and projections as the fusion-only model; at init the output equals the ORIGINAL model bit for bit;
   gates: fixed = (0.5, 0.5, 0), route = (0.468, 0.468, 0.063), sum to 1; forced gates and stage_off work;
3. gradients: probe aux loss reaches only the aux heads; supervise aux loss reaches experts + projections but not the
   encoder; the router moves once the heads have left zero; the fixed router never moves;
4. losses: interior-region focal normalized by |H| (hand formula), aux total (hand formula), gate logging, describe();
5. provenance buffers refuse mismatched loads; analysis.common rebuilds the model; SWA averaging;
6. train.py / config keys; (with MINI_DATA_ROOT) one epoch of train.py for route + supervise.
'''
import os
import itertools
import sys
import json
import tempfile
import subprocess

import numpy as np
import torch
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import contour_losses as CL                                           # noqa: E402
from models.egeunet import EGEUNet                                    # noqa: E402
from models.fusion import FUSION_STAGE_SETS, DeepSupervisionOutputs   # noqa: E402
from models.lsa_fusion import lesion_state                            # noqa: E402
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


def targets():
    t = torch.zeros(2, 1, H, W)
    t[0, :, 40:220, 50:200] = 1
    t[1, :, 100:140, 30:80] = 1
    return t


def main():
    mini = sys.argv[1] if len(sys.argv) > 1 else None
    torch.manual_seed(0)
    x = torch.rand(2, 3, H, W) * 255
    t = targets()

    print('1. parameters and refusals')
    b0, d1 = build(), build(**CSF)
    ms = {(m, a): build(lsa_mode=m, lsa_aux=a, **CSF) for m in ('fixed', 'route') for a in ('probe', 'supervise')}
    check('CSF 54965, every LSA arm 56510', nparams(d1) == 54965 and all(nparams(m) == 56510 for m in ms.values()),
          f'{[nparams(m) for m in ms.values()]}')
    check('LSA removed the fusion combine step (no fusion.heads / fusion.sum_w keys)',
          not any(k.startswith(('fusion.heads', 'fusion.sum_w')) for k in ms[('route', 'probe')].state_dict()))
    for label, m in (('baseline', b0), ('CSF', d1), ('propagate', build(propagate_mode='pair', propagate_tau=0.3, propagate_kappa=5.0,
                                                                         propagate_beta=0.5, **CSF))):
        check(f'{label}: no lsa keys', not any(k.startswith('lsa.') for k in m.state_dict()))
    for bad, kw in (('without fusion', dict(lsa_mode='route')), ('with deep3', dict(lsa_mode='route', fusion_mode='sum',
                    fusion_stages=FUSION_STAGE_SETS['deep3'])), ('with complete', dict(lsa_mode='route', complete_mode='full', **CSF)),
                    ('without gt_ds', dict(lsa_mode='route', gt_ds=False, **CSF)), ('unknown aux', dict(lsa_mode='route', lsa_aux='x', **CSF))):
        try:
            build(**kw)
            check(f'lsa {bad} is refused', False)
        except ValueError:
            check(f'lsa {bad} is refused', True)

    print('2. RNG, projections, start = original model, gates')
    draws = {}
    for label, kw in (('CSF', CSF), ('fixed', dict(lsa_mode='fixed', **CSF)), ('route', dict(lsa_mode='route', lsa_aux='supervise', **CSF))):
        build(**kw)
        draws[label] = torch.rand(4)
    check('LSA leaves the global RNG where the fusion-only model leaves it', torch.equal(draws['CSF'], draws['fixed'])
          and torch.equal(draws['CSF'], draws['route']))
    sp = {k: v for k, v in d1.state_dict().items() if k.startswith('fusion.proj')}
    check('projections identical to the fusion-only model', all(all(torch.equal(sp[k], m.state_dict()[k]) for k in sp) for m in ms.values()))
    sb = b0.state_dict()
    check('trunk tensors identical to the original model', all(all(torch.equal(sb[k], m.state_dict()[k]) for k in sb) for m in ms.values()))
    for m in [b0] + list(ms.values()):
        m.eval()
    with torch.no_grad():
        o0 = b0(x)[1]
        outs = {k: m(x) for k, m in ms.items()}
    check('at init every LSA arm outputs exactly the original model', all(torch.equal(o[1], o0) for o in outs.values()))
    gf, gr = outs[('fixed', 'probe')][0].boundary['lsa_g'], outs[('route', 'probe')][0].boundary['lsa_g']
    check('fixed gates = (0.5, 0.5, 0) at every stage', all(torch.allclose(g.mean((0, 2, 3)), torch.tensor([0.5, 0.5, 0.0])) for g in gf.values()))
    want = torch.softmax(torch.tensor([0.0, 0.0, -2.0]), 0)
    check('route gates at init = softmax(0, 0, -2) = (0.468, 0.468, 0.063), sum to 1',
          all(torch.allclose(g.mean((0, 2, 3)), want, atol=1e-6) and torch.allclose(g.sum(1), torch.ones(1)) for g in gr.values()))
    check('aux logits: detail (2, 128, 128), completion (1, 32, 32)', tuple(outs[('route', 'probe')][0].boundary['lsa_d'].shape) == (2, 2, 128, 128)
          and tuple(outs[('route', 'probe')][0].boundary['lsa_c'].shape) == (2, 1, 32, 32))
    m = build(lsa_mode='route', **CSF)
    with torch.no_grad():
        for p in m.lsa.parameters():
            p.add_(torch.randn_like(p) * 0.2)
    m.eval()
    with torch.no_grad():
        o_r = m(x)
        m.lsa.force_gates = (0.0, 0.0, 1.0)
        o_null = m(x)
        m.lsa.force_gates = None
        m.lsa.stage_off = {'dec3', 'dec4', 'dec5'}
        o_off = m(x)
        m.lsa.stage_off = set()
        m.lsa.force_gates = (1.0, 0.0, 0.0)
        g_d = m(x)[0].boundary['lsa_g']
        m.lsa.force_gates = None
    trunk = {k: v for k, v in m.state_dict().items() if not k.startswith(('lsa.', 'fusion.'))}
    b_same = build()
    b_same.load_state_dict(trunk, strict=False)
    b_same.eval()
    with torch.no_grad():
        o_trunk = b_same(x)[1]
    check('forced null gates and all stages off both give the trunk-only output', torch.equal(o_null[1], o_trunk) and torch.equal(o_off[1], o_trunk)
          and not torch.equal(o_r[1], o_trunk))
    check('forced gates (1, 0, 0) are reported as such', all(torch.allclose(g.mean((0, 2, 3)), torch.tensor([1.0, 0.0, 0.0])) for g in g_d.values()))
    q = torch.zeros(1, 1, 32, 32)
    q[..., 8:24, 8:24] = 1
    ls = lesion_state(q)
    check('lesion_state: q, local means (k 3 / 11), image mean', ls.shape[1] == 4 and float(ls[0, 3].mean()) == float(q.mean())
          and float(ls[0, 1, 16, 16]) == 1.0 and 0 < float(ls[0, 2, 8, 8]) < 1)

    print('3. gradients')
    for aux in ('probe', 'supervise'):
        mm = build(lsa_mode='route', lsa_aux=aux, **CSF)
        with torch.no_grad():
            for p in mm.lsa.parameters():
                p.add_(torch.randn_like(p) * 0.1)
        mm.train()
        gp, o = mm(x)
        crit = CL.build_criterion(lsa_aux=aux)
        terms = crit.aux_terms(gp.boundary, t)
        (0.5 * (terms['aux_d_seg'] + terms['aux_d_ring']) + 0.3 * (terms['aux_c_seg'] + terms['aux_c_ihr'])).backward()
        has = lambda pre: any(p.grad is not None and float(p.grad.abs().sum()) > 0 for n, p in mm.named_parameters() if n.startswith(pre))
        enc = any(p.grad is not None and float(p.grad.abs().sum()) > 0 for n, p in mm.named_parameters()
                  if not n.startswith(('lsa.', 'fusion.')))
        if aux == 'probe':
            check('probe: aux loss reaches the aux heads only', has('lsa.aux_') and not has('lsa.mix_') and not has('fusion.proj') and not enc)
        else:
            check('supervise: aux loss reaches experts + projections, never the encoder', has('lsa.aux_') and has('lsa.mix_')
                  and has('fusion.proj') and not enc)
    for mode in ('route', 'fixed'):
        mm = build(lsa_mode=mode, lsa_aux='supervise', **CSF)
        mm.train()
        crit = CL.build_criterion(lsa_aux='supervise')
        opt = torch.optim.AdamW(mm.parameters(), lr=1e-3, weight_decay=1e-2)
        r0 = {k: v.clone() for k, v in mm.lsa.router.state_dict().items()}
        for _ in range(3):
            opt.zero_grad()
            gp, o = mm(x)
            crit(gp, o, t).backward()
            opt.step()
        moved = any(not torch.equal(r0[k], v) for k, v in mm.lsa.router.state_dict().items())
        check(f'{mode}: router {"moves after the heads leave zero" if mode == "route" else "never moves"}', moved == (mode == 'route'))

    print('4. losses and describe()')
    u = torch.full((2, 1, H, W), 0.3)
    ihr = CL.InteriorHardPositiveRegion(0.15, 2.0)
    hmask = ((CL.normalized_depth(t) >= 0.15) & (t >= 0.5)).float()
    hand = float((hmask * 0.7 ** 2 * -np.log(0.3)).sum() / hmask.sum())
    check('interior-region focal = sum_H (1-u)^2 (-log u) / |H| (batch)', abs(float(ihr(u, t)) - hand) < 1e-5, f'{float(ihr(u, t)):.5f} vs {hand:.5f}')
    check('no interior pixel -> 0', float(ihr(u, torch.zeros_like(t))) == 0.0)
    ad, ac = torch.randn(2, 2, 128, 128), torch.randn(2, 1, 32, 32)
    g_route = {s: torch.softmax(torch.randn(2, 3, sz, sz), 1) for s, sz in (('dec3', 32), ('dec4', 64), ('dec5', 128))}
    gp = DeepSupervisionOutputs(tuple(torch.sigmoid(torch.randn(2, 1, H, W)) for _ in range(5)),
                                boundary={'lsa_d': ad, 'lsa_c': ac, 'lsa_g': g_route})
    out = torch.sigmoid(torch.randn(2, 1, H, W))
    crit = CL.build_criterion(lsa_aux='route' if False else 'probe')
    up = lambda z: torch.sigmoid(F.interpolate(z, size=(H, W), mode='bilinear', align_corners=True))
    d, c = up(ad), up(ac)
    expect = float(GT_BceDiceLoss(1, 1)(gp, out, t)) + 0.5 * (float(BceDiceLoss(1, 1)(d[:, :1], t)) + float(BceDiceLoss(0.5, 1)(d[:, 1:], CL.boundary_ring(t, 3)))) \
        + 0.3 * (float(BceDiceLoss(1, 1)(c, t)) + float(ihr(c, t)))
    check('aux total = original + 0.5 [BD seg + BD(.5,1) ring] + 0.3 [BD seg + interior focal]', abs(float(crit(gp, out, t)) - expect) < 1e-4)
    em = crit.epoch_means()
    les = F.adaptive_avg_pool2d(t, (128, 128)) >= 0.5
    check('gate logging: g_null inside the lesion at dec5', abs(em['gnull_les_dec5'] - float(g_route['dec5'][:, 2:3][les].mean())) < 1e-5)
    ds_ = CL.describe(lsa_aux='supervise', lsa_tau=0.15)
    check('describe() ASCII and names the mode', ds_ == 'bcedice + expert aux supervise (0.5*[BD(D seg)+BD(.5,1)(D ring +-3px)] + '
          '0.3*[BD(C seg)+focal interior depth>=0.15 /|H|])' and ds_.isascii(), ds_)
    check('older describe() unchanged', CL.describe(aff_weight=1.0, hard_frac=0.2) ==
          'bcedice + 1 * aff (edge BCE same-label: 0.5 cut / 0.25 LL+BB / 0.25 hard 20% contrast)')
    try:
        CL.build_criterion(lsa_aux='supervise', boundary_weight=0.3)
        check('expert aux + boundary loss refused', False)
    except ValueError:
        check('expert aux + boundary loss refused', True)

    print('5. provenance, rebuild, SWA')
    sd = ms[('route', 'probe')].state_dict()
    for label, kw in (('route checkpoint into fixed', dict(lsa_mode='fixed')), ('probe checkpoint into supervise', dict(lsa_mode='route', lsa_aux='supervise'))):
        try:
            build(**kw, **CSF).load_state_dict(sd, strict=True)
            check(f'{label} is refused', False)
        except RuntimeError:
            check(f'{label} is refused', True)
    from analysis import common as AC
    for (mode, aux), mm in ms.items():
        sdm = mm.state_dict()
        cfg = AC.infer_model_config(sdm)
        rb, _ = AC.build_model(sdm, strict=True)
        with torch.no_grad():
            same = torch.equal(rb(x)[1], mm.eval()(x)[1])
        check(f'{mode}/{aux}: configuration recovered, strict load, same output', cfg.get('lsa_mode') == mode and cfg.get('lsa_aux') == aux
              and cfg['fusion_mode'] == 'sum' and cfg['fusion_stages'] == ['dec3', 'dec4', 'dec5'] and cfg['fusion_dim'] == 8 and same)
    check('older configurations infer no lsa keys', 'lsa_mode' not in AC.infer_model_config(d1.state_dict()))
    from analysis.swa_average import average_state_dicts
    a = build(lsa_mode='route', **CSF).state_dict()
    b = {k: (v + 1 if v.is_floating_point() else v) for k, v in a.items()}
    avg = average_state_dicts([a, b])
    check("SWA: float tensors averaged, integer buffers kept", all(torch.allclose(avg[k], a[k] + 0.5, atol=1e-6) for k in a if a[k].is_floating_point())
          and all(torch.equal(avg[k], a[k]) for k in a if not a[k].is_floating_point()))

    print('6. train.py / config')
    py = sys.executable
    probe = ('import sys, json; sys.argv = ["train.py"] + json.loads(sys.argv[1]); import train; '
             'from configs.config_setting import get_config; c = get_config(train.parse_args()); '
             'print("CFG" + json.dumps({"model": {k: v for k, v in c.model_config.items() if k.startswith("lsa")}, '
             '"loss": sorted(getattr(c, "loss_config", {}).keys())}))')

    def cfg(flags):
        p = subprocess.run([py, '-c', probe, json.dumps(flags)], cwd=REPO, capture_output=True, text=True)
        line = [ln for ln in p.stdout.splitlines() if ln.startswith('CFG')]
        return p.returncode, (json.loads(line[0][3:]) if line else None)
    base = ['--work-dir', '/tmp/x', '--device', 'cpu']
    csf = ['--fusion', 'sum', '--fusion-stages', 'shallow3', '--fusion-dim', '8']
    rc, c0 = cfg(base + csf)
    check('CSF flags: no lsa keys, no loss_config', rc == 0 and c0 == {'model': {}, 'loss': []}, str(c0))
    rc, c1 = cfg(base + csf + ['--lsa', 'route', '--lsa-aux', 'supervise'])
    check('lsa flags: lsa_mode / lsa_aux, loss lsa_aux / lsa_tau', rc == 0 and c1['model'] == {'lsa_mode': 'route', 'lsa_aux': 'supervise'}
          and {'lsa_aux', 'lsa_tau'} <= set(c1['loss']), str(c1))
    for label, flags in (('--lsa with deep3', ['--fusion', 'sum', '--fusion-stages', 'deep3', '--lsa', 'route']),
                         ('--lsa with --boundary-weight', csf + ['--lsa', 'route', '--boundary-weight', '0.1'])):
        rc, _ = cfg(base + flags)
        check(f'{label} refused', rc == 2)
    # EXP-17: an extra term on the final output (E3a) combines with the expert aux losses
    rc, c2 = cfg(base + csf + ['--lsa', 'route', '--extra-term', 'bl', '--extra-weight', '0.095'])
    check('--lsa with --extra-term bl accepted (loss extra_term + lsa_aux)', rc == 0 and {'extra_term', 'extra_weight', 'lsa_aux'} <= set(c2['loss']), str(c2))
    if mini and os.path.isdir(os.path.join(mini, 'test', 'images')):
        wd = tempfile.mkdtemp()
        p = subprocess.run([py, '-W', 'ignore', 'train.py', '--work-dir', wd, '--data-path', mini, '--dataset', 'isic17', '--epochs', '1',
                            '--batch-size', '8', '--device', 'cpu', '--seed', '42', '--no-resume', *csf, '--lsa', 'route', '--lsa-aux',
                            'supervise', '--test-split', 'test'], cwd=REPO, capture_output=True, text=True)
        head = open(os.path.join(wd, 'metrics.csv')).readline() if os.path.isfile(os.path.join(wd, 'metrics.csv')) else ''
        check('one epoch of train.py (route + supervise): test_results.json, aux and gate columns', p.returncode == 0
              and os.path.isfile(os.path.join(wd, 'test_results.json')) and 'train_extra_aux_c_ihr' in head and 'train_extra_gnull_les_dec5' in head,
              p.stderr[-300:])
    else:
        print('  [skip] no MINI_DATA_ROOT')

    print('7. analysis/exp15_mechanism.py helpers')
    from analysis import exp15_mechanism as M
    sq = np.zeros((64, 64), bool)
    sq[16:48, 16:48] = True
    e = M.edge(sq)
    check('edge(): 32x32 square has 124 contour pixels', int(e.sum()) == 124, str(int(e.sum())))
    ed = M.distance_transform_edt(~e)
    sh1, sh3 = np.roll(sq, 1, 1), np.roll(sq, 3, 1)
    check('bf_score: identical 1, shifted 1 px 1, shifted 3 px < 1, empty pred 0',
          M.bf_score(sq, e, ed) == 1.0 and M.bf_score(sh1, e, ed) == 1.0 and M.bf_score(sh3, e, ed) < 1.0
          and M.bf_score(np.zeros_like(sq), e, ed) == 0.0)
    rng = np.random.default_rng(0)
    cnt = rng.integers(0, 50, size=(4, 30, 4, 3)).astype(float)
    tert = rng.integers(0, 3, size=30)
    rule = M.fit_rule(cnt, tert, 3)
    base = M.rule_score(cnt, tert, rule).mean()
    local = True
    for t, z, v in itertools.product(range(3), range(4), range(4)):
        r2 = rule.copy()
        r2[t, z] = v
        local &= M.rule_score(cnt, tert, r2).mean() <= base + 1e-12
    hand = np.mean([M.C.dsc_from_counts(*cnt[rule[tert[i]], i, np.arange(4)].sum(0)) for i in range(30)])
    check('fit_rule: no single change improves it; rule_score = hand DSC from summed zone counts', local and abs(hand - base) < 1e-12)
    zone_best = max(M.dsc3(cnt[list(c), 0, np.arange(4)].sum(0)) for c in itertools.product(range(4), repeat=4))
    check('zone oracle >= image oracle >= every variant (image 0)', zone_best >= max(M.dsc3(cnt[v, 0].sum(0)) for v in range(4)) - 1e-12)
    mr = build(lsa_mode='route', **CSF)
    for h in list(mr.lsa.head_d.values()) + list(mr.lsa.head_c.values()):
        torch.nn.init.normal_(h.weight, std=0.1)
    mr.eval()
    x = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        y0 = mr(x)[1]
        for v in M.VARIANTS:
            undo = M.set_variant(mr, v)
            yv = mr(x)[1]
            undo()
            check(f'set_variant {v}: changes the output unless actual, undone after', (v == 'actual') == torch.equal(yv, y0)
                  and mr.lsa.force_gates is None and torch.equal(mr(x)[1], y0))
        rec = M.Recorder(mr.lsa)
        mr(x)
        st = rec.store['dec4']
        rec.remove()
        f_hand = st['g'][:, 0:1] * st['cd'] + st['g'][:, 1:2] * st['cc']
        check('Recorder: f = gD * head_D(F_D) + gC * head_C(F_C); stage restored', torch.allclose(f_hand, st['f'], atol=1e-6)
              and 'stage' not in mr.lsa.__dict__ and torch.equal(mr(x)[1], y0))
    mc = build(**CSF)
    for h in mc.fusion.heads.values():
        torch.nn.init.normal_(h.weight, std=0.1)
    mc.eval()
    with torch.no_grad():
        y0 = mc(x)[1]
        ok = True
        for v in M.CSF_VARIANTS:
            undo = M.set_variant(mc, v)
            yv = mc(x)[1]
            undo()
            ok &= ((v == 'actual') == torch.equal(yv, y0)) and torch.equal(mc(x)[1], y0)
        check('set_variant on plain CSF (control): off_* change the output, weights restored', ok)

    print()
    if FAILED:
        print(f'{len(FAILED)} check(s) FAILED: {FAILED}')
        return 1
    print('all checks passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
