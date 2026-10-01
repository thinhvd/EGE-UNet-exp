"""EXP-13: Boundary-Constrained Core-to-Region Propagation (CRP) at dec5, on top of the cross-stage fusion.

EXP-11 already grew the confident core along 8-connected paths, but with UNARY weights: an evidence map A that
had no supervision of its own (AUC 0.31-0.64 on the missed rim vs nearby skin) and a ring-trained barrier that
fired on the dark-core edge; the segmentation loss flowing back through the growth taught both to stop at the
core. EXP-12 then showed that the evidence for the rim exists but is RELATIVE: inside each image the model's own
probability ranks the rim above the nearby skin (per-image AUC 0.98), one global level does not.

This module propagates from the core along EDGES between neighbouring cells of the 128 grid:

    g     = GELU(GN(conv1x1(cat[D5, P1..P5, z_base, rgb, t_z, t_rgb])))        8 channels, inputs DETACHED
    S_o   = sigmoid(mlp([g_x + g_y, |g_x - g_y|, |g_{x-o} - g_{x+2o}|]) + b_o)  one value per edge (x, x+o),
                                                                             4 half-offsets o, symmetric
    U     = sigmoid(kappa * (t_z - tau))                                       relative cap (destination cell)
    W     = S^beta * U   ('pair', 'pair_coupled');   W = S^beta  ('pair_nocap')
    R_0   = C = 1[sigmoid(z_base) >= seed_thr];   R <- max(R, max_k shift_k(R) * W_k)   (steps times)
    G     = R - C;   p = 1 - (1 - sigmoid(up(z_base))) * (1 - up(G))           probabilistic OR at 256

S = P(both cells carry the same label): the affinity and the barrier of the original idea in one head (the
barrier is 1 - S on the edge). t_z = (z - m_bg) / (m_core - m_bg) places each cell between this image's mean
background logit (cells with p <= 0.05) and its mean core logit (seed cells); t_rgb does the same with colour on
the image's background -> core colour axis. The cap U makes a leak through a gap of the contour die out as soon
as it reaches cells that sit at the image's background level, instead of flooding the skin.

Gradients: in 'pair' and 'pair_nocap' the edge head learns only from the affinity loss on GT edge labels
(contour_losses.WithPropagationAux) and the propagation runs without gradient; the backbone sees the output
through p_base only (dp / dp_base = 1 - G). 'pair_coupled' lets the segmentation loss (and the completion loss)
reach S through the propagation, which is how EXP-11 trained its maps.

Initialization (set_init, AFTER the parent's apply(_init_weights)): the last conv of the edge MLP has zero weights
and bias -2, so S = 0.119 everywhere: G <= 0.12 next to the seed and <= 0.015 two cells away.

Mode, steps, seed threshold, tau, kappa and beta drive forward() as plain attributes and are stored as buffers;
loading a checkpoint into a module built with other values raises. Switches for readouts only
(analysis/exp13_mechanism.py): enabled (read by EGEUNet.forward), s_one (S = 1: only the relative cap),
cap_off (U = 1), steps_override.
"""
import torch
from torch import nn
import torch.nn.functional as F

PROPAGATE_MODES = ('none', 'pair', 'pair_coupled', 'pair_nocap')
_MODE_ID = {'pair': 0, 'pair_coupled': 1, 'pair_nocap': 2}
OFFS_HALF = ((0, 1), (1, 0), (1, 1), (1, -1))                      # (dy, dx): right, down, down-right, down-left
DIRS8 = OFFS_HALF + tuple((-dy, -dx) for dy, dx in OFFS_HALF)        # neighbour x + k feeds cell x


def shift(t, dy, dx, fill=0.0):
    '''out[..., y, x] = t[..., y + dy, x + dx]; positions outside the grid get `fill`.'''
    p = max(abs(dy), abs(dx))
    if p == 0:
        return t
    h, w = t.shape[-2:]
    tp = F.pad(t, (p, p, p, p), value=fill)
    return tp[..., p + dy:p + dy + h, p + dx:p + dx + w]


def full_weights(s_half, u=None, beta=1.0):
    '''(B,4,H,W) edge values S(x, x+o) -> (B,8,H,W) weights into each cell x from its neighbour x+k (DIRS8 order):
    k = +o reads S stored at x, k = -o reads S stored at x - o, so both directions of an edge carry the same S.
    The cap u (B,1,H,W) multiplies at the destination.'''
    w = [s_half[:, i] for i in range(4)] + [shift(s_half[:, i], -dy, -dx) for i, (dy, dx) in enumerate(OFFS_HALF)]
    w = torch.stack(w, 1)
    if beta != 1.0:
        w = w.clamp_min(0) ** beta
    return w if u is None else w * u


def propagate(seed, w8, steps):
    '''R_0 = seed; R_t = max(R_{t-1}, max_k R_{t-1}(x + k) * w8_k(x)) for t = 1..steps. seed (B,1,H,W) in [0,1].'''
    r = seed
    h, w = r.shape[-2:]
    for _ in range(int(steps)):
        rp = F.pad(r, (1, 1, 1, 1))
        nb = torch.cat([rp[:, :, 1 + dy:1 + dy + h, 1 + dx:1 + dx + w] for dy, dx in DIRS8], 1)
        r = torch.maximum(r, (nb * w8).amax(1, keepdim=True))
    return r


def image_relative(z, rgb, seed_thr, bg_thr=0.05, clamp=3.0):
    '''Per image: t_z = (z - m_bg) / (m_core - m_bg) and t_rgb = projection of the colour on the background -> core
    colour axis, both 0 at the background mean and 1 at the core mean. Images without core or background cells
    (or a degenerate axis) get t = 0 and valid = False. Returns (t_z, t_rgb, valid (B,) bool).'''
    with torch.no_grad():
        p = torch.sigmoid(z)
        core = (p >= seed_thr).float()
        bg = (p <= bg_thr).float()
        n_core, n_bg = core.sum((1, 2, 3)), bg.sum((1, 2, 3))
        m_core = (z * core).sum((1, 2, 3)) / n_core.clamp_min(1)
        m_bg = (z * bg).sum((1, 2, 3)) / n_bg.clamp_min(1)
        span = m_core - m_bg
        valid = (n_core > 0) & (n_bg > 0) & (span > 1e-3)
        t_z = (z - m_bg.view(-1, 1, 1, 1)) / span.clamp_min(1e-3).view(-1, 1, 1, 1)
        c_core = (rgb * core).sum((2, 3)) / n_core.clamp_min(1).view(-1, 1)
        c_bg = (rgb * bg).sum((2, 3)) / n_bg.clamp_min(1).view(-1, 1)
        axis = c_core - c_bg
        norm2 = (axis ** 2).sum(1)
        valid_rgb = valid & (norm2 > 1e-6)
        t_rgb = ((rgb - c_bg[:, :, None, None]) * axis[:, :, None, None]).sum(1, keepdim=True) / \
            norm2.clamp_min(1e-6).view(-1, 1, 1, 1)
        m = valid.float().view(-1, 1, 1, 1)
        t_z = (t_z * m).clamp(-clamp, clamp)
        t_rgb = (t_rgb * valid_rgb.float().view(-1, 1, 1, 1)).clamp(-clamp, clamp)
    return t_z, t_rgb, valid


class CorePropagation(nn.Module):
    def __init__(self, c_dec, fdim, mode, steps=32, seed_thr=0.9, tau=0.3, kappa=10.0, beta=1.0, n_src=5, hidden=8):
        super().__init__()
        if mode not in PROPAGATE_MODES or mode == 'none':
            raise ValueError(f'propagate mode must be one of {PROPAGATE_MODES[1:]}, got {mode!r}')
        if int(steps) < 1:
            raise ValueError(f'propagate steps must be >= 1, got {steps}')
        if not 0.5 <= float(seed_thr) < 1:
            raise ValueError(f'propagate seed threshold must be in [0.5, 1), got {seed_thr}')
        if float(kappa) <= 0 or float(beta) <= 0:
            raise ValueError(f'propagate kappa and beta must be > 0, got {kappa}, {beta}')
        self.mode = mode
        self.steps, self.seed_thr = int(steps), float(seed_thr)
        self.tau, self.kappa, self.beta = float(tau), float(kappa), float(beta)
        self.n_src = n_src
        self.enabled = True
        self.s_one = False
        self.cap_off = False
        self.steps_override = None
        # registration order (cell, edge, off_bias) fixes the RNG draws and the state_dict keys
        n_in = c_dec + n_src * fdim + 1 + 3 + 2
        self.cell = nn.Sequential(nn.Conv2d(n_in, hidden, 1), nn.GroupNorm(4, hidden), nn.GELU())
        self.edge = nn.Sequential(nn.Conv2d(3 * hidden, hidden, 1), nn.GELU(), nn.Conv2d(hidden, 1, 1))
        self.off_bias = nn.Parameter(torch.zeros(len(OFFS_HALF)))
        for name, v, dt in (('mode_id', _MODE_ID[mode], torch.long), ('n_steps', self.steps, torch.long),
                            ('seed_threshold', self.seed_thr, torch.float32), ('cap_tau', self.tau, torch.float32),
                            ('cap_kappa', self.kappa, torch.float32), ('edge_beta', self.beta, torch.float32)):
            self.register_buffer(name, torch.tensor(v, dtype=dt))

    def set_init(self):
        """S = sigmoid(-2) = 0.119 on every edge. Call AFTER apply(_init_weights)."""
        last = self.edge[-1]
        nn.init.zeros_(last.weight)
        nn.init.constant_(last.bias, -2.0)
        with torch.no_grad():
            self.off_bias.zero_()

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        mine = {'mode_id': int(self.mode_id), 'n_steps': int(self.n_steps)}
        mine_f = {k: float(getattr(self, k)) for k in ('seed_threshold', 'cap_tau', 'cap_kappa', 'edge_beta')}
        for k, v in list(mine.items()) + list(mine_f.items()):
            if prefix + k not in state_dict:
                continue
            got = state_dict[prefix + k]
            bad = (int(got) != v) if k in mine else (abs(float(got) - v) > 1e-6)
            if bad:
                raise RuntimeError(f'{prefix}{k}: the checkpoint was trained with {got.item()}, this model is built with {v} '
                                   f'(mode ids {_MODE_ID}); build it with the checkpoint\'s --propagate settings')
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    @property
    def coupled(self):
        return self.mode == 'pair_coupled'

    def cell_features(self, z_base, d5, projs, rgb):
        size = d5.shape[2:4]
        p = [q if q.shape[2:4] == size else F.interpolate(q, size=size, mode='bilinear', align_corners=True)
             for q in projs[:self.n_src]]
        t_z, t_rgb, valid = image_relative(z_base.detach(), rgb, self.seed_thr)
        x = torch.cat([d5, *p, z_base, rgb, t_z, t_rgb], 1).detach()
        return self.cell(x), t_z, valid, p[0].detach()

    def edge_logits(self, g):
        '''(B,4,H,W) edge logits; edges leaving the grid get a finite value but are never used.'''
        out = []
        for i, (dy, dx) in enumerate(OFFS_HALF):
            gy = shift(g, dy, dx)
            pair = torch.cat([g + gy, (g - gy).abs(), (shift(g, -dy, -dx) - shift(g, 2 * dy, 2 * dx)).abs()], 1)
            out.append(self.edge(pair) + self.off_bias[i])
        return torch.cat(out, 1)

    @staticmethod
    def edge_contrast(rgb, p1):
        '''(B,4,H,W) colour + P1 contrast across each edge, each part divided by its per-image mean (for hard negatives).'''
        out = []
        for dy, dx in OFFS_HALF:
            c = (rgb - shift(rgb, dy, dx)).norm(dim=1, keepdim=True)
            f = (p1 - shift(p1, dy, dx)).norm(dim=1, keepdim=True)
            out.append(c / c.mean((2, 3), keepdim=True).clamp_min(1e-6) + f / f.mean((2, 3), keepdim=True).clamp_min(1e-6))
        return torch.cat(out, 1).detach()

    def forward(self, z_base, d5, projs, rgb):
        """rgb = the network input average-pooled to the 128 grid (0-255 after the author's min-max normalization).
        Returns dict: g (grown part G at 128, B,1,H,W), s_logit (B,4,H,W), r (R_T), seed, u (cap), t_z, contrast."""
        rgb = rgb / 255.0
        g, t_z, valid, p1 = self.cell_features(z_base, d5, projs, rgb)
        s_logit = self.edge_logits(g)
        seed = (torch.sigmoid(z_base.detach()) >= self.seed_thr).to(z_base.dtype)
        if self.mode == 'pair_nocap' or self.cap_off:
            u = torch.ones_like(t_z)
        else:
            # images without a usable background / core get t_z = 0 from image_relative; without background the cap
            # would block everything, so open it there (no seed means no propagation anyway)
            t_cap = torch.where(valid.view(-1, 1, 1, 1), t_z, torch.ones_like(t_z))
            u = torch.sigmoid(self.kappa * (t_cap - self.tau))
        steps = self.steps if self.steps_override is None else int(self.steps_override)
        if self.coupled:
            s = torch.ones_like(s_logit) if self.s_one else torch.sigmoid(s_logit)
            r = propagate(seed, full_weights(s, u, self.beta), steps)
        else:
            with torch.no_grad():
                s = torch.ones_like(s_logit) if self.s_one else torch.sigmoid(s_logit)
                r = propagate(seed, full_weights(s, u, self.beta), steps)
        return {'g': r - seed, 's_logit': s_logit, 'r': r, 'seed': seed, 'u': u, 't_z': t_z,
                'contrast': self.edge_contrast(rgb, p1)}
