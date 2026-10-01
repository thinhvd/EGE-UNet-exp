"""EXP-11: Boundary-Constrained Region Completion (BCRC) at dec5, on top of the cross-stage fusion.

The failure it targets (reports/large_interior_fn/): on large lesions the model paints the dark core
with near-perfect precision and misses a pale ring 10-50 px wide inside the true contour, with a low
probability there. EXP-10's 3x3 residual on the logit (6 px reach) could not reach that ring without
also spilling outside. This module grows the model's own confident region outwards, step by step, and
may only ADD lesion:

At dec5 (grid 128 for a 256 input), with D5 the decoder feature after the GAB skip and the fusion add,
P1..P5 the fusion's projections of the five encoder stages (8 channels each, resized to 128) and
z_base = final(D5):

    C   = 1[sigmoid(z_base) >= seed_thr]                     seed: the confident core, no gradient
    A   = sigmoid(evidence(cat[D5, P1..P5, mean_hw(D5)]))     where growth may continue ("is lesion")
    B   = sigmoid(barrier(cat[D5, P1].detach()))              the contour, trained on a +-3 px ring
    R_0 = C
    R_t = max(R_{t-1}, maxpool3x3(R_{t-1}) * A * (1 - B))     'full', t = 1..steps
    R_t = max(R_{t-1}, maxpool3x3(R_{t-1}) * A)               'nobarrier' (B is trained, not used)
    z_final = z_base + alpha * (R_steps - C),  alpha = softplus(rho)

R_t(x) is the best product of A * (1 - B) along a path of at most t cells from the core to x, so a
cell far from the core, or behind a cell with high B or low A, stays near 0. R_steps >= C, hence the
added logit is >= 0 everywhere and 0 on the core itself. evidence = 1x1 conv (56 -> 8) + GroupNorm(4) +
GELU + 3x3 conv (8 -> 1); barrier = 3x3 conv (16 -> 1).

Initialization (set_init, called AFTER the parent's apply(_init_weights)): the last conv of A and the
barrier conv start with zero weights and biases +3 / -3, so A = sigmoid(3) = 0.953 and B = 0.047
everywhere: growth is open at the start and training carves it down. rho = -4 gives alpha = 0.018,
so the output starts within 0.018 logit of the fusion-only model; softplus keeps alpha > 0 and its
gradient alive.

Mode, steps and seed threshold are plain attributes that drive forward(); they are also stored as
buffers (in the state_dict) only to record what a checkpoint was trained with. 'full' and
'nobarrier' have identical parameters, so loading a checkpoint into a module built with another
mode, step count or threshold raises instead of silently changing the arm.

Switches for inference-time readouts only (analysis/exp11_mechanism.py), read by EGEUNet.forward or
here: enabled = False (z_final = z_base), barrier_off (B = 0 in the growth), evidence_one (A = 1),
steps_override (another step count).
"""
import torch
from torch import nn
import torch.nn.functional as F

COMPLETE_MODES = ('none', 'full', 'nobarrier')
_MODE_ID = {'full': 0, 'nobarrier': 1}


def grow(seed, step, steps):
    '''R_0 = seed; R_t = max(R_{t-1}, maxpool3x3(R_{t-1}) * step), t = 1..steps (8-connected: a wall of
    step = 0 must be 4-connected to stop the growth; a one-cell diagonal line lets it through).'''
    r = seed
    for _ in range(int(steps)):
        r = torch.maximum(r, F.max_pool2d(r, 3, 1, 1) * step)
    return r


class RegionCompletion(nn.Module):
    def __init__(self, c_dec, fdim, mode, steps=32, seed_thr=0.9, n_src=5, hidden=8):
        super().__init__()
        if mode not in COMPLETE_MODES or mode == 'none':
            raise ValueError(f'complete mode must be one of {COMPLETE_MODES[1:]}, got {mode!r}')
        if int(steps) < 1:
            raise ValueError(f'complete steps must be >= 1, got {steps}')
        if not 0.5 <= float(seed_thr) < 1:
            raise ValueError(f'complete seed threshold must be in [0.5, 1), got {seed_thr}')
        self.mode = mode
        self.steps = int(steps)
        self.seed_thr = float(seed_thr)
        self.n_src = n_src
        self.enabled = True
        self.barrier_off = False
        self.evidence_one = False
        self.steps_override = None
        # registration order (evidence, barrier, rho) fixes the RNG draws and the state_dict keys
        self.evidence = nn.Sequential(nn.Conv2d(2 * c_dec + n_src * fdim, hidden, 1), nn.GroupNorm(4, hidden),
                                      nn.GELU(), nn.Conv2d(hidden, 1, 3, padding=1))
        self.barrier = nn.Conv2d(c_dec + fdim, 1, 3, padding=1)
        self.rho = nn.Parameter(torch.tensor(-4.0))
        self.register_buffer('mode_id', torch.tensor(_MODE_ID[mode], dtype=torch.long))
        self.register_buffer('n_steps', torch.tensor(self.steps, dtype=torch.long))
        self.register_buffer('seed_threshold', torch.tensor(self.seed_thr, dtype=torch.float32))

    def set_init(self):
        """A = 0.953 and B = 0.047 everywhere, alpha = softplus(-4). Call AFTER apply(_init_weights)."""
        last = self.evidence[-1]
        nn.init.zeros_(last.weight)
        nn.init.constant_(last.bias, 3.0)
        nn.init.zeros_(self.barrier.weight)
        nn.init.constant_(self.barrier.bias, -3.0)
        with torch.no_grad():
            self.rho.fill_(-4.0)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        mine = {'mode_id': int(self.mode_id), 'n_steps': int(self.n_steps), 'seed_threshold': float(self.seed_threshold)}
        for k, v in mine.items():
            if prefix + k in state_dict:
                got = state_dict[prefix + k]
                got = float(got) if k == 'seed_threshold' else int(got)
                if (abs(got - v) > 1e-6) if k == 'seed_threshold' else got != v:
                    raise RuntimeError(f'{prefix}{k}: the checkpoint was trained with {got}, this model is built with {v} '
                                       f'(mode ids {_MODE_ID}); build it with the checkpoint\'s --complete / '
                                       f'--complete-steps / --complete-seed')
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    @property
    def alpha(self):
        return F.softplus(self.rho)

    def forward(self, z_base, d5, projs):
        """Returns (logit to add, barrier logit, R_T, A, B). projs = the fusion's P1..P5 at their own grids."""
        size = d5.shape[2:4]
        p = [q if q.shape[2:4] == size else F.interpolate(q, size=size, mode='bilinear', align_corners=True)
             for q in projs[:self.n_src]]
        ctx = d5.mean(dim=(2, 3), keepdim=True).expand_as(d5)
        a = torch.sigmoid(self.evidence(torch.cat([d5, *p, ctx], dim=1)))
        ring_logit = self.barrier(torch.cat([d5, p[0]], dim=1).detach())
        b = torch.sigmoid(ring_logit)
        seed = (torch.sigmoid(z_base.detach()) >= self.seed_thr).to(z_base.dtype)
        a_use = torch.ones_like(a) if self.evidence_one else a
        if self.mode == 'full' and not self.barrier_off:
            step = a_use * (1 - b)
        else:
            step = a_use
        r = grow(seed, step, self.steps if self.steps_override is None else self.steps_override)
        return self.alpha * (r - seed), ring_logit, r, a, b
