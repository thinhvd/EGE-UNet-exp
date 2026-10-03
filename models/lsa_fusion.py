"""EXP-15: Lesion-State-Aware Selective Cross-Stage Fusion (LSA-CSF).

The plain cross-stage fusion (models/fusion.py, 'sum' on dec3..dec5) adds ONE fused view of the five encoder stages
to every location of the target decoder stages. Averaged over the late epochs it tightened every mask on both datasets
(small lesions spill less, large lesions are missed more): a single shared branch can only move all masks one way,
while small lesions need tighter masks and large faint lesions looser ones. LSA-CSF keeps the fusion's projections
P1..P5 (8 channels each, built exactly as the plain fusion builds them) and replaces its combine step by two
complementary experts and a spatial router:

    F_D = mix_D(cat[P1, up P2, up P3])            Detail expert, grid 128 (shallow / mid stages)
    F_C = mix_C(cat[P3, up P4, up P5])            Completion expert, grid 32 (mid / deep stages)
    mix = depthwise 3x3 (24 ch) -> conv 1x1 24->8 -> GroupNorm(4) -> GELU

At each target stage s (dec3 / dec4 / dec5, grids 32 / 64 / 128), with D_s the decoder feature after the GAB skip and
q_s = sigmoid(deep-supervision logit of stage s) (detached, the stage's own lesion estimate):

    g_s = softmax(router_s(cat[D_s, q_s, avgpool(q_s, ~G/10), avgpool(q_s, ~G/3), mean(q_s)]))   over {D, C, null}
    f_s = g_D * H_D,s(F_D at grid G) + g_C * H_C,s(F_C at grid G)         added to D_s

The q channels tell the router about the lesion state around a location (inside a confident region, near its edge,
how much of the image is lesion), which a 1x1 conv on the local feature alone cannot know. 'fixed' mode uses
g = (0.5, 0.5, 0) everywhere (experts without routing). The roles 'detail' / 'completion' are the intended ones
(source stages, optional supervision); whether the experts really specialize is measured, not assumed.

Auxiliary heads (present in every mode, so the arms have the same parameters): detail -> 2 channels at grid 128
(segmentation, contour ring), completion -> 1 channel at grid 32 (segmentation). 'probe': they read detached expert
features (a readout of what each expert encodes; no effect on the network). 'supervise': they read experts computed
from detached encoder features, so their losses train the experts and the projections but never the encoder
(contour_losses.WithExpertAux).

Initialization (set_init, AFTER the parent's apply(_init_weights)): the heads H are zero, so the network starts as the
original model bit for bit; the router weights are zero and its biases (0, 0, -2) give g = (0.47, 0.47, 0.06), close to
the fixed (0.5, 0.5, 0) and with the null option low so the router does not start by switching the fusion off.

Readout switches (analysis/exp15_mechanism.py): force_gates = None | (gD, gC, gnull); stage_off = set of stages.
"""
import torch
from torch import nn
import torch.nn.functional as F

LSA_MODES = ('none', 'fixed', 'route')
LSA_AUX = ('probe', 'supervise')
_ROUTER_ID = {'fixed': 0, 'route': 1}
_AUX_ID = {'probe': 0, 'supervise': 1}
LSA_STAGES = ('dec3', 'dec4', 'dec5')
_STAGE_C_IDX = {'dec3': 2, 'dec4': 1, 'dec5': 0}            # channels of the decoder feature = c_list[idx]
FIXED_GATES = (0.5, 0.5, 0.0)
ROUTER_BIAS = (0.0, 0.0, -2.0)


def _odd(x):
    k = max(1, int(round(x)))
    return k if k % 2 else k + 1


def _resize(x, size):
    return x if x.shape[2:4] == tuple(size) else F.interpolate(x, size=size, mode='bilinear', align_corners=True)


def lesion_state(q):
    '''(B,1,G,G) probability -> (B,4,G,G): q, its local means over ~G/10 and ~G/3 windows, and its image mean.'''
    g = q.shape[-1]
    k1, k2 = _odd(g / 10), _odd(g / 3)
    a = F.avg_pool2d(q, k1, 1, k1 // 2, count_include_pad=False)
    b = F.avg_pool2d(q, k2, 1, k2 // 2, count_include_pad=False)
    return torch.cat([q, a, b, q.mean((2, 3), keepdim=True).expand_as(q)], 1)


class LesionStateFusion(nn.Module):
    def __init__(self, c_list, fdim=8, mode='route', aux='probe'):
        super().__init__()
        if mode not in LSA_MODES or mode == 'none':
            raise ValueError(f'lsa mode must be one of {LSA_MODES[1:]}, got {mode!r}')
        if aux not in LSA_AUX:
            raise ValueError(f'lsa aux must be one of {LSA_AUX}, got {aux!r}')
        self.mode, self.aux_mode, self.fdim = mode, aux, fdim
        self.stages = LSA_STAGES
        self.force_gates = None
        self.stage_off = set()
        c3 = 3 * fdim

        def mix():
            return nn.Sequential(nn.Conv2d(c3, c3, 3, padding=1, groups=c3), nn.Conv2d(c3, fdim, 1),
                                 nn.GroupNorm(4, fdim), nn.GELU())
        # registration order (mix_d, mix_c, head_d, head_c, router, aux_d, aux_c) fixes the RNG draws and the keys
        self.mix_d = mix()
        self.mix_c = mix()
        self.head_d = nn.ModuleDict({s: nn.Conv2d(fdim, c_list[_STAGE_C_IDX[s]], 1) for s in self.stages})
        self.head_c = nn.ModuleDict({s: nn.Conv2d(fdim, c_list[_STAGE_C_IDX[s]], 1) for s in self.stages})
        self.router = nn.ModuleDict({s: nn.Conv2d(c_list[_STAGE_C_IDX[s]] + 4, 3, 1) for s in self.stages})
        self.aux_d = nn.Conv2d(fdim, 2, 1)
        self.aux_c = nn.Conv2d(fdim, 1, 1)
        self.register_buffer('router_id', torch.tensor(_ROUTER_ID[mode], dtype=torch.long))
        self.register_buffer('aux_id', torch.tensor(_AUX_ID[aux], dtype=torch.long))

    def set_init(self):
        """Heads at zero (start = original model), router weights zero with biases (0, 0, -2). Call AFTER apply()."""
        for s in self.stages:
            for h in (self.head_d[s], self.head_c[s]):
                nn.init.zeros_(h.weight)
                nn.init.zeros_(h.bias)
            nn.init.zeros_(self.router[s].weight)
            with torch.no_grad():
                self.router[s].bias.copy_(torch.tensor(ROUTER_BIAS))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        for k, v in (('router_id', int(self.router_id)), ('aux_id', int(self.aux_id))):
            if prefix + k in state_dict and int(state_dict[prefix + k]) != v:
                raise RuntimeError(f'{prefix}{k}: the checkpoint was trained with {int(state_dict[prefix + k])}, this model is '
                                   f'built with {v} (router {_ROUTER_ID}, aux {_AUX_ID}); build it with the checkpoint\'s '
                                   f'--lsa / --lsa-aux')
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def experts(self, p):
        """p = the fusion's projections P1..P5 at their own grids -> (F_D at P1's grid, F_C at P3's grid)."""
        g1, g3 = p[0].shape[2:4], p[2].shape[2:4]
        fd = self.mix_d(torch.cat([p[0], _resize(p[1], g1), _resize(p[2], g1)], 1))
        fc = self.mix_c(torch.cat([p[2], _resize(p[3], g3), _resize(p[4], g3)], 1))
        return fd, fc

    def aux_logits(self, fd, fc):
        """Detail (2 ch: segmentation, contour ring) at F_D's grid and completion (1 ch) at F_C's grid."""
        return self.aux_d(fd), self.aux_c(fc)

    def gates(self, s, d, q):
        if self.force_gates is not None:
            g = torch.tensor(self.force_gates, dtype=d.dtype, device=d.device).view(1, 3, 1, 1)
            return g.expand(d.shape[0], 3, *d.shape[2:4])
        if self.mode == 'fixed':
            g = torch.tensor(FIXED_GATES, dtype=d.dtype, device=d.device).view(1, 3, 1, 1)
            return g.expand(d.shape[0], 3, *d.shape[2:4])
        return torch.softmax(self.router[s](torch.cat([d, lesion_state(q)], 1)), 1)

    def stage(self, s, fd, fc, d, q):
        """Returns (feature to add to d, gates (B,3,G,G))."""
        size = d.shape[2:4]
        g = self.gates(s, d, q)
        if s in self.stage_off:
            return torch.zeros_like(d), g
        f = g[:, 0:1] * self.head_d[s](_resize(fd, size)) + g[:, 1:2] * self.head_c[s](_resize(fc, size))
        return f, g
