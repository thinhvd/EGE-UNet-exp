"""EXP-4: cross-stage fusion for the EGE-UNet decoder (CSAA-inspired).

Motivation. EXP-1..3 established that the GHPA gate helps large lesions and does not help small
ones, and that this tilt is not fixable by thresholding or by moving the gate to another resolution
band. One remaining explanation is that each decoder stage only ever sees the single encoder stage
its skip connection carries, so fine-scale evidence never reaches the deep stages where the gate
operates. EFCNet's CSAA module attacks exactly that by letting every decoder stage read a fused
view of ALL encoder stages.

This module implements the fusion variants on one code path, so they differ only in the ingredient
under test. Each mode is a choice of how the five maps are combined (sum or concat) and whether the
cross-stage attention runs first:

    'sum'      - resize the 5 projected encoder maps to the target grid and take a learned weighted
                 sum. Control: it adds an extra path into the decoder with almost no capacity and no
                 cross-scale mixing beyond a per-source scalar.
    'concat'   - concatenate the 5 resized maps and mix them with a 1x1 conv. Control: full linear
                 cross-scale mixing, still no attention.
    'csaa'     - 'concat' plus a two-step axial attention over the stacked stages (EFCNet's CSAA,
                 shrunk to a small common channel width).
    'sum_attn' - 'sum' plus the same attention. With 'sum' the combine step is a fixed scalar per
                 source, so the attention is the only input-dependent interaction between stages;
                 'sum' vs 'sum_attn' isolates what the attention contributes more sharply than
                 'concat' vs 'csaa', where the 1x1 conv already mixes the stages.

The attention is applied as an additive delta on the projections, so each attention variant differs
from its control by exactly the attention term (and 1,632 parameters) and by nothing else.

Init-equivalence. Every per-stage head is zero-initialized (weight and bias), so at initialization
the fusion contributes exactly 0 and the whole network is bitwise identical to the baseline for the
same seed. This keeps the comparison honest: the variants start from the same function, not merely
from a similar one. After the first optimizer step the heads leave zero and gradients reach the
projections and the attention.
"""
import torch
from torch import nn
import torch.nn.functional as F


FUSION_MODES = ('none', 'sum', 'concat', 'csaa', 'sum_attn')
_SUM_COMBINE = ('sum', 'sum_attn')
_WITH_ATTENTION = ('csaa', 'sum_attn')

# Which decoder stages receive the fused feature. dec1..dec5 are the decoder outputs out5..out1,
# i.e. dec1 is the deepest (8x8 with a 256x256 input) and dec5 the shallowest (128x128).
FUSION_STAGE_SETS = {
    'deep3': ['dec1', 'dec2', 'dec3'],          # grids 8 / 16 / 32
    'all5': ['dec1', 'dec2', 'dec3', 'dec4', 'dec5'],
}
_FUSION_ALLOWED_STAGES = ('dec1', 'dec2', 'dec3', 'dec4', 'dec5')

# decoder stage -> index of the encoder skip feature whose channel count and grid it matches.
# t1..t5 have channels c_list[0..4] at H/2, H/4, H/8, H/16, H/32.
_STAGE_SOURCE_IDX = {'dec1': 4, 'dec2': 3, 'dec3': 2, 'dec4': 1, 'dec5': 0}


class CrossStageAxialAttention(nn.Module):
    """Two-step (width then height) axial attention across the stacked encoder stages.

    Input and output are (B, S, C, G, G) with S stages on a common GxG grid. In each step a token
    is one axial slice of one stage (dim C*G); queries come from a single stage while keys and
    values are the concatenation of all stages, which is what makes the attention cross-stage.

    forward() returns the DELTA (attended minus input) so the caller can add it back at each
    source's own resolution, keeping this variant a pure addition on top of 'concat'.
    """

    def __init__(self, fdim=16, grid=16):
        super().__init__()
        self.fdim = fdim
        self.grid = grid
        self.scale = (fdim * grid) ** -0.5
        self.q_w = nn.Conv2d(fdim, fdim, 1)
        self.k_w = nn.Conv2d(fdim, fdim, 1)
        self.v_w = nn.Conv2d(fdim, fdim, 1)
        self.q_h = nn.Conv2d(fdim, fdim, 1)
        self.k_h = nn.Conv2d(fdim, fdim, 1)
        self.v_h = nn.Conv2d(fdim, fdim, 1)

    def _axial(self, x, q_conv, k_conv, v_conv, axis):
        # x: (B, S, C, H, W); axis 'w' -> tokens are columns, 'h' -> tokens are rows
        B, S, C, H, W = x.shape
        flat = x.reshape(B * S, C, H, W)
        q, k, v = q_conv(flat), k_conv(flat), v_conv(flat)
        if axis == 'w':
            n_tok, dim = W, C * H
            # (B*S, C, H, W) -> (B, S, W, C*H)
            def pack(t):
                return t.permute(0, 3, 1, 2).reshape(B, S, W, C * H)
            def unpack(t):
                return t.reshape(B, S, W, C, H).permute(0, 1, 3, 4, 2)
        else:
            n_tok, dim = H, C * W
            # (B*S, C, H, W) -> (B, S, H, C*W)
            def pack(t):
                return t.permute(0, 2, 1, 3).reshape(B, S, H, C * W)
            def unpack(t):
                return t.reshape(B, S, H, C, W).permute(0, 1, 3, 2, 4)

        q, k, v = pack(q), pack(k), pack(v)                      # (B, S, n_tok, dim)
        k = k.reshape(B, S * n_tok, dim)                         # keys/values pool every stage
        v = v.reshape(B, S * n_tok, dim)
        att = torch.einsum('bstd,bnd->bstn', q, k) * self.scale
        att = att.softmax(dim=-1)
        out = torch.einsum('bstn,bnd->bstd', att, v)             # (B, S, n_tok, dim)
        return unpack(out)

    def forward(self, x):
        y = x + self._axial(x, self.q_w, self.k_w, self.v_w, 'w')
        y = y + self._axial(y, self.q_h, self.k_h, self.v_h, 'h')
        return y - x


class CrossStageFusion(nn.Module):
    """Fuse all five encoder skip features and inject the result into chosen decoder stages.

    Args:
        c_list: the network's channel list; c_list[0..4] are the channels of t1..t5.
        mode: one of FUSION_MODES ('none' means this module should not be built at all).
        target_stages: decoder stages to feed, e.g. FUSION_STAGE_SETS['deep3'].
        fdim: common channel width the sources are projected to (must be divisible by 4 for
            GroupNorm(4)). This is the reduction ratio under test in the attention variant.
        common_grid: spatial size the attention runs on (attention variants only).
    """

    def __init__(self, c_list, mode, target_stages, fdim=16, common_grid=16):
        super().__init__()
        if mode not in FUSION_MODES or mode == 'none':
            raise ValueError(f'fusion mode must be one of {FUSION_MODES[1:]}, got {mode!r}')
        if fdim % 4 != 0:
            raise ValueError(f'fusion_dim must be divisible by 4 (GroupNorm(4)), got {fdim}')
        target_stages = list(target_stages)
        for s in target_stages:
            if s not in _FUSION_ALLOWED_STAGES:
                raise ValueError(f'invalid fusion stage {s!r}; allowed: {_FUSION_ALLOWED_STAGES}')
        if not target_stages:
            raise ValueError('fusion needs at least one target stage')

        self.mode = mode
        self.combine = 'sum' if mode in _SUM_COMBINE else 'concat'
        self.use_attn = mode in _WITH_ATTENTION
        self.fdim = fdim
        self.common_grid = common_grid
        self.target_stages = target_stages
        self.n_src = 5

        # registration order (proj, attn, sum_w, heads) fixes both the RNG draws and the state_dict
        # keys; keep it, or checkpoints trained before 'sum_attn' existed stop meaning the same thing
        self.proj = nn.ModuleList([
            nn.Sequential(nn.Conv2d(c_list[i], fdim, 1), nn.GroupNorm(4, fdim), nn.GELU())
            for i in range(self.n_src)
        ])

        if self.use_attn:
            self.attn = CrossStageAxialAttention(fdim=fdim, grid=common_grid)

        if self.combine == 'sum':
            # one scalar per source per target stage; plain (no softmax) - the zero-init head
            # already guarantees init-equivalence, so these weights need no special handling.
            self.sum_w = nn.ParameterDict({
                s: nn.Parameter(torch.full((self.n_src,), 1.0 / self.n_src)) for s in target_stages
            })

        head_in = fdim if self.combine == 'sum' else self.n_src * fdim
        self.heads = nn.ModuleDict({
            s: nn.Conv2d(head_in, c_list[_STAGE_SOURCE_IDX[s]], 1) for s in target_stages
        })

    def zero_init_heads(self):
        """Zero every head so the fusion output starts at exactly 0.

        MUST be called AFTER the parent's ``self.apply(_init_weights)``, which would otherwise
        overwrite these weights with its normal_ initialization.
        """
        for head in self.heads.values():
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, feats):
        """feats: (t1, ..., t5) encoder skip features, captured BEFORE the GAB bridges rewrite them.

        Returns {stage: tensor} with one entry per target stage, each already shaped like that
        stage's decoder output.
        """
        p = [self.proj[i](feats[i]) for i in range(self.n_src)]

        if self.use_attn:
            g = self.common_grid
            stacked = torch.stack([
                F.interpolate(f, size=(g, g), mode='bilinear', align_corners=True) for f in p
            ], dim=1)
            delta = self.attn(stacked)
            p = [f + F.interpolate(delta[:, i], size=f.shape[2:4], mode='bilinear', align_corners=True)
                 for i, f in enumerate(p)]

        out = {}
        for s in self.target_stages:
            size = feats[_STAGE_SOURCE_IDX[s]].shape[2:4]
            resized = [f if f.shape[2:4] == size
                       else F.interpolate(f, size=size, mode='bilinear', align_corners=True)
                       for f in p]
            if self.combine == 'sum':
                w = self.sum_w[s]
                fused = sum(w[i] * resized[i] for i in range(self.n_src))
            else:
                fused = torch.cat(resized, dim=1)
            out[s] = self.heads[s](fused)
        return out
