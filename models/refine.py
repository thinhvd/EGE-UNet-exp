"""EXP-10: Dec5 boundary residual refinement (BRR) on top of the cross-stage fusion.

EXP-9 put boundary heads on dec3/dec4/dec5 and let their maps reweight the fusion sources; the heads
did find the contour, but the fusion branch reached only 1.6-3 % of the decoder feature at dec5, the
stage that produces the final mask, so the errors around the contour did not move. This module acts
where the mask is made instead: on the segmentation logit itself.

At dec5 (grid 128 for a 256 input), with D5 the decoder feature after the GAB skip and the fusion add:

    logit_B = bnd_head(D5)                               boundary head, 1x1 conv   (mode 'gate' only)
    dz      = delta(mix(cat[D5, P1, P2]))                 refinement residual, 1 channel
    z_final = z_base + sigmoid(logit_B) * dz              'gate'
    z_final = z_base + dz                                 'plain' (control: same residual, no gate)

P1 and P2 are the fusion's own projections of the two shallowest encoder stages (already 8 channels;
P2 is resized 64 -> 128), so no new projection is learned. mix = 1x1 conv (24 -> 8) + GroupNorm(4) +
GELU, delta = 3x3 conv (8 -> 1) zero-initialized: at init dz = 0 and the network computes exactly the
fusion-only model (which itself starts as the baseline). The boundary head is trained by the
contour-zone loss (contour_losses.boundary_zone) and, in 'gate', also through the product.

Two switches exist for inference-time readouts only (analysis/exp10_mechanism.py):
    enabled = False      -> z_final = z_base (the residual is dropped)
    gate_override        -> None (learned gate) | 'ones' (gate = 1 everywhere) | a (N,1,H,W) tensor
                            (e.g. the ground-truth zone); in 'plain' a tensor restricts dz to it.
"""
import torch
from torch import nn
import torch.nn.functional as F

REFINE_MODES = ('none', 'gate', 'plain')


class Dec5BoundaryResidual(nn.Module):
    def __init__(self, c_dec, fdim, mode, hidden=8):
        super().__init__()
        if mode not in REFINE_MODES or mode == 'none':
            raise ValueError(f'refine mode must be one of {REFINE_MODES[1:]}, got {mode!r}')
        self.mode = mode
        self.enabled = True
        self.gate_override = None
        # registration order (bnd_head, mix, delta) fixes the RNG draws and the state_dict keys
        if mode == 'gate':
            self.bnd_head = nn.Conv2d(c_dec, 1, 1)
        self.mix = nn.Sequential(nn.Conv2d(c_dec + 2 * fdim, hidden, 1), nn.GroupNorm(4, hidden), nn.GELU())
        self.delta = nn.Conv2d(hidden, 1, 3, padding=1)

    def zero_init_delta(self):
        """Zero the last conv so the residual starts at exactly 0. Call AFTER apply(_init_weights)."""
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def forward(self, d5, p1, p2):
        """Returns (residual to add to the logit, raw residual dz, boundary logit or None)."""
        size = d5.shape[2:4]
        p1 = p1 if p1.shape[2:4] == size else F.interpolate(p1, size=size, mode='bilinear', align_corners=True)
        p2 = p2 if p2.shape[2:4] == size else F.interpolate(p2, size=size, mode='bilinear', align_corners=True)
        dz = self.delta(self.mix(torch.cat([d5, p1, p2], dim=1)))
        logit = None
        if self.mode == 'gate':
            logit = self.bnd_head(d5)
            gate = torch.sigmoid(logit)
            if self.gate_override is not None:
                gate = torch.ones_like(gate) if isinstance(self.gate_override, str) else self.gate_override
            out = dz * gate
        else:
            out = dz if self.gate_override is None or isinstance(self.gate_override, str) else dz * self.gate_override
        return out, dz, logit
