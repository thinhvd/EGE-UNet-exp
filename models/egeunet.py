import torch
from torch import nn
import torch.nn.functional as F

from torch.nn.init import trunc_normal_
import math

from models.fusion import CrossStageFusion, DeepSupervisionOutputs, FUSION_MODES, FUSION_STAGE_SETS


class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size=3, padding=1, stride=1, dilation=1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(dim_in, dim_in, kernel_size=kernel_size, padding=padding, 
                      stride=stride, dilation=dilation, groups=dim_in)
        self.norm_layer = nn.GroupNorm(4, dim_in)
        self.conv2 = nn.Conv2d(dim_in, dim_out, kernel_size=1)

    def forward(self, x):
        return self.conv2(self.norm_layer(self.conv1(x)))


class LayerNorm(nn.Module):
    r""" From ConvNeXt (https://arxiv.org/pdf/2201.03545.pdf)
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
    

class group_aggregation_bridge(nn.Module):
    def __init__(self, dim_xh, dim_xl, k_size=3, d_list=[1,2,5,7]):
        super().__init__()
        self.pre_project = nn.Conv2d(dim_xh, dim_xl, 1)
        group_size = dim_xl // 2
        self.g0 = nn.Sequential(
            LayerNorm(normalized_shape=group_size+1, data_format='channels_first'),
            nn.Conv2d(group_size + 1, group_size + 1, kernel_size=3, stride=1, 
                      padding=(k_size+(k_size-1)*(d_list[0]-1))//2, 
                      dilation=d_list[0], groups=group_size + 1)
        )
        self.g1 = nn.Sequential(
            LayerNorm(normalized_shape=group_size+1, data_format='channels_first'),
            nn.Conv2d(group_size + 1, group_size + 1, kernel_size=3, stride=1, 
                      padding=(k_size+(k_size-1)*(d_list[1]-1))//2, 
                      dilation=d_list[1], groups=group_size + 1)
        )
        self.g2 = nn.Sequential(
            LayerNorm(normalized_shape=group_size+1, data_format='channels_first'),
            nn.Conv2d(group_size + 1, group_size + 1, kernel_size=3, stride=1, 
                      padding=(k_size+(k_size-1)*(d_list[2]-1))//2, 
                      dilation=d_list[2], groups=group_size + 1)
        )
        self.g3 = nn.Sequential(
            LayerNorm(normalized_shape=group_size+1, data_format='channels_first'),
            nn.Conv2d(group_size + 1, group_size + 1, kernel_size=3, stride=1, 
                      padding=(k_size+(k_size-1)*(d_list[3]-1))//2, 
                      dilation=d_list[3], groups=group_size + 1)
        )
        self.tail_conv = nn.Sequential(
            LayerNorm(normalized_shape=dim_xl * 2 + 4, data_format='channels_first'),
            nn.Conv2d(dim_xl * 2 + 4, dim_xl, 1)
        )
    def forward(self, xh, xl, mask):
        xh = self.pre_project(xh)
        xh = F.interpolate(xh, size=[xl.size(2), xl.size(3)], mode ='bilinear', align_corners=True)
        xh = torch.chunk(xh, 4, dim=1)
        xl = torch.chunk(xl, 4, dim=1)
        x0 = self.g0(torch.cat((xh[0], xl[0], mask), dim=1))
        x1 = self.g1(torch.cat((xh[1], xl[1], mask), dim=1))
        x2 = self.g2(torch.cat((xh[2], xl[2], mask), dim=1))
        x3 = self.g3(torch.cat((xh[3], xl[3], mask), dim=1))
        x = torch.cat((x0,x1,x2,x3), dim=1)
        x = self.tail_conv(x)
        return x


HPA_MODES = ('learnable', 'frozen_ones', 'none')

# Where the six GHPA modules sit (3 encoder + 3 decoder stages), grouped by operating resolution
# with a 256x256 input. 'low' is the original EGE-UNet. enc1 is not eligible (input has 3 channels,
# GHPA needs dim_in divisible by 4).
GHPA_PLACEMENTS = {
    'low':  ['enc4', 'enc5', 'enc6', 'dec1', 'dec2', 'dec3'],   # res {32,16,8 | 8,8,16}  (original)
    'mid':  ['enc3', 'enc4', 'enc5', 'dec2', 'dec3', 'dec4'],   # res {64,32,16 | 8,16,32}
    'high': ['enc2', 'enc3', 'enc4', 'dec3', 'dec4', 'dec5'],   # res {128,64,32 | 16,32,64}
}
_GHPA_ALLOWED_STAGES = ('enc2', 'enc3', 'enc4', 'enc5', 'enc6', 'dec1', 'dec2', 'dec3', 'dec4', 'dec5')


class Grouped_multi_axis_Hadamard_Product_Attention(nn.Module):
    '''
    hpa_mode (ablation of the static learned prior P, i.e. params_xy/zx/zy):
      'learnable'   - original model.
      'frozen_ones' - P stays at its init value (ones) and is not trained; conv_xy/zx/zy are still
                      trained. Note: a conv of a constant map is a per-channel constant in the interior
                      plus a 1-px border band from zero padding, so this variant keeps a per-channel
                      scaling but has no learned spatial prior.
      'none'        - no Hadamard gating at all: groups 1-3 pass through untouched and the P/conv_*
                      modules are not created (fewer parameters; checkpoints are not interchangeable
                      with the other modes).
    '''
    def __init__(self, dim_in, dim_out, x=8, y=8, hpa_mode='learnable'):
        super().__init__()
        if hpa_mode not in HPA_MODES:
            raise ValueError(f'hpa_mode must be one of {HPA_MODES}, got {hpa_mode!r}')
        self.hpa_mode = hpa_mode

        c_dim_in = dim_in//4
        k_size=3
        pad=(k_size-1) // 2

        if hpa_mode != 'none':
            learn_p = (hpa_mode == 'learnable')
            self.params_xy = nn.Parameter(torch.Tensor(1, c_dim_in, x, y), requires_grad=learn_p)
            nn.init.ones_(self.params_xy)
            self.conv_xy = nn.Sequential(nn.Conv2d(c_dim_in, c_dim_in, kernel_size=k_size, padding=pad, groups=c_dim_in), nn.GELU(), nn.Conv2d(c_dim_in, c_dim_in, 1))

            self.params_zx = nn.Parameter(torch.Tensor(1, 1, c_dim_in, x), requires_grad=learn_p)
            nn.init.ones_(self.params_zx)
            self.conv_zx = nn.Sequential(nn.Conv1d(c_dim_in, c_dim_in, kernel_size=k_size, padding=pad, groups=c_dim_in), nn.GELU(), nn.Conv1d(c_dim_in, c_dim_in, 1))

            self.params_zy = nn.Parameter(torch.Tensor(1, 1, c_dim_in, y), requires_grad=learn_p)
            nn.init.ones_(self.params_zy)
            self.conv_zy = nn.Sequential(nn.Conv1d(c_dim_in, c_dim_in, kernel_size=k_size, padding=pad, groups=c_dim_in), nn.GELU(), nn.Conv1d(c_dim_in, c_dim_in, 1))

        self.dw = nn.Sequential(
                nn.Conv2d(c_dim_in, c_dim_in, 1),
                nn.GELU(),
                nn.Conv2d(c_dim_in, c_dim_in, kernel_size=3, padding=1, groups=c_dim_in)
        )
        
        self.norm1 = LayerNorm(dim_in, eps=1e-6, data_format='channels_first')
        self.norm2 = LayerNorm(dim_in, eps=1e-6, data_format='channels_first')
        
        self.ldw = nn.Sequential(
                nn.Conv2d(dim_in, dim_in, kernel_size=3, padding=1, groups=dim_in),
                nn.GELU(),
                nn.Conv2d(dim_in, dim_out, 1),
        )
        
    def forward(self, x):
        x = self.norm1(x)
        x1, x2, x3, x4 = torch.chunk(x, 4, dim=1)
        B, C, H, W = x1.size()
        if self.hpa_mode != 'none':
            #----------xy----------#
            params_xy = self.params_xy
            x1 = x1 * self.conv_xy(F.interpolate(params_xy, size=x1.shape[2:4],mode='bilinear', align_corners=True))
            #----------zx----------#
            x2 = x2.permute(0, 3, 1, 2)
            params_zx = self.params_zx
            x2 = x2 * self.conv_zx(F.interpolate(params_zx, size=x2.shape[2:4],mode='bilinear', align_corners=True).squeeze(0)).unsqueeze(0)
            x2 = x2.permute(0, 2, 3, 1)
            #----------zy----------#
            x3 = x3.permute(0, 2, 1, 3)
            params_zy = self.params_zy
            x3 = x3 * self.conv_zy(F.interpolate(params_zy, size=x3.shape[2:4],mode='bilinear', align_corners=True).squeeze(0)).unsqueeze(0)
            x3 = x3.permute(0, 2, 1, 3)
        #----------dw----------#
        x4 = self.dw(x4)
        #----------concat----------#
        x = torch.cat([x1,x2,x3,x4],dim=1)
        #----------ldw----------#
        x = self.norm2(x)
        x = self.ldw(x)
        return x



    
    

class EGEUNet(nn.Module):
    
    def __init__(self, num_classes=1, input_channels=3, c_list=[8,16,24,32,48,64], bridge=True, gt_ds=True,
                 hpa_mode='learnable', ghpa_stages=None,
                 fusion_mode='none', fusion_stages=None, fusion_dim=16):
        super().__init__()

        self.bridge = bridge
        self.gt_ds = gt_ds
        self.hpa_mode = hpa_mode
        if ghpa_stages is None:
            ghpa_stages = GHPA_PLACEMENTS['low']
        ghpa_stages = list(ghpa_stages)
        for s in ghpa_stages:
            if s not in _GHPA_ALLOWED_STAGES:
                raise ValueError(f'invalid GHPA stage {s!r}; allowed: {_GHPA_ALLOWED_STAGES}')
        self.ghpa_stages = ghpa_stages

        # channel dims per stage; a stage is either the original plain Conv2d 3x3 or a GHPA block.
        # Construction ORDER must stay identical to the original so the default placement is
        # bit-identical (module __init__ consumes RNG).
        enc_dims = [(input_channels, c_list[0]), (c_list[0], c_list[1]), (c_list[1], c_list[2]),
                    (c_list[2], c_list[3]), (c_list[3], c_list[4]), (c_list[4], c_list[5])]
        dec_dims = [(c_list[5], c_list[4]), (c_list[4], c_list[3]), (c_list[3], c_list[2]),
                    (c_list[2], c_list[1]), (c_list[1], c_list[0])]

        def _stage(tag, dim_in, dim_out):
            if tag in ghpa_stages:
                if dim_in % 4 != 0:
                    raise ValueError(f'GHPA at {tag} needs dim_in divisible by 4, got {dim_in}')
                return nn.Sequential(
                    Grouped_multi_axis_Hadamard_Product_Attention(dim_in, dim_out, hpa_mode=hpa_mode),
                )
            return nn.Sequential(
                nn.Conv2d(dim_in, dim_out, 3, stride=1, padding=1),
            )

        self.encoder1 = _stage('enc1', *enc_dims[0])
        self.encoder2 = _stage('enc2', *enc_dims[1])
        self.encoder3 = _stage('enc3', *enc_dims[2])
        self.encoder4 = _stage('enc4', *enc_dims[3])
        self.encoder5 = _stage('enc5', *enc_dims[4])
        self.encoder6 = _stage('enc6', *enc_dims[5])
        if hpa_mode != 'learnable':
            print(f'GHPA hpa_mode = {hpa_mode}')
        if ghpa_stages != GHPA_PLACEMENTS['low']:
            print(f'GHPA placement = {ghpa_stages}')

        if bridge: 
            self.GAB1 = group_aggregation_bridge(c_list[1], c_list[0])
            self.GAB2 = group_aggregation_bridge(c_list[2], c_list[1])
            self.GAB3 = group_aggregation_bridge(c_list[3], c_list[2])
            self.GAB4 = group_aggregation_bridge(c_list[4], c_list[3])
            self.GAB5 = group_aggregation_bridge(c_list[5], c_list[4])
            print('group_aggregation_bridge was used')
        if gt_ds:
            self.gt_conv1 = nn.Sequential(nn.Conv2d(c_list[4], 1, 1))
            self.gt_conv2 = nn.Sequential(nn.Conv2d(c_list[3], 1, 1))
            self.gt_conv3 = nn.Sequential(nn.Conv2d(c_list[2], 1, 1))
            self.gt_conv4 = nn.Sequential(nn.Conv2d(c_list[1], 1, 1))
            self.gt_conv5 = nn.Sequential(nn.Conv2d(c_list[0], 1, 1))
            print('gt deep supervision was used')
        
        self.decoder1 = _stage('dec1', *dec_dims[0])
        self.decoder2 = _stage('dec2', *dec_dims[1])
        self.decoder3 = _stage('dec3', *dec_dims[2])
        self.decoder4 = _stage('dec4', *dec_dims[3])
        self.decoder5 = _stage('dec5', *dec_dims[4])
        self.ebn1 = nn.GroupNorm(4, c_list[0])
        self.ebn2 = nn.GroupNorm(4, c_list[1])
        self.ebn3 = nn.GroupNorm(4, c_list[2])
        self.ebn4 = nn.GroupNorm(4, c_list[3])
        self.ebn5 = nn.GroupNorm(4, c_list[4])
        self.dbn1 = nn.GroupNorm(4, c_list[4])
        self.dbn2 = nn.GroupNorm(4, c_list[3])
        self.dbn3 = nn.GroupNorm(4, c_list[2])
        self.dbn4 = nn.GroupNorm(4, c_list[1])
        self.dbn5 = nn.GroupNorm(4, c_list[0])

        self.final = nn.Conv2d(c_list[0], num_classes, kernel_size=1)

        if fusion_mode not in FUSION_MODES:
            raise ValueError(f'fusion_mode must be one of {FUSION_MODES}, got {fusion_mode!r}')
        self.fusion_mode = fusion_mode
        self.fusion_stages = None
        self.boundary_guided = False

        self.apply(self._init_weights)

        # EXP-4 cross-stage fusion (models/fusion.py). Built AFTER apply() on purpose: everything
        # above then consumes the RNG exactly as in the original code — both in the constructors
        # and in apply()'s re-draws — so the default path stays bit-identical whatever this block
        # does. The fusion gets the same initialization by hand, and its heads are then zeroed so
        # the variant starts from the baseline function itself, not merely near it.
        if fusion_mode != 'none':
            if fusion_stages is None:
                fusion_stages = FUSION_STAGE_SETS['deep3']
            self.fusion = CrossStageFusion(c_list, fusion_mode, fusion_stages, fdim=fusion_dim)
            self.fusion.apply(self._init_weights)
            self.fusion.zero_init_heads()
            self.fusion_stages = self.fusion.target_stages
            # EXP-9: the boundary logits travel with the deep-supervision outputs (see forward)
            self.boundary_guided = self.fusion.boundary_guided
            if self.boundary_guided and not gt_ds:
                raise ValueError('boundary-guided fusion returns its boundary maps with the deep-supervision '
                                 'outputs, so it needs gt_ds=True')
            print(f'cross-stage fusion = {fusion_mode} on {self.fusion_stages} (dim {fusion_dim})')

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        
        out = F.gelu(F.max_pool2d(self.ebn1(self.encoder1(x)),2,2))
        t1 = out # b, c0, H/2, W/2

        out = F.gelu(F.max_pool2d(self.ebn2(self.encoder2(out)),2,2))
        t2 = out # b, c1, H/4, W/4 

        out = F.gelu(F.max_pool2d(self.ebn3(self.encoder3(out)),2,2))
        t3 = out # b, c2, H/8, W/8
        
        out = F.gelu(F.max_pool2d(self.ebn4(self.encoder4(out)),2,2))
        t4 = out # b, c3, H/16, W/16
        
        out = F.gelu(F.max_pool2d(self.ebn5(self.encoder5(out)),2,2))
        t5 = out # b, c4, H/32, W/32
        
        out = F.gelu(self.encoder6(out)) # b, c5, H/32, W/32
        t6 = out

        # EXP-4: computed here because the GAB bridges below overwrite t1..t5 in place. The fused
        # features are added AFTER each stage's gt_pre is taken, so deep supervision keeps reading
        # exactly what it read in the baseline.
        # EXP-9 (boundary-guided): only the projections are computed here; each target stage then
        # reads its boundary map from its decoder feature after the GAB add, fuses and adds.
        fuse, p, bnd = {}, None, {}
        if self.fusion_mode != 'none':
            if self.boundary_guided:
                p = self.fusion.project((t1, t2, t3, t4, t5))
            else:
                fuse = self.fusion((t1, t2, t3, t4, t5))

        out5 = F.gelu(self.dbn1(self.decoder1(out))) # b, c4, H/32, W/32
        if self.gt_ds: 
            gt_pre5 = self.gt_conv1(out5)
            t5 = self.GAB5(t6, t5, gt_pre5)
            gt_pre5 = F.interpolate(gt_pre5, scale_factor=32, mode ='bilinear', align_corners=True)
        else: t5 = self.GAB5(t6, t5)
        out5 = torch.add(out5, t5) # b, c4, H/32, W/32
        if 'dec1' in fuse: out5 = out5 + fuse['dec1']
        elif p is not None and 'dec1' in self.fusion_stages:
            f, bnd['dec1'] = self.fusion.guided_stage('dec1', p, out5)
            out5 = out5 + f
        
        out4 = F.gelu(F.interpolate(self.dbn2(self.decoder2(out5)),scale_factor=(2,2),mode ='bilinear',align_corners=True)) # b, c3, H/16, W/16
        if self.gt_ds: 
            gt_pre4 = self.gt_conv2(out4)
            t4 = self.GAB4(t5, t4, gt_pre4)
            gt_pre4 = F.interpolate(gt_pre4, scale_factor=16, mode ='bilinear', align_corners=True)
        else:t4 = self.GAB4(t5, t4)
        out4 = torch.add(out4, t4) # b, c3, H/16, W/16
        if 'dec2' in fuse: out4 = out4 + fuse['dec2']
        elif p is not None and 'dec2' in self.fusion_stages:
            f, bnd['dec2'] = self.fusion.guided_stage('dec2', p, out4)
            out4 = out4 + f
        
        out3 = F.gelu(F.interpolate(self.dbn3(self.decoder3(out4)),scale_factor=(2,2),mode ='bilinear',align_corners=True)) # b, c2, H/8, W/8
        if self.gt_ds: 
            gt_pre3 = self.gt_conv3(out3)
            t3 = self.GAB3(t4, t3, gt_pre3)
            gt_pre3 = F.interpolate(gt_pre3, scale_factor=8, mode ='bilinear', align_corners=True)
        else: t3 = self.GAB3(t4, t3)
        out3 = torch.add(out3, t3) # b, c2, H/8, W/8
        if 'dec3' in fuse: out3 = out3 + fuse['dec3']
        elif p is not None and 'dec3' in self.fusion_stages:
            f, bnd['dec3'] = self.fusion.guided_stage('dec3', p, out3)
            out3 = out3 + f
        
        out2 = F.gelu(F.interpolate(self.dbn4(self.decoder4(out3)),scale_factor=(2,2),mode ='bilinear',align_corners=True)) # b, c1, H/4, W/4
        if self.gt_ds: 
            gt_pre2 = self.gt_conv4(out2)
            t2 = self.GAB2(t3, t2, gt_pre2)
            gt_pre2 = F.interpolate(gt_pre2, scale_factor=4, mode ='bilinear', align_corners=True)
        else: t2 = self.GAB2(t3, t2)
        out2 = torch.add(out2, t2) # b, c1, H/4, W/4
        if 'dec4' in fuse: out2 = out2 + fuse['dec4']
        elif p is not None and 'dec4' in self.fusion_stages:
            f, bnd['dec4'] = self.fusion.guided_stage('dec4', p, out2)
            out2 = out2 + f
        
        out1 = F.gelu(F.interpolate(self.dbn5(self.decoder5(out2)),scale_factor=(2,2),mode ='bilinear',align_corners=True)) # b, c0, H/2, W/2
        if self.gt_ds: 
            gt_pre1 = self.gt_conv5(out1)
            t1 = self.GAB1(t2, t1, gt_pre1)
            gt_pre1 = F.interpolate(gt_pre1, scale_factor=2, mode ='bilinear', align_corners=True)
        else: t1 = self.GAB1(t2, t1)
        out1 = torch.add(out1, t1) # b, c0, H/2, W/2
        if 'dec5' in fuse: out1 = out1 + fuse['dec5']
        elif p is not None and 'dec5' in self.fusion_stages:
            f, bnd['dec5'] = self.fusion.guided_stage('dec5', p, out1)
            out1 = out1 + f
        
        out0 = F.interpolate(self.final(out1),scale_factor=(2,2),mode ='bilinear',align_corners=True) # b, num_class, H, W
        
        if self.gt_ds:
            gt_pres = (torch.sigmoid(gt_pre5), torch.sigmoid(gt_pre4), torch.sigmoid(gt_pre3), torch.sigmoid(gt_pre2), torch.sigmoid(gt_pre1))
            if bnd:   # EXP-9 only; every other configuration returns the plain tuple as before
                gt_pres = DeepSupervisionOutputs(gt_pres, boundary=bnd)
            return gt_pres, torch.sigmoid(out0)
        else:
            return torch.sigmoid(out0)