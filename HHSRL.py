import torch
from torch import nn
import torch
import torchvision
from torch import nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from models.RALA import RALA_Block, RoPE
from until.utils import *
from einops import rearrange
import timm
from timm.models.layers import to_2tuple, trunc_normal_
import types
import math
from pdb import set_trace as st


def momentum_update(old_value, new_value, momentum, debug=False):
    update = momentum * old_value + (1 - momentum) * new_value
    if debug:
        print("old prot: {:.3f} x |{:.3f}|, new val: {:.3f} x |{:.3f}|, result= |{:.3f}|".format(
            momentum, torch.norm(old_value, p=2), (1 - momentum), torch.norm(new_value, p=2),
            torch.norm(update, p=2)))
    return update


def l2_normalize(x):
    return F.normalize(x, p=2, dim=-1)


class ProjectionHead(nn.Module):
    def __init__(self, dim_in, proj_dim=256):
        super(ProjectionHead, self).__init__()

        self.proj = self.mlp2 = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_in, proj_dim, 1))

    def forward(self, x):
        return l2_normalize(self.proj(x))
    

def _get_stats(x):
    # x: (P', C)
    mu = x.mean(dim=0, keepdim=True)  # (1, C)
    var = x.var(dim=0, unbiased=False, keepdim=True) + 1e-6
    return mu, var

def _frechet_diag(mu1, var1, mu2, var2):
    # 对角协方差的 Fréchet 距离
    return torch.sum((mu1 - mu2) ** 2) + \
           torch.sum(var1 + var2 - 2.0 * torch.sqrt(var1 * var2))

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        # self.norm = nn.LayerNorm(embed_dim)
        self.norm = nn.GroupNorm(num_groups=embed_dim // 16, num_channels=embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        # x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


class ConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvLayer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)

class D_ConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(D_ConvLayer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class HHSRL(nn.Module):
    def __init__(self, img_size=224, embed_dims=[256, 320, 512], num_heads=[5,8,4], drop_path_rate=0., norm_layer=None, depths=[2, 2, 2, 2], **kwargs):
        super().__init__()
        
        if norm_layer is None:
            norm_layer = lambda dim: nn.GroupNorm(32, dim) 

        kan_input_dim = embed_dims[0]
        self.size = 256

        self.encoder1 = ConvLayer(3, kan_input_dim//8)  
        self.encoder2 = ConvLayer(kan_input_dim//8, kan_input_dim//4)  
        self.encoder3 = ConvLayer(kan_input_dim//4, kan_input_dim)

        self.norm3 = norm_layer(embed_dims[1])
        self.norm4 = norm_layer(embed_dims[2])

        self.dnorm3 = norm_layer(embed_dims[1])
        self.dnorm4 = norm_layer(embed_dims[0])


        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0     # pointer
        
        self.RoPE1 = RoPE(embed_dim=embed_dims[1], num_heads=num_heads[0])
        self.block1 = nn.ModuleList([
            RALA_Block(
                embed_dim=embed_dims[1],
                num_heads=num_heads[0],
                ffn_dim=embed_dims[1] * 4,
                drop_path=dpr[cur + i],
                layerscale=True
            )
            for i in range(depths[0])
        ])
        cur += depths[0]
        
        self.RoPE2 = RoPE(embed_dim=embed_dims[2], num_heads=num_heads[1])
        self.block2 = nn.ModuleList([
            RALA_Block(
                embed_dim=embed_dims[2],
                num_heads=num_heads[1],
                ffn_dim=embed_dims[2] * 4,
                drop_path=dpr[cur + i],
                layerscale=True
            )
            for i in range(depths[1])
        ])
        cur += depths[1]
        
        self.RoPEd1 = RoPE(embed_dim=embed_dims[1], num_heads=num_heads[0])
        self.dblock1 = nn.ModuleList([
            RALA_Block(
                embed_dim=embed_dims[1],
                num_heads=num_heads[0],
                ffn_dim=embed_dims[1] * 4,
                drop_path=dpr[cur + i],
                layerscale=True
            )
            for i in range(depths[2])
        ])
        cur += depths[2]
        
        self.RoPEd2 = RoPE(embed_dim=embed_dims[0], num_heads=num_heads[2])
        self.dblock2 = nn.ModuleList([
            RALA_Block(
                embed_dim=embed_dims[0],
                num_heads=num_heads[2],
                ffn_dim=embed_dims[0] * 4,
                drop_path=dpr[cur + i],
                layerscale=True
            )
            for i in range(depths[3])
        ])

        self.patch_embed3 = PatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed4 = PatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])

        self.decoder1 = D_ConvLayer(embed_dims[2], embed_dims[1])  
        self.decoder2 = D_ConvLayer(embed_dims[1], embed_dims[0])  
        self.decoder3 = D_ConvLayer(embed_dims[0], embed_dims[0]//4) 
        self.decoder4 = D_ConvLayer(embed_dims[0]//4, embed_dims[0]//8)
        self.decoder5 = D_ConvLayer(embed_dims[0]//8, embed_dims[0]//8)

        ###########
        self.gamma = 0.99
        self.num_prototype = 1
        self.use_prototype = True
        self.update_prototype = True
        self.num_classes = 2
        
        self.prototypes = nn.ParameterList([
            nn.Parameter(torch.zeros(self.num_classes, self.num_prototype, embed_dims[0]//4)),
            nn.Parameter(torch.zeros(self.num_classes, self.num_prototype, embed_dims[0]//8)),
            nn.Parameter(torch.zeros(self.num_classes, self.num_prototype, embed_dims[0]//8)),
        ])
        
        self.feat_norms = nn.ModuleList([
                nn.LayerNorm(embed_dims[0] // 4),
                nn.LayerNorm(embed_dims[0] // 8),
                nn.LayerNorm(embed_dims[0] // 8),])
        
        self.scale_weight = nn.Parameter(torch.ones(3))
        ############
        
    def prototype_learning(self, _c, level, out_seg, gt_seg, valid_mask=None):
        """
        """
        pred_seg = torch.max(out_seg, 1)[1]
        mask = (gt_seg == pred_seg.view(-1))

        proto_bank = self.prototypes[level]
        # cosine_similarity = torch.mm(_c, proto_bank.view(-1, proto_bank.shape[-1]).t())
        # proto_logits = cosine_similarity
        protos = self.prototypes[level].data.clone()

        for k in range(self.num_classes):

            if valid_mask is not None:
                class_index = (gt_seg == k) & (valid_mask.view(-1) == 1)
  
            else:
                class_index = (gt_seg == k)

            if class_index.sum() == 0:
                continue
            
            with torch.no_grad():
                # attention update
                proto_k = protos[k]              # (P, C)
                c_k = _c[class_index]            # (N, C)
                m_k = mask[class_index].float()  # (N,)
                # m_k = prob[class_index, k] > 0.7

                c_k = c_k * m_k[:, None]

                # cross-attention
                attn_logits = torch.matmul(proto_k, c_k.T)   # (P, N)
                attn = F.softmax(attn_logits / 0.1, dim=-1)  

                f = torch.matmul(attn, c_k)                  # (P, C)
                n = attn.sum(dim=-1)

                if torch.sum(n) > 0 and self.update_prototype is True:
                    f = F.normalize(f, p=2, dim=-1)
                    new_value = momentum_update(old_value=protos[k, n != 0, :], new_value=f[n != 0, :], momentum=self.gamma, debug=False)
                    protos[k, n != 0, :] = new_value

        self.prototypes[level] = nn.Parameter(l2_normalize(protos), requires_grad=False)

        # return proto_logits
    
    def proto_forward(self, proto, level, gt_seg=None, valid_mask=None):
        
        proto = F.interpolate(proto, size=(self.size, self.size), mode='bilinear',align_corners=False)
        # proto = F.interpolate(proto, scale_factor=(4,4),mode ='bilinear')
        _c = rearrange(proto, 'b c h w -> (b h w) c')
        _c = self.feat_norms[level](_c)
        _c = l2_normalize(_c)

        proto_bank = self.prototypes[level]
        proto_bank.data.copy_(l2_normalize(proto_bank))

        masks = torch.einsum('nd,kmd->nmk', _c, proto_bank)
        mask_seg = rearrange(masks, "(b h w) m k -> b (k m) h w", b=proto.shape[0], h=proto.shape[2])

        out_seg = torch.logsumexp(masks / 0.1, dim=1)
        out_seg = rearrange(out_seg, "(b h w) k -> b k h w", b=proto.shape[0], h=proto.shape[2])
        
        if gt_seg is not None:
            gt_seg = gt_seg.float().view(-1)
            self.prototype_learning(_c, level, out_seg, gt_seg, valid_mask)

        return out_seg
    

    def forward(self, x, gt_semantic_seg=None, valid_mask=None):
        B = x.shape[0]
        ### Encoder
        ### Conv Stage

        ### Stage 1
        out = F.relu(F.max_pool2d(self.encoder1(x), 2, 2))
        t1 = out
        ### Stage 2
        out = F.relu(F.max_pool2d(self.encoder2(out), 2, 2))
        t2 = out
        ### Stage 3
        out = F.relu(F.max_pool2d(self.encoder3(out), 2, 2))
        t3 = out

        out, H, W = self.patch_embed3(out)  #(2,256,160)
        
        sin, cos = self.RoPE1((H, W))
        # for blk in self.block1:
        for i, blk in enumerate(self.block1):
            out = blk(out, sin, cos)
  
        out = self.norm3(out)
        t4 = out

        ### Bottleneck
        out, H, W= self.patch_embed4(out)   #(2,64,256)
        
        sin, cos = self.RoPE2((H, W))
        # for blk in self.block2:
        for i, blk in enumerate(self.block2):
            out = blk(out, sin, cos)

        out = self.norm4(out)

        ### Stage 4
        out = F.relu(F.interpolate(self.decoder1(out), scale_factor=(2,2), mode ='bilinear'))

        out = torch.add(out, t4)
        _, _, H, W = out.shape
        sin, cos = self.RoPEd1((H, W))
        # for blk in self.dblock1:
        for i, blk in enumerate(self.dblock1):
            out = blk(out, sin, cos)
        
        ### Stage 3
        out = self.dnorm3(out)
        #######
        out = F.relu(F.interpolate(self.decoder2(out),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,t3)
        _,_,H,W = out.shape
        sin, cos = self.RoPEd2((H, W))
        # for blk in self.dblock2:
        for i, blk in enumerate(self.dblock2):
            out = blk(out, sin, cos)
  
        
        out = self.dnorm4(out)
        ################

        out = F.relu(F.interpolate(self.decoder3(out),scale_factor=(2,2),mode ='bilinear'))
        out_L = self.proto_forward(out, level=0, gt_seg=gt_semantic_seg, valid_mask=valid_mask)

        
        out = torch.add(out,t2)
        out = F.relu(F.interpolate(self.decoder4(out),scale_factor=(2,2),mode ='bilinear'))
        out_M = self.proto_forward(out, level=1, gt_seg=gt_semantic_seg, valid_mask=valid_mask)

        
        out = torch.add(out,t1)
        out = F.relu(F.interpolate(self.decoder5(out),scale_factor=(2,2),mode ='bilinear'))
        out_S = self.proto_forward(out, level=2, gt_seg=gt_semantic_seg, valid_mask=valid_mask)   #org
    
        w = torch.softmax(self.scale_weight, dim=0)
        out = w[0] * out_L + w[1] * out_M + w[2] * out_S

        return out
        # return out, {'out_L': out_L, 'out_M': out_M, 'out_S': out_S}

        
        