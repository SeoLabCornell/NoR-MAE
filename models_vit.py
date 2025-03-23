# # Copyright (c) Meta Platforms, Inc. and affiliates.
# # All rights reserved.

# # This source code is licensed under the license found in the
# # LICENSE file in the root directory of this source tree.
# # --------------------------------------------------------
# # References:
# # timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# # DeiT: https://github.com/facebookresearch/deit
# # --------------------------------------------------------

# from functools import partial

# import math
# import torch
# import torch.nn as nn

# import timm
# import timm.models.vision_transformer

# # new_timm = '0.9' in timm.__version__ 
# new_timm = False

# def get_local_index(N_patches, k_size):
#     """
#     Get the local neighborhood of each patch 
#     """
#     loc_weight = []
#     w = torch.LongTensor(list(range(int(math.sqrt(N_patches)))))
#     for i in range(N_patches):
#         ix, iy = i//len(w), i%len(w)
#         wx = torch.zeros(int(math.sqrt(N_patches)))
#         wy = torch.zeros(int(math.sqrt(N_patches)))
#         wx[ix]=1
#         wy[iy]=1
#         for j in range(1,int(k_size//2)+1):
#             wx[(ix+j)%len(wx)]=1
#             wx[(ix-j)%len(wx)]=1
#             wy[(iy+j)%len(wy)]=1
#             wy[(iy-j)%len(wy)]=1
#         weight = (wy.unsqueeze(0)*wx.unsqueeze(1)).view(-1)
#         weight[i] = 0
#         loc_weight.append(weight.nonzero().squeeze())
#     return torch.stack(loc_weight).cuda()

# def sim_patches(x, loc224, k_num):
#     N, L, D = x.shape
    
#     x_norm = nn.functional.normalize(x, dim=-1)
#     sim_matrix = x_norm[:,loc224] @ x_norm.unsqueeze(2).transpose(-2,-1)
#     top_idx = sim_matrix.squeeze().topk(k=k_num,dim=-1)[1].view(N, L, k_num, 1)
#     top_sim = sim_matrix.squeeze().topk(k=k_num,dim=-1)[0].view(N, L, k_num, 1)

#     # reshape top_idx
#     idx2d = top_idx.squeeze(0).squeeze(1)
#     sim_idx = torch.gather(loc224, 1, idx2d)

#     x_loc = x[:, loc224]
#     x_loc = torch.gather(x_loc, 1, top_idx.expand(-1, -1, -1, D))
#     return x_loc, sim_idx.transpose(0,1), top_sim


# class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
#     """ Vision Transformer with support for global average pooling
#     """
#     def __init__(self, global_pool=False, get_features=False, **kwargs):
#         global_pool = "avg" if global_pool else "token"
#         super(VisionTransformer, self).__init__(
#             global_pool=global_pool,
#             **kwargs
#         )

#         if global_pool == "avg":
#             norm_layer = kwargs['norm_layer']
#             embed_dim = kwargs['embed_dim']
#             self.fc_norm = norm_layer(embed_dim)

#             self.norm = nn.Identity()  # remove the original norm
        
#         self.get_features = get_features
#         self.neighbor = False
#         self.k_num = 1
#         self.k_size = 3
#         self.loc224 = get_local_index(196, self.k_size)


#     def forward_features(self, x): 
#         if new_timm:
#             x = super().forward_features(x)
#             return x
            
#         B = x.shape[0]
#         x = self.patch_embed(x)

#         cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        
#         if self.neighbor:
#             x, sim_idx, sim_matrix = sim_patches(x, self.loc224, self.k_num)
#             x = x.squeeze(2)

#         x = torch.cat((cls_tokens, x), dim=1)
#         x = x + self.pos_embed
#         x = self.pos_drop(x)

#         for blk in self.blocks:
#             x = blk(x)

#         if self.global_pool == "avg":
#             if not self.get_features:
#                 x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
#                 outcome = self.fc_norm(x)
#             else:
#                 outcome = x  # global pool without cls token
#         else:
#             x = self.norm(x)
#             outcome = x[:, 0]
#         return outcome

#     def forward(self, x):
#         x = self.forward_features(x)
#         if not self.get_features:
#             x = self.forward_head(x)
#         return x[:, 1:, :]

# def vit_tiny_patch16(**kwargs):
#     model = VisionTransformer(
#         patch_size=16, embed_dim=192, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     return model

# def vit_small_patch16(**kwargs):
#     model = VisionTransformer(
#         patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     return model


# def vit_base_patch16(**kwargs):
#     model = VisionTransformer(
#         patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     return model


# def vit_large_patch16(**kwargs):
#     model = VisionTransformer(
#         patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     return model


# def vit_huge_patch14(**kwargs):
#     model = VisionTransformer(
#         patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     return model

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

import timm
import timm.models.vision_transformer

new_timm = '0.9' in timm.__version__ 

class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, **kwargs):
        global_pool = "avg" if global_pool else "token"
        super(VisionTransformer, self).__init__(
            global_pool=global_pool,
            **kwargs
        )

        if global_pool == "avg":
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            self.norm = nn.Identity()  # remove the original norm

    def forward_features(self, x): 
        if new_timm:
            x = super().forward_features(x)
            return x
            
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool == "avg":
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome

def vit_tiny_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_tiny_patch4(**kwargs):
    model = VisionTransformer(
        patch_size=4, embed_dim=192, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_small_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model