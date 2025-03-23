"""
MAE training with Asymmtrical Masking
"""
from functools import partial

import math
import torch
import torch.nn as nn
from models_mae import MaskedAutoencoderViT

class AMAEViT(MaskedAutoencoderViT):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=384, depth=12, num_heads=12, decoder_embed_dim=512, decoder_depth=12, decoder_num_heads=16, mlp_ratio=4, norm_layer=nn.LayerNorm, 
                 norm_pix_loss=False, mask_ratio=0.75, temp=0.1, lamda:float=1e-4, token_reg=False, off_diag=False):
        super().__init__(img_size, patch_size, in_channels, embed_dim, depth, num_heads, decoder_embed_dim, decoder_depth, decoder_num_heads, mlp_ratio, norm_layer, norm_pix_loss)

        self.embed_dim = embed_dim
        self.k_size = 3
        self.loc224 = self.get_local_index(196, self.k_size)
        self.k_num = 1
        self.mask_ratio = mask_ratio

        # distillation temperature
        self.temp = temp

        # regularization
        self.lamda = lamda
        self.token_reg = token_reg
        self.off_diag = off_diag
        print(f"Distillation Penalty: {self.lamda} | # of neighbor = {self.k_num} | token = {self.token_reg} | off_diag = {self.off_diag}")

    @staticmethod
    def get_local_index(N_patches, k_size):
        """
        Get the local neighborhood of each patch 
        """
        loc_weight = []
        w = torch.LongTensor(list(range(int(math.sqrt(N_patches)))))
        for i in range(N_patches):
            ix, iy = i//len(w), i%len(w)
            wx = torch.zeros(int(math.sqrt(N_patches)))
            wy = torch.zeros(int(math.sqrt(N_patches)))
            wx[ix]=1
            wy[iy]=1
            for j in range(1,int(k_size//2)+1):
                wx[(ix+j)%len(wx)]=1
                wx[(ix-j)%len(wx)]=1
                wy[(iy+j)%len(wy)]=1
                wy[(iy-j)%len(wy)]=1
            weight = (wy.unsqueeze(0)*wx.unsqueeze(1)).view(-1)
            weight[i] = 0
            loc_weight.append(weight.nonzero().squeeze())
        return torch.stack(loc_weight)
    
    def sim_patches(self, x):
        N, L, D = x.shape
        
        x_norm = nn.functional.normalize(x, dim=-1)
        sim_matrix = x_norm[:,self.loc224] @ x_norm.unsqueeze(2).transpose(-2,-1)
        top_idx = sim_matrix.squeeze().topk(k=self.k_num,dim=-1)[1].view(N, L, self.k_num, 1)

        x_loc = x[:,self.loc224]
        x_loc = torch.gather(x_loc, 1, top_idx.expand(-1, -1, -1, D))
        return x_loc

    def random_relevent_mask(self, xs, mask_ratio):
        N, L, D = xs.shape

        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=xs.device)

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        xs_masked = torch.gather(xs, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary masks
        mask = torch.ones([N, L], device=xs.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        # get the neighbor patches (for teacher input)
        with torch.no_grad():
            x_loc = self.sim_patches(xs)
            x_sim = torch.gather(x_loc, dim=1, index=ids_keep.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.k_num, D))
            x_sim = x_sim.view(N, int(x_sim.size(1)*x_sim.size(2)), D)

        return xs_masked, x_sim, mask, ids_restore
    
    def duo_embed(self, x):
        """
        Embedding of teacher and student model
        """
        # student
        xs = self.patch_embed(x)
        xs = xs + self.pos_embed[:, 1:, :]

        with torch.no_grad():
            xt = self.patch_embed(x)
            xt = xt + self.pos_embed[:, 1:, :]
        return xs, xt

    def fwd_encoder(self, x):
        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x
    
    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        x, xt, mask, ids_restore = self.random_relevent_mask(x, self.mask_ratio)
        
        # forward pass
        x = self.fwd_encoder(x)
        
        with torch.no_grad():
            xt = self.fwd_encoder(xt)

        return x, xt, mask, ids_restore
    
    def normalize_feature(self, pred):
        if self.norm_pix_loss:
            mean = pred.mean(dim=-1, keepdim=True)
            var = pred.var(dim=-1, keepdim=True)
            pred = (pred - mean) / (var + 1e-6)**0.5
        return pred
    
    def corr_loss(self, preds, predt):
        N, D, K = preds.shape

        zs = self.normalize_feature(preds)
        zt = self.normalize_feature(predt)
        
        # correlation
        if not self.token_reg:
            corr = torch.einsum("bki, bkj -> bij", zs, zt).div(D).mean(dim=0)
            diag = torch.eye(K, device=corr.device)
        else:
            corr = torch.einsum("bik, bjk -> bij", zs, zt).div(K).mean(dim=0)
            diag = torch.eye(D, device=corr.device)

        # diagonal
        if not self.off_diag:
            corr_loss = corr[diag.bool()].sum()
        else:
            corr_loss = corr[~diag.bool()].sum()
        return corr_loss.mul(self.lamda)
    
    def forward(self, imgs, mask_ratio=0.75, **kwargs):
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            latent_s, latent_t, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
            preds = self.forward_decoder(latent_s, ids_restore)  # [N, L, p*p*3]

            with torch.no_grad():
                predt = self.forward_decoder(latent_t, ids_restore)  # [N, L, p*p*3]
            
            corr_loss = self.corr_loss(preds, predt)
            rec_loss = self.forward_loss(imgs, preds, mask)

            loss = rec_loss + corr_loss
        return loss


def amae_vit_tiny_patch16(**kwargs):
    model = AMAEViT(
        patch_size=16, embed_dim=192, depth=12, num_heads=12,
        decoder_embed_dim=192, decoder_num_heads=12,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def amae_vit_tiny_patch4(**kwargs):
    model = AMAEViT(
        patch_size=4, embed_dim=192, depth=12, num_heads=12,
        decoder_embed_dim=192, decoder_num_heads=12,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def amae_vit_small_patch16(**kwargs):
    model = AMAEViT(
        patch_size=16, embed_dim=384, depth=12, num_heads=6,
        decoder_embed_dim=256, decoder_num_heads=8,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def amae_vit_base_patch16(**kwargs):
    model = AMAEViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


mae_vit_tiny_patch16 = amae_vit_tiny_patch16
mae_vit_tiny_patch4 = amae_vit_tiny_patch4
mae_vit_small_patch16 = amae_vit_small_patch16
mae_vit_base_patch16 = amae_vit_base_patch16