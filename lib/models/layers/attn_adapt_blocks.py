import math
import numpy as np
import torch
import torch.nn as nn
from timm.models.layers import Mlp, DropPath, trunc_normal_, lecun_normal_

from lib.models.layers.attn import Attention

import torch.nn.functional as F

def candidate_elimination(attn: torch.Tensor, tokens: torch.Tensor, lens_t: int, keep_ratio: float, global_index: torch.Tensor, box_mask_z: torch.Tensor):
    """
    Eliminate potential background candidates for computation reduction and noise cancellation.
    Args:
        attn (torch.Tensor): [B, num_heads, L_t + L_s, L_t + L_s], attention weights
        tokens (torch.Tensor):  [B, L_t + L_s, C], template and search region tokens
        lens_t (int): length of template
        keep_ratio (float): keep ratio of search region tokens (candidates)
        global_index (torch.Tensor): global index of search region tokens
        box_mask_z (torch.Tensor): template mask used to accumulate attention weights

    Returns:
        tokens_new (torch.Tensor): tokens after candidate elimination
        keep_index (torch.Tensor): indices of kept search region tokens
        removed_index (torch.Tensor): indices of removed search region tokens
    """
    lens_s = attn.shape[-1] - lens_t    
    bs, hn, _, _ = attn.shape

    lens_keep = math.ceil(keep_ratio * lens_s)
    if lens_keep == lens_s:
        return tokens, global_index, None

    attn_t = attn[:, :, :lens_t, lens_t:]

    


    if box_mask_z is not None:
        #print("\n1\n1\n1")
        box_mask_z = box_mask_z.unsqueeze(1).unsqueeze(-1).expand(-1, attn_t.shape[1], -1, attn_t.shape[-1])
        # attn_t = attn_t[:, :, box_mask_z, :]
        attn_t = attn_t[box_mask_z]
        attn_t = attn_t.view(bs, hn, -1, lens_s)
        attn_t = attn_t.mean(dim=2).mean(dim=1)  # B, H, L-T, L_s --> B, L_s

        # attn_t = [attn_t[i, :, box_mask_z[i, :], :] for i in range(attn_t.size(0))]
        # attn_t = [attn_t[i].mean(dim=1).mean(dim=0) for i in range(len(attn_t))]
        # attn_t = torch.stack(attn_t, dim=0)
    else:
        attn_t = attn_t.mean(dim=2).mean(dim=1)  # B, H, L-T, L_s --> B, L_s

    # use sort instead of topk, due to the speed issue
    # https://github.com/pytorch/pytorch/issues/22812
    sorted_attn, indices = torch.sort(attn_t, dim=1, descending=True)



    topk_attn, topk_idx = sorted_attn[:, :lens_keep], indices[:, :lens_keep]
    non_topk_attn, non_topk_idx = sorted_attn[:, lens_keep:], indices[:, lens_keep:]
    
    keep_index = global_index.gather(dim=1, index=topk_idx)
    
    removed_index = global_index.gather(dim=1, index=non_topk_idx)
    

    # separate template and search tokens
    tokens_t = tokens[:, :lens_t]
    tokens_s = tokens[:, lens_t:]

    # obtain the attentive and inattentive tokens
    B, L, C = tokens_s.shape
    # topk_idx_ = topk_idx.unsqueeze(-1).expand(B, lens_keep, C)

    attentive_tokens = tokens_s.gather(dim=1, index=topk_idx.unsqueeze(-1).expand(B, -1, C))
    # inattentive_tokens = tokens_s.gather(dim=1, index=non_topk_idx.unsqueeze(-1).expand(B, -1, C))

    # compute the weighted combination of inattentive tokens
    # fused_token = non_topk_attn @ inattentive_tokens
    
    # concatenate these tokens
    # tokens_new = torch.cat([tokens_t, attentive_tokens, fused_token], dim=0)
    tokens_new = torch.cat([tokens_t, attentive_tokens], dim=1)

    #print("finish ce func")

    return tokens_new, keep_index, removed_index                       # x, global_index_search, removed_index_search


class CEABlock_ori(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, keep_ratio_search=1.0,):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)     #from timm.models.layers import Mlp, DropPath, trunc_normal_, lecun_normal_

        self.keep_ratio_search = keep_ratio_search


        # self.adap_t = Bi_direct_adapter_w()        
        # self.adap2_t = Bi_direct_adapter_w()
        # self.adap_cross = CrossModal_Templates_Update(dim)


    def forward(self, x, xi, global_index_template, global_index_templatei, global_index_search, global_index_searchi, mask=None, ce_template_mask=None, keep_ratio_search=None):
        
        xori = x
        
        x_attn, attn = self.attn(self.norm1(x), mask, True)   
        x = x + self.drop_path(x_attn)
        # x = x + self.drop_path(x_attn) + self.drop_path(self.adap_t(self.norm1(xi)))  #########-------------------------adapter

        xi_attn, i_attn = self.attn(self.norm1(xi), mask,True)
        xi = xi + self.drop_path(xi_attn)
        # xi = xi + self.drop_path(xi_attn) + self.drop_path(self.adap_t(self.norm1(xori)))  #########-------------------------adapter
                     
        lens_t = global_index_template.shape[1]

        removed_index_search = None
        removed_index_searchi = None
        if self.keep_ratio_search < 1 and (keep_ratio_search is None or keep_ratio_search < 1):
            keep_ratio_search = self.keep_ratio_search if keep_ratio_search is None else keep_ratio_search
            x, global_index_search, removed_index_search = candidate_elimination(attn, x, lens_t, keep_ratio_search, global_index_search, ce_template_mask)
            xi, global_index_searchi, removed_index_searchi = candidate_elimination(i_attn, xi, lens_t, keep_ratio_search, global_index_searchi, ce_template_mask)

        xori = x
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        # x = x + self.drop_path(self.mlp(self.norm2(x))) + self.drop_path(self.adap2_t(self.norm2(xi)))   ###-------adapter

        xi = xi + self.drop_path(self.mlp(self.norm2(xi)))
        # xi = xi + self.drop_path(self.mlp(self.norm2(xi))) + self.drop_path(self.adap2_t(self.norm2(xori)))   ###-------adapter
        # x,xi,rgb_att,tir_att = self.adap_cross(x,xi)

        return x, global_index_template, global_index_search, removed_index_search, attn, xi, global_index_templatei, global_index_searchi, removed_index_searchi, i_attn



class CEABlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, keep_ratio_search=1.0,):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)     #from timm.models.layers import Mlp, DropPath, trunc_normal_, lecun_normal_

        self.keep_ratio_search = keep_ratio_search

    def forward(self, x, xi, global_index_template, global_index_templatei, global_index_search, global_index_searchi, mask=None, ce_template_mask=None, keep_ratio_search=None,Test=None,search_modal_consistent=None):
        
        xori = x

        if Test is True:
            if search_modal_consistent:
                xi_attn, i_attn = x_attn, attn = self.attn(self.norm1(x), mask, True)   
                xi = x = x + self.drop_path(x_attn) 
            else:
                x_attn, attn = self.attn(self.norm1(x), mask, True)   
                x = x + self.drop_path(x_attn)
                # x = x + self.drop_path(x_attn) + self.drop_path(self.adap_t(self.norm1(xi)))  #########-------------------------adapter
                xi_attn, i_attn = self.attn(self.norm1(xi), mask,True)
                xi = xi + self.drop_path(xi_attn) 
                # xi = xi + self.drop_path(xi_attn) + self.drop_path(self.adap_t(self.norm1(xori)))  #########-------------------------adapter
        else:
            x_attn, attn = self.attn(self.norm1(x), mask, True)   
            x = x + self.drop_path(x_attn)
            # x = x + self.drop_path(x_attn) + self.drop_path(self.adap_t(self.norm1(xi)))  #########-------------------------adapter
            xi_attn, i_attn = self.attn(self.norm1(xi), mask,True)
            xi = xi + self.drop_path(xi_attn) 
            # xi = xi + self.drop_path(xi_attn) + self.drop_path(self.adap_t(self.norm1(xori)))  #########-------------------------adapter


        lens_t = global_index_template.shape[1]

        removed_index_search = None
        removed_index_searchi = None
        if self.keep_ratio_search < 1 and (keep_ratio_search is None or keep_ratio_search < 1):
            keep_ratio_search = self.keep_ratio_search if keep_ratio_search is None else keep_ratio_search
            x, global_index_search, removed_index_search = candidate_elimination(attn, x, lens_t, keep_ratio_search, global_index_search, ce_template_mask)
            xi, global_index_searchi, removed_index_searchi = candidate_elimination(i_attn, xi, lens_t, keep_ratio_search, global_index_searchi, ce_template_mask)

        xori = x
        if Test:
            if search_modal_consistent:
                xi = x = x + self.drop_path(self.mlp(self.norm2(x)))
            else:
                x = x + self.drop_path(self.mlp(self.norm2(x)))
                xi = xi + self.drop_path(self.mlp(self.norm2(xi)))
        else:
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            # x = x + self.drop_path(x_attn) + self.drop_path(self.adap_t(self.norm1(xi)))  #########-------------------------adapter
            xi = xi + self.drop_path(self.mlp(self.norm2(xi)))         
        # x = x + self.drop_path(self.mlp(self.norm2(x)))
        # x = x + self.drop_path(self.mlp(self.norm2(x))) + self.drop_path(self.adap2_t(self.norm2(xi)))   ###-------adapter
        # xi = xi + self.drop_path(self.mlp(self.norm2(xi)))
        # xi = xi + self.drop_path(self.mlp(self.norm2(xi))) + self.drop_path(self.adap2_t(self.norm2(xori)))   ###-------adapter
        
        return x, global_index_template, global_index_search, removed_index_search, attn, xi, global_index_templatei, global_index_searchi, removed_index_searchi, i_attn


class CEABlock_Dynamic(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, keep_ratio_search=1.0,):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)     #from timm.models.layers import Mlp, DropPath, trunc_normal_, lecun_normal_

        self.keep_ratio_search = keep_ratio_search

        # self.adap_fusion_router1 = ModalityRouter(256,dim)
        # self.adap_fusion_router2 = ModalityRouter(256,dim)
        # self.adap_t = Bi_direct_adapter()        
        # self.adap2_t = Bi_direct_adapter()
        self.DSU = VitDistributionUncertainty()

    def Style_Intervener(self,x,xi):
        B,L,C = x.shape
        loss = None
        if B == 32:
            step = 8
            num = 4
        else:
            step = 4
            num = 4
        x_t = torch.cat([x,xi],dim=0)
        x_u = self.DSU(x_t)
        x_,xi_ = torch.chunk(x_u,2,dim=0)
        return x_,xi_

    def attention_loss(self,attention_matrices):
        N = len(attention_matrices)
        total_loss = 0.0
        count = 0

        for i in range(N):
            for j in range(i+1, N):
                diff = torch.norm(attention_matrices[i] - attention_matrices[j], p='fro')  # Frobenius范数
                total_loss += diff
                count += 1

        return total_loss / count  # 取平均

    def forward(self, x, xi, global_index_template, global_index_templatei, global_index_search, global_index_searchi, mask=None, ce_template_mask=None, keep_ratio_search=None,Test=None,search_modal_consistent=None):
        
        xori = x
        if Test:
            if search_modal_consistent:
                x_attn, attn = self.attn(self.norm1(x), mask, True)   
                xi_attn, i_attn = x_attn, attn
                x = x + self.drop_path(x_attn)
                xi = x
                attn_loss = None
            else: 
                x_n,xi_n = self.norm1(x),self.norm1(xi)
                # x_f = torch.cat([t_n+ti_n,s_n*w_up+si_n*w_down],dim=1)
                x_attn, attn = self.attn(x_n, mask, True)   
                x = x + self.drop_path(x_attn) 
                # x = x + self.drop_path(x_attn) + self.drop_path(self.adap_t(self.norm1(xi)))  #########-------------------------adapter
                xi_attn, i_attn = self.attn(xi_n, mask,True)
                xi = xi + self.drop_path(xi_attn) 
                attn_loss = None
            # xi = xi + self.drop_path(xi_attn) + self.drop_path(self.adap_t(self.norm1(xori)))  #########-------------------------adapter
        else:
            x_u,xi_u = self.Style_Intervener(x,xi)
            # x_f = torch.cat([t_n+ti_n,s_n*w_up+si_n*w_down],dim=1)
            # x_n,xi_n = self.norm1(x),self.norm1(xi)


            # 仅第一阶段启用
            t_n,ti_n = x[:,:64],xi[:,:64]
            s_n,si_n = x[:,64:],xi[:,64:]

            t_u,ti_u = x_u[:,:64],xi_u[:,:64]
            s_u,si_u = x_u[:,64:],xi_u[:,64:]

            x12 = torch.cat([t_n,s_u],dim=1)
            xi12 = torch.cat([ti_n,si_u],dim=1)

            x21 = torch.cat([t_u,s_n],dim=1)
            xi21 = torch.cat([ti_u,si_n],dim=1)
            x_attn_12, attn_12 = self.attn(self.norm1(x12), mask, True)
            xi_attn_12, attni_12 = self.attn(self.norm1(xi12), mask, True)
            x_attn_21, attn_21 = self.attn(self.norm1(x21), mask, True)
            xi_attn_21, attni_21 = self.attn(self.norm1(xi21), mask, True)
            x_attn, attn = self.attn(self.norm1(x), mask, True)   
            xi_attn, i_attn = self.attn(self.norm1(xi), mask,True)
            # 仅第一阶段启用

            x_attn_u, attn_u = self.attn(self.norm1(x_u), mask, True)
            xi_attn_u, i_attn_u = self.attn(self.norm1(xi_u), mask, True)            

            x = x  + self.drop_path(x_attn_u) 
            # x = x + self.drop_path(x_attn) + self.drop_path(self.adap_t(self.norm1(xi)))  #########-------------------------adapter
            xi = xi  + self.drop_path(xi_attn_u) 
            # xi = xi + self.drop_path(xi_attn) + self.drop_path(self.adap_t(self.norm1(xori)))  #########-------------------------adapter
            #计算矩阵之间的差值，应当尽可能一致，因为只是改变分布 希望都关注到因果特征中
            attn_list = [attn,attn_12,attn_21,attn_u]
            attn_up_log = F.log_softmax(torch.cat(attn_list, dim=0).mean(dim=1), dim=-1)  # 输入需取对数
            attni_list = [i_attn,attni_12,attni_21,i_attn_u]
            attn_down = F.softmax(torch.cat(attni_list,dim=0).mean(dim=1),dim=-1)
            attn_loss = (F.kl_div(attn_up_log, attn_down, reduction='batchmean'))
            # attn_loss = (self.attention_loss(attn_list)+ self.attention_loss(attni_list))/2
            # attn_loss = 0

        lens_t = global_index_template.shape[1]

        removed_index_search = None
        removed_index_searchi = None
        if self.keep_ratio_search < 1 and (keep_ratio_search is None or keep_ratio_search < 1):
            keep_ratio_search = self.keep_ratio_search if keep_ratio_search is None else keep_ratio_search
            x, global_index_search, removed_index_search = candidate_elimination(attn, x, lens_t, keep_ratio_search, global_index_search, ce_template_mask)
            xi, global_index_searchi, removed_index_searchi = candidate_elimination(i_attn, xi, lens_t, keep_ratio_search, global_index_searchi, ce_template_mask)

        xori = x

        #训练阶段 or 测试阶段 不同
        if Test:
            if search_modal_consistent:
                xi = x = x + self.drop_path(self.mlp(self.norm2(x)))
            else:
                x = x + self.drop_path(self.mlp(self.norm2(x)))
                xi = xi + self.drop_path(self.mlp(self.norm2(xi)))
        else:
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            # x = x + self.drop_path(self.mlp(self.norm2(x))) + self.drop_path(self.adap2_t(self.norm2(xi)))   ###-------adapter
            xi = xi + self.drop_path(self.mlp(self.norm2(xi)))
            # xi = xi + self.drop_path(self.mlp(self.norm2(xi))) + self.drop_path(self.adap2_t(self.norm2(xori)))   ###-------adapter
        
        return x, global_index_template, global_index_search, removed_index_search, attn, xi, global_index_templatei, global_index_searchi, removed_index_searchi, i_attn,attn_loss


class VitDistributionUncertainty(nn.Module):
    """
    Modified for ViT features (B,L,C format)
    Args:
        p (float): probability of applying uncertainty, range [0,1]
        eps (float): small value to prevent numerical instability
    """
    def __init__(self, p=1, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.p = p
        self.factor = 1.0  # 噪声系数

    def _reparameterize(self, mu, std):
        """重参数化采样"""
        epsilon = torch.randn_like(std) * self.factor
        return mu + epsilon * std

    def sqrtvar(self, x):
        """计算跨样本的方差"""
        t = (x.var(dim=0, keepdim=True) + self.eps).sqrt()  # 计算各通道方差
        return t.repeat(x.size(0), 1)  # 重复到batch维度

    def forward(self, x):
        """
        Input format: (B, L, C)
        B: batch size, L: sequence length, C: feature dimension
        """
        if not self.training or np.random.random() > self.p:
            return x

        # 计算统计量 (B, C)
        mean = x.mean(dim=1, keepdim=False)  # 沿序列维度L计算均值
        std = (x.var(dim=1, keepdim=False) + self.eps).sqrt()

        # 计算统计量的不确定性 (B, C)
        sqrtvar_mu = self.sqrtvar(mean)  # 跨样本的均值方差
        sqrtvar_std = self.sqrtvar(std)  # 跨样本的标准差方差

        # 重参数化得到扰动后的统计量
        beta = self._reparameterize(mean, sqrtvar_mu)
        gamma = self._reparameterize(std, sqrtvar_std)

        # 重塑形状以匹配ViT特征维度 (B, L, C)
        beta = beta.unsqueeze(1)  # (B, 1, C)
        gamma = gamma.unsqueeze(1)  # (B, 1, C)

        # 标准化和扰动
        x = (x - mean.unsqueeze(1)) / std.unsqueeze(1)
        x = x * gamma + beta

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        #print("class Block ")
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask=None):
        #print("class Block forward")
        x = x + self.drop_path(self.attn(self.norm1(x), mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x