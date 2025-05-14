import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np


def conv(in_channels, out_channels, freeze_bn=False):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    ]
    module = nn.Sequential(*layers)
    if freeze_bn:
        for m in module.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False
    return module

# --------------------
# 1. Center Predictor
# --------------------
class CenterPredictor(nn.Module):
    def __init__(self, inplanes=512, channel=256, feat_sz=16, stride=16, freeze_bn=False):
        super(CenterPredictor, self).__init__()
        self.feat_sz = feat_sz
        self.stride = stride
        self.img_sz = self.feat_sz * self.stride
        # center score map
        self.conv1_ctr = conv(inplanes, channel, freeze_bn)
        self.conv2_ctr = conv(channel, channel//2, freeze_bn)
        self.conv3_ctr = conv(channel//2, channel//4, freeze_bn)
        self.conv4_ctr = conv(channel//4, channel//8, freeze_bn)
        self.conv5_ctr = nn.Conv2d(channel//8, 1, kernel_size=1)
        # offset regression (dx, dy)
        self.conv1_offset = conv(inplanes, channel, freeze_bn)
        self.conv2_offset = conv(channel, channel//2, freeze_bn)
        self.conv3_offset = conv(channel//2, channel//4, freeze_bn)
        self.conv4_offset = conv(channel//4, channel//8, freeze_bn)
        self.conv5_offset = nn.Conv2d(channel//8, 2, kernel_size=1)
        # size regression (w, h)
        self.conv1_size = conv(inplanes, channel, freeze_bn)
        self.conv2_size = conv(channel, channel//2, freeze_bn)
        self.conv3_size = conv(channel//2, channel//4, freeze_bn)
        self.conv4_size = conv(channel//4, channel//8, freeze_bn)
        self.conv5_size = nn.Conv2d(channel//8, 2, kernel_size=1)
        # 参数初始化
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, gt_score_map=None):
        score_map_ctr, size_map, offset_map = self._get_maps(x)
        score_input = gt_score_map.unsqueeze(1) if gt_score_map is not None else score_map_ctr
        bbox = self._cal_bbox(score_input, size_map, offset_map)
        return score_map_ctr, bbox, size_map, offset_map

    def _get_maps(self, x):
        def _sigmoid(x):
            y = torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)
            return y
        # center
        c = self.conv1_ctr(x)
        c = self.conv2_ctr(c)
        c = self.conv3_ctr(c)
        c = self.conv4_ctr(c)
        score_map_ctr = self.conv5_ctr(c)
        # offset
        o = self.conv1_offset(x)
        o = self.conv2_offset(o)
        o = self.conv3_offset(o)
        o = self.conv4_offset(o)
        offset_map = self.conv5_offset(o)
        # size
        s = self.conv1_size(x)
        s = self.conv2_size(s)
        s = self.conv3_size(s)
        s = self.conv4_size(s)
        size_map = self.conv5_size(s)
        # return score_map_ctr, size_map, offset_map
        return _sigmoid(score_map_ctr), _sigmoid(size_map), offset_map

    def _cal_bbox(self, score_map, size_map, offset_map, return_score=False):
        B,_,H,W = score_map.shape
        assert H == self.feat_sz and W == self.feat_sz, \
            f"Expected score_map size ({self.feat_sz}x{self.feat_sz}), got ({H}x{W})"
        score_vec = score_map.view(B, -1)
        max_score, idx = torch.max(score_vec, dim=1, keepdim=True)
        # 坐标索引
        idx_y = idx // self.feat_sz
        idx_x = idx % self.feat_sz
        # 提取 size 和 offset
        idx_expand = idx.unsqueeze(1).expand(-1, 2, -1)
        size = size_map.view(B,2,-1).gather(2, idx_expand).squeeze(-1)
        offset = offset_map.view(B,2,-1).gather(2, idx_expand).squeeze(-1)


        # cx, cy, w, h
        bbox = torch.cat([(idx_x.to(torch.float) + offset[:, :1]) / self.feat_sz,
                          (idx_y.to(torch.float) + offset[:, 1:]) / self.feat_sz,
                          size.squeeze(-1)], dim=1)

        return (bbox, max_score) if return_score else bbox


class SiameseTracker(nn.Module):
    def __init__(self, backbone='resnet50', pretrained=True, feat_sz=16, stride=16, freeze_bn=False):
        super(SiameseTracker, self).__init__()

        res = getattr(models, backbone)(pretrained=pretrained)
        self.backbone = nn.Sequential(*list(res.children())[:-2])
        C = res.fc.in_features  
        self.fuse = nn.Conv2d(C*2, C, kernel_size=1, bias=False)
        self.adap_down = nn.Conv2d(2048,768,1)
        self.DSU = DistributionUncertainty()

    def Style_Intervener(self,x,xi):
        # B,L,C = x.shape
        # loss = None
        x_t = torch.cat([x,xi],dim=0)
        x_u = self.DSU(x_t)
        x_,xi_ = torch.chunk(x_u,2,dim=0)
        return x_,xi_



    def forward(self, tmpl_rgb, tmpl_tir, search_rgb, search_tir, Test=None,gt_score_map=None):

        f_t_rgb = self.backbone(tmpl_rgb)
        f_t_tir = self.backbone(tmpl_tir)
        f_t_rgb,f_t_tir = self.Style_Intervener(f_t_rgb,f_t_tir)
        f_t = self.fuse(torch.cat([f_t_rgb, f_t_tir], dim=1))

        f_s_rgb = self.backbone(search_rgb)
        f_s_tir = self.backbone(search_tir)
        f_s_rgb,f_s_tir = self.Style_Intervener(f_s_rgb,f_s_tir)

        f_s = self.fuse(torch.cat([f_s_rgb, f_s_tir], dim=1))
        f_s = F.interpolate(f_s, size=(19, 19), mode='bilinear', align_corners=False)
        f_s_rgb = F.interpolate(f_s_rgb, size=(19, 19), mode='bilinear', align_corners=False)
        f_s_tir = F.interpolate(f_s_rgb, size=(19, 19), mode='bilinear', align_corners=False)

        if Test is None:
            # depthwise
            xcorrs_rgb = []
            B, C, h, w = f_t.size()
            for i in range(B):
                kerr = f_t_rgb[i:i+1].permute(1,0,2,3)
                fer = f_s_rgb[i:i+1]
                xcorrs_rgb.append(F.conv2d(fer, kerr, groups=C))
            xcorrs_rgb = torch.cat(xcorrs_rgb,dim=0)


            # depthwise 
            xcorrs_tir = []
            B, C, h, w = f_t.size()
            for i in range(B):
                keri = f_t_tir[i:i+1].permute(1,0,2,3)
                fei = f_s_tir[i:i+1]
                xcorrs_tir.append(F.conv2d(fei, keri, groups=C))   
            xcorrs_tir = torch.cat(xcorrs_tir,dim=0)     


        # depthwise 
        xcorrs = []
        B, C, h, w = f_t.size()
        for i in range(B):
            ker = f_t[i:i+1].permute(1,0,2,3)
            fe = f_s[i:i+1]
            xcorrs.append(F.conv2d(fe, ker, groups=C))
        xcorrs = xcorr = torch.cat(xcorrs, dim=0)
        if Test is None:
            xcorrs_rgb = F.softmax(xcorrs_rgb.mean(dim=1), dim=-1) 
            xcorrs_tir = F.softmax(xcorrs_tir.mean(dim=1), dim=-1)  
            xcorr = F.log_softmax(xcorr.mean(dim=1), dim=-1) 
            rgb_loss = (F.kl_div(xcorr, xcorrs_rgb, reduction='batchmean'))
            tir_loss = (F.kl_div(xcorr, xcorrs_rgb, reduction='batchmean'))
            loss = (rgb_loss+tir_loss)/2
        else:
            loss = None

        # Center & bbox 预测
        xcorrs = self.adap_down(xcorrs).reshape(B,768,256).permute(0,2,1)
        return xcorrs,loss



class DistributionUncertainty(nn.Module):
    """
    Distribution Uncertainty Module
        Args:
        p   (float): probabilty of foward distribution uncertainty module, p in [0,1].

    """

    def __init__(self, p=1, eps=1e-6):
        super(DistributionUncertainty, self).__init__()
        self.eps = eps
        self.p = p
        self.factor = 1.0

    def _reparameterize(self, mu, std):
        epsilon = torch.randn_like(std) * self.factor
        return mu + epsilon * std

    def sqrtvar(self, x):
        t = (x.var(dim=0, keepdim=True) + self.eps).sqrt()
        t = t.repeat(x.shape[0], 1)
        return t

    def forward(self, x):
        if (not self.training) or (np.random.random()) > self.p:
            return x

        mean = x.mean(dim=[2, 3], keepdim=False)
        std = (x.var(dim=[2, 3], keepdim=False) + self.eps).sqrt()

        sqrtvar_mu = self.sqrtvar(mean)
        sqrtvar_std = self.sqrtvar(std)

        beta = self._reparameterize(mean, sqrtvar_mu)
        gamma = self._reparameterize(std, sqrtvar_std)

        x = (x - mean.reshape(x.shape[0], x.shape[1], 1, 1)) / std.reshape(x.shape[0], x.shape[1], 1, 1)
        x = x * gamma.reshape(x.shape[0], x.shape[1], 1, 1) + beta.reshape(x.shape[0], x.shape[1], 1, 1)

        return x
