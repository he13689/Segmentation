# 基于 微软BEiT的网络
from functools import partial

import torch
from torch import nn

from models.attention import MSDeformAttn, DropPath


class SpatialPriorModule(nn.Module):
    # 对输入图像进行特征提取， 并且编码成embed dim
    def __init__(self, inplanes=64, embed_dim=384):
        super(SpatialPriorModule, self).__init__()

        # stem part, to extract features from image
        self.stem = nn.Sequential(
            nn.Conv2d(3, inplanes, 3, 2, 1, bias=False), nn.SyncBatchNorm(inplanes), nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, inplanes, 3, 1, 1, bias=False), nn.SyncBatchNorm(inplanes), nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, inplanes, 3, 1, 1, bias=False), nn.SyncBatchNorm(inplanes), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1))

        # F1 part
        self.conv2 = nn.Sequential(nn.Conv2d(inplanes, inplanes * 2, 3, 2, 1, bias=False), nn.SyncBatchNorm(inplanes * 2), nn.ReLU(inplace=True))

        # F2 part
        self.conv3 = nn.Sequential(nn.Conv2d(inplanes * 2, inplanes * 4, 3, 2, 1, bias=False), nn.SyncBatchNorm(inplanes * 4), nn.ReLU(inplace=True))

        # F3 part
        self.conv4 = nn.Sequential(nn.Conv2d(inplanes * 4, inplanes * 4, 3, 2, 1, bias=False), nn.SyncBatchNorm(inplanes * 4), nn.ReLU(inplace=True))

        # final conv
        self.fc1 = nn.Conv2d(inplanes, embed_dim, 1, 1, 0, bias=True)
        self.fc2 = nn.Conv2d(inplanes * 2, embed_dim, 1, 1, 0, bias=True)
        self.fc3 = nn.Conv2d(inplanes * 4, embed_dim, 1, 1, 0, bias=True)
        self.fc4 = nn.Conv2d(inplanes * 4, embed_dim, 1, 1, 0, bias=True)

    def forward(self, x):
        x = self.stem(x)
        y2 = self.conv2(x)
        y3 = self.conv2(y2)
        y4 = self.conv2(y3)

        y1 = self.fc1(x)
        y2 = self.fc2(y2)
        y3 = self.fc2(y3)
        y4 = self.fc2(y4)

        bs, dim, _, _ = x.shape
        y2 = y2.view(bs, dim, -1).transpose(1, 2)
        y3 = y3.view(bs, dim, -1).transpose(1, 2)
        y4 = y4.view(bs, dim, -1).transpose(1, 2)

        return y1, y2, y3, y4


class Injector(nn.Module):
    def __init__(self, dim, num_heads=6, n_points=4, n_levels=1, deform_ratio=1.0, norm_layer=partial(nn.LayerNorm, eps=1e-6), init_values=0.):
        super(Injector, self).__init__()

        self.query_norm = norm_layer(dim)  # in norm
        self.feat_norm = norm_layer(dim)  # out norm

        # ms deform attention is the cross-attention in the pic
        self.attn = MSDeformAttn(d_model=dim, n_levels=n_levels, n_heads=num_heads, n_points=n_points, ratio=deform_ratio)
        self.gamma = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

    def forward(self, query, reference_points, feat, spatial_shapes, level_start_index):
        attn = self.attn(self.query_norm(query), reference_points, self.feat_norm(feat), spatial_shapes, level_start_index, None)
        return query + self.gamma * attn


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)  # add groups param in the conv2d

    def forward(self, x, H, W):
        B, N, C = x.shape
        n = N // 21

        # divide x into three parts
        x1 = x[:, 0:16 * n, :].transpose(1, 2).view(B, C, H * 2, W * 2)
        x2 = x[:, 16 * n:20 * n, :].transpose(1, 2).view(B, C, H, W)
        x3 = x[:, 20 * n:, :].transpose(1, 2).view(B, C, H // 2, W // 2)

        # apply dwconv to the three parts
        x1 = self.dwconv(x1).flatten(2).transpose(1, 2)
        x2 = self.dwconv(x2).flatten(2).transpose(1, 2)
        x3 = self.dwconv(x3).flatten(2).transpose(1, 2)

        x = torch.cat([x1, x2, x3], dim=1)
        return x


class ConvFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super(ConvFFN, self).__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)  # depth wise convolution
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        y = self.fc1(x)
        y = self.dwconv(y, H, W)
        y = self.act(y)
        y = self.drop(y)
        y = self.fc2(y)
        y = self.drop(y)
        return y


class Extractor(nn.Module):
    def __init__(self, dim, num_heads=6, n_points=4, n_levels=1, deform_ratio=1.0, with_cffn=True, cffn_ratio=0.25, drop=0., drop_path=0., norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super(Extractor, self).__init__()

        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)

        self.with_cffn = with_cffn

        # cross attention
        self.attn = MSDeformAttn(d_model=dim, n_levels=n_levels, n_heads=num_heads, n_points=n_points, ratio=deform_ratio)

        if self.with_cffn:  # cffn means conv FFN
            self.ffn = ConvFFN(in_features=dim, hidden_features=int(dim * cffn_ratio), drop=drop)
            self.ffn_norm = norm_layer(dim)
            self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, query, reference_points, feat, spatial_shapes, level_start_index, H, W):
        attn = self.attn(self.query_norm(query), reference_points, self.feat_norm(feat), spatial_shapes, level_start_index, None)
        query = query + attn

        if self.with_cffn:
            query = query + self.drop_path(self.ffn(self.ffn_norm(query), H, W))
        return query


class InteractionBlock:
    def __init__(self, dim, num_heads=6, n_points=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), drop=0., drop_path=0., with_cffn=True, cffn_ratio=0.25, init_values=0., deform_ratio=1.0, extra_extractor=False):
        super(InteractionBlock, self).__init__()

        self.injector = Injector(dim, num_heads, n_points, 3, deform_ratio, norm_layer, init_values)

        self.extractor = Extractor(dim, num_heads, n_points, 1, deform_ratio, with_cffn, cffn_ratio, drop, drop_path, norm_layer)

        if extra_extractor:  # 添加额外的extractor以提供特征信息
            self.extra_extractors = nn.Sequential(*[Extractor(dim=dim, num_heads=num_heads, n_points=n_points, norm_layer=norm_layer, with_cffn=with_cffn,
                                                              cffn_ratio=cffn_ratio, deform_ratio=deform_ratio) for _ in range(2)])
        else:
            self.extra_extractors = None

    def forward(self, x, c, blocks, deform_inputs1, deform_inputs2, H, W):
        # the output of spm will first enter an injector
        y = self.injector(query=x, reference_points=deform_inputs1[0], feat=c, spatial_shapes=deform_inputs1[1], level_start_index=deform_inputs1[2])

        # enumerate all the vit blocks  这个blocks应该是
        for idx, blk in enumerate(blocks):
            y = blk(y, H, W)

        c = self.extractor(query=c, reference_points=deform_inputs2[0],
                           feat=y, spatial_shapes=deform_inputs2[1],
                           level_start_index=deform_inputs2[2], H=H, W=W)
        if self.extra_extractors is not None:
            for extractor in self.extra_extractors:
                c = extractor(query=c, reference_points=deform_inputs2[0],
                              feat=y, spatial_shapes=deform_inputs2[1],
                              level_start_index=deform_inputs2[2], H=H, W=W)
        return x, c
