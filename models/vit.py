# use the config of upernet_beit_adapter_large_480_80k_pascal_context_59_ss
from functools import partial

import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_


class PatchEmbed(nn.Module):
    """ use CNN to seperate the image and make patches. """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        # 经过resize后输入图像大小  这里是640
        img_size = (img_size, img_size)
        # 每个 patch 大小，即每个patch多少像素
        patch_size = (patch_size, patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, **kwargs):
        # use CNN to extract patches, and then flatten
        B, C, H, W = x.shape
        x = self.proj(x)
        Hp, Wp = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)
        return x, Hp, Wp


class HybridEmbed(nn.Module):
    """ CNN Feature Map Embedding  ---- unlike patch embed, it uses the feature map from CNN
    Extract feature map from CNN, flatten, project to embedding dim.
    """

    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = (img_size, img_size)
        self.img_size = img_size
        self.backbone = backbone  # this is a CNN net, which is used to extract features
        if feature_size is None:
            with torch.no_grad():
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))[-1]
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = (feature_size, feature_size)
            feature_dim = self.backbone.feature_info.channels()[-1]
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Linear(feature_dim, embed_dim)  # 将seq从feature dim 投射到 embed dim

    def forward(self, x):
        x = self.backbone(x)[-1]
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class RelativePosition(nn.Module):
    # 相对位置偏置，用于表示seq序列的位置信息
    def __init__(self, window_size, num_heads):
        super().__init__()
        self.window_size = window_size
        self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
        # 这里相对位置编码表
        self.relative_position_bias_table = nn.Parameter(torch.zeros(self.num_relative_distance, num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # cls to token & token 2 cls & cls to cls
        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = \
            torch.zeros(size=(window_size[0] * window_size[1] + 1,) * 2, dtype=relative_coords.dtype)
        relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        relative_position_index[0, 0:] = self.num_relative_distance - 3
        relative_position_index[0:, 0] = self.num_relative_distance - 2
        relative_position_index[0, 0] = self.num_relative_distance - 1

        # we can set a check point to see the relative position
        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self):
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1] + 1, self.window_size[0] * self.window_size[1] + 1, -1)  # Wh*Ww,Wh*Ww,nH
        return relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww


class Attention(nn.Module):
    # 注意力模块
    def __init__(self):
        super(Attention, self).__init__()


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm, window_size=None, attn_head_dim=None, with_cp=False):
        super(Block, self).__init__()
        self.with_cp = with_cp
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                              proj_drop=drop, window_size=window_size, attn_head_dim=attn_head_dim)


# B idirectional E ncoder representation from I mage T ransformers  = BEIT
# 基于图像重建进行预训练  预训练的目标是基于损坏的图像patch恢复原始视觉token
class BEIT(nn.Module):
    def __init__(self, img_size=512, patch_size=16, in_chans=3, num_classes=80, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., hybrid_backbone=None, norm_layer=None, init_values=None, use_checkpoint=False, use_abs_pos_emb=False, use_rel_pos_bias=True,
                 use_shared_rel_pos_bias=False, pretrained=None, with_cp=False):
        super(BEIT, self).__init__()

        # 定义正则化层
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.norm_layer = norm_layer

        self.num_classes = num_classes  # 类别
        self.num_features = self.embed_dim = embed_dim  # 特征数等于编码维度
        self.drop_path_rate = drop_path_rate  # drop path 几率

        # 2种不同的patch embed的方式， 选择一种对输入进行patch
        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        # 分类token 是一个parameter 可训练  作用见Q&As  将 cls 作为序列的第一个元素，随输入向量传入网络
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        if use_abs_pos_emb:  # 是否使用绝对位置编码
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:
            self.pos_embed = None

        # 使用相对位置编码
        if use_rel_pos_bias:
            self.rel_pos_bias = RelativePosition(window_size=self.patch_embed.patch_shape, num_heads=num_heads)
        else:
            self.rel_pos_bias = None

        # 位置dropout
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # 根据随机深度衰减而计算dropout path rate

        # 是否使用相对位置bias
        self.use_rel_pos_bias = use_rel_pos_bias

        # 这个是 backbone vit 的blocks，是一个个的transform块
        self.blocks = nn.ModuleList(
            [Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                   attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, with_cp=with_cp, init_values=init_values,
                   window_size=self.patch_embed.patch_shape if use_rel_pos_bias else None) for i in range(depth)])

        trunc_normal_(self.cls_token, std=.02)
        self.init_weights(pretrained)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            load_checkpoint(self, pretrained, strict=False)


# 基于 微软BEiT的网络
class BackBone(BEIT):
    def __init__(self):
        super(BackBone, self).__init__()
