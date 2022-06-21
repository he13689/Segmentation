# use the config of upernet_beit_adapter_large_480_80k_pascal_context_59_ss
import math
from functools import partial

import torch
import torch.nn as nn
from loguru import logger
from torch.nn.init import trunc_normal_, normal_

from models.attention import AttentionBlock, MSDeformAttn
from models.embedding import HybridEmbed, PatchEmbed
from utils.loader import load_checkpoint
from utils.relative_position import RelativePosition


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

        # 创建attention blocks, block的数量取决于depth深度
        self.blocks = nn.ModuleList(
            [AttentionBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                            attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, with_cp=with_cp, init_values=init_values,
                            window_size=self.patch_embed.patch_shape if use_rel_pos_bias else None) for i in range(depth)])

        trunc_normal_(self.cls_token, std=.02)

        # 初始化模型参数
        self.apply(self._init_weights)

        # 这是加载模型参数的函数  功能过于繁琐 有待商榷
        self.init_weights(pretrained)

    def init_weights(self, pretrained=None):
        # 加载预训练权重
        if isinstance(pretrained, str):
            _, state_dict = load_checkpoint(self, pretrained, strict=False)
            info = self.load_state_dict(state_dict, strict=False)
            logger.warning(f'The loading info is {info}')

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


# 基于 微软BEiT的网络
class SpatialPriorModule(nn.Module):
    def __init__(self, inplanes=64, embed_dim=384):
        super(SpatialPriorModule, self).__init__()


class InteractionBlock:
    def __init__(self):
        super(InteractionBlock, self).__init__()


class BackBone(BEIT):
    def __init__(self, pretrain_size=224, conv_inplane=64, n_points=4, deform_num_heads=6,
                 init_values=0., cffn_ratio=0.25, deform_ratio=1.0, with_cffn=True, interaction_indexes=None, add_vit_feature=True, *args, **kwargs):
        super(BackBone, self).__init__(init_values=init_values, *args, **kwargs)
        self.interaction_indexes = interaction_indexes
        self.add_vit_feature = add_vit_feature
        self.pretrain_size = (pretrain_size, pretrain_size)
        self.num_blocks = len(self.blocks)
        self.flags = [i for i in range(-1, self.num_blocks, self.num_blocks // 4)][1:]

        embed_dim = self.embed_dim  # =feature dim
        self.level_embed = nn.Parameter(torch.zeros(3, embed_dim))

        self.spm = SpatialPriorModule()

        self.interactions = nn.Sequential(*[
            InteractionBlock(dim=embed_dim, num_heads=deform_num_heads, n_points=n_points, init_values=init_values, drop_path=self.drop_path_rate,
                             norm_layer=self.norm_layer, with_cffn=with_cffn, cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
                             extra_extractor=True if i == len(interaction_indexes) - 1 else False) for i in range(len(interaction_indexes))])

        self.up = nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2)  # 放大2倍
        self.norm1 = nn.SyncBatchNorm(embed_dim)
        self.norm2 = nn.SyncBatchNorm(embed_dim)
        self.norm3 = nn.SyncBatchNorm(embed_dim)
        self.norm4 = nn.SyncBatchNorm(embed_dim)

        self.up.apply(self._init_weights)
        self.spm.apply(self._init_weights)
        self.interactions.apply(self._init_weights)
        self.apply(self._init_deform_weights)
        normal_(self.level_embed)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def _init_deform_weights(self, m):
        if isinstance(m, MSDeformAttn):
            m._reset_parameters()


if __name__ == '__main__':
    model = BEIT()
