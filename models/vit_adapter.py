import math

import torch
from torch import nn
from torch.nn.init import normal_, trunc_normal_

from models.adapter import SpatialPriorModule, InteractionBlock
from models.attention import MSDeformAttn
from models.vit import BEIT
from utils.commons import deform


class Model(BEIT):
    def __init__(self, pretrain_size=224, conv_inplane=64, n_points=4, deform_num_heads=6,
                 init_values=0., cffn_ratio=0.25, deform_ratio=1.0, with_cffn=True, interaction_indexes=None, add_vit_feature=True, *args, **kwargs):
        super(Model, self).__init__(init_values=init_values, *args, **kwargs)
        self.interaction_indexes = interaction_indexes
        self.add_vit_feature = add_vit_feature
        self.pretrain_size = (pretrain_size, pretrain_size)
        self.num_blocks = len(self.blocks)
        self.flags = [i for i in range(-1, self.num_blocks, self.num_blocks // 4)][1:]

        embed_dim = self.embed_dim  # =feature dim
        self.level_embed = nn.Parameter(torch.zeros(3, embed_dim))

        self.spm = SpatialPriorModule(conv_inplane, embed_dim)

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

    def forward(self, x):
        x1, x2 = deform(x)

        c1, c2, c3, c4 = self.spm(x)

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

    def _add_level_embed(self, c2, c3, c4):
        c2 = c2 + self.level_embed[0]
        c3 = c3 + self.level_embed[1]
        c4 = c4 + self.level_embed[2]
        return c2, c3, c4


if __name__ == '__main__':
    model = Model()
