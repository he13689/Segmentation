# use the config of upernet_beit_adapter_large_480_80k_pascal_context_59_ss
import torch, warnings, math
from torch import nn
from torch.nn.init import normal_, trunc_normal_
import torch.nn.functional as F
from models.adapter import SpatialPriorModule, InteractionBlock
from models.attention import MSDeformAttn
from models.vit import BEIT
from utils.commons import deform

warnings.filterwarnings('ignore')


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

        # 层级编码 3 1024  这个编码会被加到spm的输出上
        self.level_embed = nn.Parameter(torch.zeros(3, embed_dim))

        self.spm = SpatialPriorModule(conv_inplane, embed_dim)

        self.interactions = nn.Sequential(*[
            InteractionBlock(dim=embed_dim, num_heads=deform_num_heads, n_points=n_points, init_values=init_values, drop_path=self.drop_path_rate,
                             norm_layer=self.norm_layer, with_cffn=with_cffn, cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
                             extra_extractor=True if i == len(interaction_indexes) - 1 else False) for i in range(len(interaction_indexes))])

        self.up = nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2)  # 放大2倍
        self.norm1 = nn.BatchNorm2d(embed_dim)
        self.norm2 = nn.BatchNorm2d(embed_dim)
        self.norm3 = nn.BatchNorm2d(embed_dim)
        self.norm4 = nn.BatchNorm2d(embed_dim)

        self.up.apply(self._init_weights)
        self.spm.apply(self._init_weights)
        self.interactions.apply(self._init_weights)
        self.apply(self._init_deform_weights)
        normal_(self.level_embed)

    def forward(self, x):
        # deform x to x1 and x2
        x1, x2 = deform(x)

        # spatial prior module
        # c1 : 16 1024 120 120  c2 : 16 3600 1024  c3 : 16 900 1024  c4 : 16 225 1024
        c1, c2, c3, c4 = self.spm(x)
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)  # add level embedding to the output of spm
        c = torch.cat([c2, c3, c4], dim=1)  # 16 4725 1024  this is the concat

        x, H, W = self.patch_embed(x)
        bs, n, dim = x.shape
        cls = self.cls_token.expand(bs, -1, -1)  # create class token for each batch

        if self.pos_embed is not None:
            pos_emb = self._get_pos_embed(self.pos_embed, H, W)
            x += pos_emb

        x = self.pos_drop(x)

        # interaction part
        out = list()
        for i, layer in enumerate(self.interactions):
            indexes = self.interaction_indexes[i]  # indexes记录了那些block对应哪个interaction
            # self.blocks[indexes[0]:indexes[-1] + 1] 取出对应的block
            x, c, cls = layer(x, c, cls, self.blocks[indexes[0]:indexes[-1] + 1], x1, x2, H, W)
            out.append(x.transpose(1, 2).view(bs, dim, H, W).contiguous())

        # split and reshape
        c2 = c[:, 0:c2.size(1), :]
        c3 = c[:, c2.size(1):c2.size(1) + c3.size(1), :]
        c4 = c[:, c2.size(1) + c3.size(1):, :]

        c2 = c2.transpose(1, 2).view(bs, dim, H * 2, W * 2).contiguous()
        c3 = c3.transpose(1, 2).view(bs, dim, H, W).contiguous()
        c4 = c4.transpose(1, 2).view(bs, dim, H // 2, W // 2).contiguous()
        c1 = self.up(c2) + c1

        if self.add_vit_feature:
            x1, x2, x3, x4 = out
            x1 = F.interpolate(x1, scale_factor=4, mode='bilinear', align_corners=False)
            x2 = F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=False)
            x4 = F.interpolate(x4, scale_factor=0.5, mode='bilinear', align_corners=False)
            c1, c2, c3, c4 = c1 + x1, c2 + x2, c3 + x3, c4 + x4

        # Final Norm
        f1 = self.norm1(c1)
        f2 = self.norm2(c2)
        f3 = self.norm3(c3)
        f4 = self.norm4(c4)
        return [f1, f2, f3, f4]


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

    def _get_pos_embed(self, pos_embed, H, W):
        pos_embed = pos_embed.reshape(
            1, self.pretrain_size[0] // 16, self.pretrain_size[1] // 16, -1).permute(0, 3, 1, 2)
        pos_embed = F.interpolate(pos_embed, size=(H, W), mode='bicubic', align_corners=False). \
            reshape(1, -1, H * W).permute(0, 2, 1)
        return pos_embed

    def _add_level_embed(self, c2, c3, c4):
        c2 = c2 + self.level_embed[0]
        c3 = c3 + self.level_embed[1]
        c4 = c4 + self.level_embed[2]
        return c2, c3, c4


if __name__ == '__main__':
    model = Model(img_size=480, patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
                  use_abs_pos_emb=False, use_rel_pos_bias=True, init_values=1e-6, drop_path_rate=0.3, conv_inplane=64,
                  n_points=4, deform_num_heads=16, cffn_ratio=0.25, deform_ratio=0.5, interaction_indexes=[[0, 5], [6, 11], [12, 17], [18, 23]])

    fake_image = torch.rand(16, 3, 480, 480)
    result = model(fake_image)
