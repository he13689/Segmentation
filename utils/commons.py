import torch


def deform(x):
    bs, c, h, w = x.shape

    spatial_shapes = torch.as_tensor([(h // 8, w // 8), (h // 16, w // 16), (h // 32, w // 32)], dtype=torch.long, device=x.device)
