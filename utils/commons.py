import torch


def get_reference_points(spatial_shapes, device):
    reference_points_list = []
    for lvl, (H_, W_) in enumerate(spatial_shapes):
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
            torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
        ref_y = ref_y.reshape(-1)[None] / H_
        ref_x = ref_x.reshape(-1)[None] / W_
        ref = torch.stack((ref_x, ref_y), -1)
        reference_points_list.append(ref)
    reference_points = torch.cat(reference_points_list, 1)
    reference_points = reference_points[:, :, None]
    return reference_points


def deform(x):
    bs, c, h, w = x.shape

    # 用于spm中  default is tensor([[60, 60], [30, 30], [15, 15]])
    spatial_shapes = torch.as_tensor([(h // 8, w // 8), (h // 16, w // 16), (h // 32, w // 32)], dtype=torch.long, device=x.device)
    level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))  # 0
    #  根据scale计算参考点
    reference_points = get_reference_points([(h // 16, w // 16)], x.device)  # 1 900 1 2

    deform_inputs1 = [reference_points, spatial_shapes, level_start_index]

    spatial_shapes = torch.as_tensor([(h // 16, w // 16)], dtype=torch.long, device=x.device)
    level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
    # h // 8, w // 8 -> torch.Size([1, 3600, 1, 2])   h // 16, w // 16 -> torch.Size([1, 900, 1, 2])   h // 32, w // 32 -> torch.Size([1, 225, 1, 2])
    reference_points = get_reference_points([(h // 8, w // 8), (h // 16, w // 16), (h // 32, w // 32)], x.device)

    deform_inputs2 = [reference_points, spatial_shapes, level_start_index]

    # return 2 lists, and each list contain 3 tensors that stands for reference_points, spatial_shapes, level_start_index
    return deform_inputs1, deform_inputs2
