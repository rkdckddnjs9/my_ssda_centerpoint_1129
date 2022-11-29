import torch
from torch import nn
from pcdet.utils.simplevis import center_to_corner_box2d_pillar

def bilinear_interpolate_torch(im, x, y):
    """
    Args:
        im: (H, W, C) [y, x]
        x: (N)
        y: (N)

    Returns:

    """
    x0 = torch.floor(x).long()
    x1 = x0 + 1

    y0 = torch.floor(y).long()
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, im.shape[1] - 1)
    x1 = torch.clamp(x1, 0, im.shape[1] - 1)
    y0 = torch.clamp(y0, 0, im.shape[0] - 1)
    y1 = torch.clamp(y1, 0, im.shape[0] - 1)

    Ia = im[y0, x0]
    Ib = im[y1, x0]
    Ic = im[y0, x1]
    Id = im[y1, x1]

    wa = (x1.type_as(x) - x) * (y1.type_as(y) - y)
    wb = (x1.type_as(x) - x) * (y - y0.type_as(y))
    wc = (x - x0.type_as(x)) * (y1.type_as(y) - y)
    wd = (x - x0.type_as(x)) * (y - y0.type_as(y))
    ans = torch.t((torch.t(Ia) * wa)) + torch.t(torch.t(Ib) * wb) + torch.t(torch.t(Ic) * wc) + torch.t(torch.t(Id) * wd)
    return ans

class BEVFeatureExtractor(nn.Module): 
    def __init__(self, model_cfg, pc_start, 
            voxel_size, out_stride, num_point):
        super().__init__()
        self.pc_start = pc_start 
        self.voxel_size = voxel_size
        self.out_stride = out_stride
        self.num_point = num_point #아마 spatial feature수인거 같은데 한번더 확인 ㄱㄱ

    def absl_to_relative(self, absolute):
        a1 = (absolute[..., 0] - self.pc_start[0]) / self.voxel_size[0] / self.out_stride 
        a2 = (absolute[..., 1] - self.pc_start[1]) / self.voxel_size[1] / self.out_stride 

        return a1, a2
    
    def get_box_center(self, batch_dict):
        # box [List]
        boxes = batch_dict['final_box_dicts']
        centers = [] 
        for box in boxes:            
            if self.num_point == 1 or len(box['pred_boxes']) == 0:
                centers.append(box['pred_boxes'][:, :3])
                
            elif self.num_point == 5:
                center2d = box['pred_boxes'][:, :2]
                height = box['pred_boxes'][:, 2:3]
                dim2d = box['pred_boxes'][:, 3:5]
                rotation_y = box['pred_boxes'][:, -1]

                corners = center_to_corner_box2d_pillar(center2d, dim2d, rotation_y)

                front_middle = torch.cat([(corners[:, 0] + corners[:, 1])/2, height], dim=-1)
                back_middle = torch.cat([(corners[:, 2] + corners[:, 3])/2, height], dim=-1)
                left_middle = torch.cat([(corners[:, 0] + corners[:, 3])/2, height], dim=-1)
                right_middle = torch.cat([(corners[:, 1] + corners[:, 2])/2, height], dim=-1) 

                points = torch.cat([box['pred_boxes'][:, :3], front_middle, back_middle, left_middle, \
                    right_middle], dim=0)

                centers.append(points)
            else:
                raise NotImplementedError()

        return centers

    def forward(self, batch_dict):
        batch_size = len(batch_dict['spatial_features_2d'])
        bev_feature = batch_dict['spatial_features_2d'].permute(0, 2, 3, 1) #.contiguous() # pred_dicts[bs_idx]['bev_feature'] = bev_feature[bs_idx].permute(0, 2, 3, 1).contiguous()
        ret_maps = [] 

        # batch_centers = batch_dict['rois'][..., :3]
        batch_centers = self.get_box_center(batch_dict)

        for batch_idx in range(batch_size):
            xs, ys = self.absl_to_relative(batch_centers[batch_idx])
            
            # N x C 
            feature_map = bilinear_interpolate_torch(bev_feature[batch_idx],
             xs, ys)

            if self.num_point >1:
                section_size = len(feature_map) // self.num_point
                feature_map = torch.cat([feature_map[i*section_size: (i+1)*section_size] for i in range(self.num_point)], dim=1)

            ret_maps.append(feature_map)

        batch_dict['roi_features'] = ret_maps #ret_maps[0].shape = [NMX_MAX_size, -]
        return batch_dict