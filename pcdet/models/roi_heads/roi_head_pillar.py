# ------------------------------------------------------------------------------
# Portions of this code are from
# OpenPCDet (https://github.com/open-mmlab/OpenPCDet)
# Licensed under the Apache License.
# ------------------------------------------------------------------------------

from torch import batch_norm
import torch.nn as nn
import torch 
# from .roi_head_template import RoIHeadTemplate
from .roi_head_template_centerpoint_pointpillar import RoIHeadTemplate_CenterPoint_PointPillar

# from det3d.core import box_torch_ops

# from ..registry import ROI_HEAD

# @ROI_HEAD.register_module
class RoIHeadPillar(RoIHeadTemplate_CenterPoint_PointPillar):
    def __init__(self, input_channels, model_cfg, num_class=1, code_size=7, add_box_param=False, test_cfg=None):
        super().__init__(num_class=num_class, model_cfg=model_cfg)
        self.model_cfg = model_cfg
        self.test_cfg = test_cfg 
        self.code_size = code_size
        self.add_box_param = add_box_param

        # pre_channel = 384 #input_channels
        #pre_channel = 1920 #input_channels #for dynpillar
        pre_channel = 1280 #for pillarnet

        shared_fc_list = []
        for k in range(0, self.model_cfg.SHARED_FC.__len__()):
            shared_fc_list.extend([
                nn.Conv1d(pre_channel, self.model_cfg.SHARED_FC[k], kernel_size=1, bias=False),
                nn.BatchNorm1d(self.model_cfg.SHARED_FC[k]),
                nn.ReLU()
            ])
            pre_channel = self.model_cfg.SHARED_FC[k]

            if k != self.model_cfg.SHARED_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                shared_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))

        self.shared_fc_layer = nn.Sequential(*shared_fc_list)

        self.cls_layers = self.make_fc_layers(
            input_channels=pre_channel, output_channels=self.num_class, fc_list=self.model_cfg.CLS_FC
        )
        self.reg_layers = self.make_fc_layers(
            input_channels=pre_channel,
            output_channels=code_size,
            fc_list=self.model_cfg.REG_FC
        )
        self.init_weights(weight_init='xavier')

    def init_weights(self, weight_init='xavier'):
        if weight_init == 'kaiming':
            init_func = nn.init.kaiming_normal_
        elif weight_init == 'xavier':
            init_func = nn.init.xavier_normal_
        elif weight_init == 'normal':
            init_func = nn.init.normal_
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.reg_layers[-1].weight, mean=0, std=0.001)
    
    def reorder_first_stage_pred_and_feature(self, batch_dict):
        batch_size = batch_dict['batch_size']
        box_length = batch_dict['final_box_dicts'][0]['pred_boxes'].shape[1] 
        features = batch_dict['roi_features']
        feature_vector_length = features[0].shape[-1] #sum([feat[0].shape[-1] for feat in features])
        NMS_POST_MAXSIZE= batch_dict['rois'].shape[1] #3000

        rois = batch_dict['rois'].new_zeros((batch_size, 
            NMS_POST_MAXSIZE, box_length 
        ))
        roi_scores = batch_dict['roi_scores'].new_zeros((batch_size,
            NMS_POST_MAXSIZE
        ))
        roi_labels = batch_dict['roi_labels'].new_zeros((batch_size,
            NMS_POST_MAXSIZE), dtype=torch.long
        )
        roi_features = features[0].new_zeros((batch_size, 
            NMS_POST_MAXSIZE, feature_vector_length 
        ))

        for i in range(batch_size):
            num_obj = features[i].shape[0]
            # basically move rotation to position 6, so now the box is 7 + C . C is 2 for nuscenes to
            # include velocity target

            box_preds = batch_dict['rois'][i]

            if box_length == 9:
                # x, y, z, w, l, h, rotation_y, velocity_x, velocity_y
                box_preds = box_preds[:, [0, 1, 2, 3, 4, 5, 8, 6, 7]]

            rois[i, :num_obj] = box_preds[:num_obj]
            roi_labels[i, :num_obj] = batch_dict['roi_labels'][i, :num_obj] #+ 1
            roi_scores[i, :num_obj] = batch_dict['roi_scores'][i, :num_obj]
            roi_features[i, :num_obj] = features[i] #torch.cat([feat for feat in features[i]], dim=-1)

        batch_dict['rois'] = rois 
        batch_dict['roi_labels'] = roi_labels 
        batch_dict['roi_scores'] = roi_scores  
        batch_dict['roi_features'] = roi_features

        batch_dict['has_class_labels']= True 

        return batch_dict 

    def forward(self, batch_dict, training=True):
        """
        :param input_data: input dict
        :return:
        """
        batch_dict = self.reorder_first_stage_pred_and_feature(batch_dict)
        if training:
            targets_dict = self.assign_targets(batch_dict)
            batch_dict['rois'] = targets_dict['rois']
            batch_dict['roi_labels'] = targets_dict['roi_labels']
            batch_dict['roi_features'] = targets_dict['roi_features']
            batch_dict['roi_scores'] = targets_dict['roi_scores']

        # RoI aware pooling
        if self.add_box_param:
            batch_dict['roi_features'] = torch.cat([batch_dict['roi_features'], batch_dict['rois'], batch_dict['roi_scores'].unsqueeze(-1)], dim=-1)

        pooled_features = batch_dict['roi_features'].reshape(-1, 1,
            batch_dict['roi_features'].shape[-1]).contiguous()  # (BxN, 1, C)

        batch_size_rcnn = pooled_features.shape[0]
        pooled_features = pooled_features.permute(0, 2, 1).contiguous() # (BxN, C, 1)

        shared_features = self.shared_fc_layer(pooled_features.view(batch_size_rcnn, -1, 1))
        rcnn_cls = self.cls_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, 1 or 2)
        rcnn_reg = self.reg_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, C)

        if not training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=batch_dict['batch_size'], rois=batch_dict['rois'], cls_preds=rcnn_cls, box_preds=rcnn_reg
            )
            batch_dict['batch_cls_preds'] = batch_cls_preds
            batch_dict['batch_box_preds'] = batch_box_preds
            batch_dict['cls_preds_normalized'] = False
        else:
            targets_dict['rcnn_cls'] = rcnn_cls
            targets_dict['rcnn_reg'] = rcnn_reg

            self.forward_ret_dict = targets_dict
        
        return batch_dict        