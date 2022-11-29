from pcdet.models.dense_heads.bev_feature_extractor_v2 import BEVFeatureExtractorV2
from .detector3d_template_v2 import Detector3DTemplateV2
import numpy as np

class CenterPoint_PointPillar_SingelHead(Detector3DTemplateV2):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        batch_dict['spatial_features_stride']=1
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)
            
        pred_dicts, _ = self.post_processing_for_refine(batch_dict)
        rois, roi_scores, roi_labels = self.reorder_rois_for_refining(batch_dict['batch_size'], pred_dicts)
        batch_dict['rois'] = rois
        batch_dict['roi_scores'] = roi_scores
        batch_dict['roi_labels'] = roi_labels
        batch_dict['has_class_labels'] = True

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts = self.reorder_rois_for_nuscenes_eval(batch_dict['batch_size'], pred_dicts)
            pred_dicts, recall_dicts = self.post_processing_for_nuscenes(batch_dict) #for roi head
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss()
        loss = loss_rpn

        return loss, tb_dict, disp_dict