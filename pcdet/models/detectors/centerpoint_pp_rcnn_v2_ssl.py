import os

import torch
import copy

from pcdet.datasets.augmentor.augmentor_utils import *
from pcdet.ops.iou3d_nms import iou3d_nms_utils
from .detector3d_template_v2 import Detector3DTemplateV2
from .centerpoint_pp_rcnn_v2 import CenterPoint_PointPillar_RCNNV2


class CenterPoint_PointPillar_RCNNV2_SSL(Detector3DTemplateV2):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        # something changes so need deep copy
        model_cfg_copy = copy.deepcopy(model_cfg)
        dataset_copy = copy.deepcopy(dataset)
        self.centerpoint_rcnn = CenterPoint_PointPillar_RCNNV2(model_cfg=model_cfg, num_class=num_class, dataset=dataset)

        self.centerpoint_rcnn_ema = CenterPoint_PointPillar_RCNNV2(model_cfg=model_cfg_copy, num_class=num_class, dataset=dataset_copy)
        for param in self.centerpoint_rcnn_ema.parameters():
            param.detach_()
        self.add_module('centerpoint_rcnn', self.centerpoint_rcnn)
        self.add_module('centerpoint_rcnn_ema', self.centerpoint_rcnn_ema)

        # self.module_list = self.build_networks()
        # self.module_list_ema = self.build_networks()
        self.thresh = model_cfg.THRESH
        self.sem_thresh = model_cfg.SEM_THRESH
        self.unlabeled_supervise_cls = model_cfg.UNLABELED_SUPERVISE_CLS
        self.unlabeled_supervise_refine = model_cfg.UNLABELED_SUPERVISE_REFINE
        self.unlabeled_weight = model_cfg.UNLABELED_WEIGHT
        self.no_nms = model_cfg.NO_NMS
        self.supervise_mode = model_cfg.SUPERVISE_MODE
        self.no_data = model_cfg.get('NO_DATA', None)

    def forward(self, batch_dict):
        if self.training:
            mask = batch_dict['mask'].view(-1)

            labeled_mask = torch.nonzero(mask).squeeze(1).long()
            unlabeled_mask = torch.nonzero(1-mask).squeeze(1).long()
            #
            if self.no_data is not None:
                if self.no_data == 'TL':
                    labeled_mask = labeled_mask[:1]
                elif self.no_data == 'TU':
                    unlabeled_mask = unlabeled_mask[:0]
                    self.unlabeled_supervise_refine = False
                    self.unlabeled_supervise_cls = False
                    self.unlabeled_supervise_box = False
                elif self.no_data == 'SL':
                    labeled_mask = labeled_mask[1:]
                else:
                    NotImplementedError('No timplement self.no_data')
            #
            batch_dict_ema = {}
            keys = list(batch_dict.keys())
            for k in keys:
                if k + '_ema' in keys:
                    continue
                if k.endswith('_ema'):
                    batch_dict_ema[k[:-4]] = batch_dict[k]
                else:
                    batch_dict_ema[k] = batch_dict[k]

            with torch.no_grad():
                vis_flag = False
                # self.centerpoint_rcnn_ema.eval()  # Important! must be in train mode
                for cur_module in self.centerpoint_rcnn_ema.module_list:
                    if str(cur_module) == "BEVFeatureExtractorV2()" or str(cur_module) == "PVRCNNHead()":
                        pred_dicts, _ = self.post_processing_for_refine(batch_dict_ema) #centerpoint prediction box
                        # rois, roi_scores, roi_labels = self.reorder_rois_for_refining(batch_dict_ema['batch_size'], pred_dicts)
                        rois, roi_scores, roi_labels = self.reorder_rois_for_refining(batch_dict_ema['batch_size'], pred_dicts)
                        batch_dict_ema['rois'] = rois
                        batch_dict_ema['roi_scores'] = roi_scores
                        batch_dict_ema['roi_labels'] = roi_labels
                        batch_dict_ema['has_class_labels'] = True
                    try:
                        batch_dict_ema = cur_module(batch_dict_ema, disable_gt_roi_when_pseudo_labeling=True)
                    except:
                        batch_dict_ema = cur_module(batch_dict_ema)

                pred_dicts = self.centerpoint_rcnn_ema.post_process(batch_dict_ema) #test 1025
                rois, roi_scores, roi_labels = self.centerpoint_rcnn_ema.reorder_rois_for_refining(batch_dict_ema['batch_size'], pred_dicts)
                batch_dict_ema['rois'] = rois
                # batch_dict_ema['roi_scores'] = roi_scores #roi scores는 first-stage의 score를 사용
                batch_dict_ema['roi_labels'] = roi_labels
                batch_dict_ema['has_class_labels'] = True
                pred_dicts, recall_dicts = self.centerpoint_rcnn_ema.post_processing_for_roi_ssl_(batch_dict_ema)
                #pred_dicts, recall_dicts = self.centerpoint_rcnn_ema.post_processing_for_roi__(batch_dict_ema)

                # pred_dicts, recall_dicts = self.centerpoint_rcnn_ema.post_processing_ssl(batch_dict_ema,
                #                                                                 no_recall_dict=True, override_thresh=0.0, no_nms=self.no_nms)

                if vis_flag:
                    for batch in range(batch_dict_ema['gt_boxes'].squeeze(0).cpu().detach().numpy().shape[0]):
                        if batch != 0:
                            gt_=[]
                            #for GT box visualization in forward 
                            # where, xyz,lwh,heading
                            gt_box = batch_dict_ema['gt_boxes'].squeeze(0).cpu().detach().numpy()[batch]
                            #gt_box = [[gt_box[0][3], gt_box[0][4], gt_box[0][5]], gt_box[0][6], [gt_box[0][0], gt_box[0][1], gt_box[0][2]]] 
                            points = batch_dict_ema['points'].cpu().detach().numpy() 
                            pc_mask = (points[:, 0] == float(batch))
                            points = points[pc_mask]
                            np.save("/home/changwon/detection_task/SSOD/kakao/my_ssda_2/vis_in_model/pc/gt_{}.npy".format(batch_dict_ema['frame_id'][batch].item().split(".")[0]), points)
                            file = open("/home/changwon/detection_task/SSOD/kakao/my_ssda_2/vis_in_model/box/gt_{}.txt".format(batch_dict_ema['frame_id'][batch].item().split(".")[0]), "w")
                            with open("/home/changwon/detection_task/SSOD/kakao/my_ssda_2/vis_in_model/box/gt_{}.txt".format(batch_dict_ema['frame_id'][batch].item().split(".")[0]), "w") as f:
                                for num in range(gt_box.shape[0]):
                                    f.writelines("{},{},{},{},{},{},{},".format(gt_box[num][3],gt_box[num][4],gt_box[num][5],gt_box[num][6],gt_box[num][0],gt_box[num][1],gt_box[num][2]))
                                    gt_.append([gt_box[num][3],gt_box[num][4],gt_box[num][5],gt_box[num][6],gt_box[num][0],gt_box[num][1],gt_box[num][2]])
                            
                            # pred_box = batch_dict_ema['batch_box_preds_roi'].squeeze(0).cpu().detach().numpy()[batch]
                            # pred_box = pred_dicts[batch]['pred_boxes'].squeeze(0).cpu().detach().numpy()
                            pred_box = batch_dict_ema['rois'][batch].squeeze(0).cpu().detach().numpy()
                            with open("/home/changwon/detection_task/SSOD/kakao/my_ssda_2/vis_in_model/box/pred_{}.txt".format(batch_dict['frame_id'][batch].item().split(".")[0]), "w") as f:
                                for num in range(pred_box.shape[0]):
                                    f.writelines("{},{},{},{},{},{},{},".format(pred_box[num][3],pred_box[num][4],pred_box[num][5],pred_box[num][6],pred_box[num][0],pred_box[num][1],pred_box[num][2]))
                                    gt_.append([pred_box[num][3],pred_box[num][4],pred_box[num][5],pred_box[num][6],pred_box[num][0],pred_box[num][1],pred_box[num][2]])
                            #scene_viz(gt_box, points)
                            #token = batch_dict['metadata'][0]['token']
                            print(batch_dict_ema['frame_id'][batch].item().split(".")[0])

                pseudo_boxes = []
                pseudo_scores = []
                pseudo_labels = []
                max_box_num = batch_dict['gt_boxes'].shape[1]
                #max_box_num = batch_dict_ema['rois'].shape[1]
                max_pseudo_box_num = 0
                batch_size = batch_dict['batch_size']
                for ind in unlabeled_mask:
                    # pseudo_score = batch_dict_ema['roi_scores'][ind]#.detach()
                    # pseudo_box = batch_dict_ema['rois'][ind]#.detach()
                    # pseudo_label = batch_dict_ema['roi_labels'][ind]#.detach()
                    # pseudo_sem_score = batch_dict_ema['roi_ious'][ind]#.detach()

                    pseudo_score = pred_dicts[ind]['pred_scores'].clone().detach()
                    pseudo_box = pred_dicts[ind]['pred_boxes'].clone().detach()
                    pseudo_label = pred_dicts[ind]['pred_labels'].clone().detach()
                    # pseudo_sem_score = pred_dicts[ind]['pred_sem_scores']
                    pseudo_sem_score = pred_dicts[ind]['pred_sem_scores'].squeeze().clone().detach()

                    ind_ = []
                    for i, lab in enumerate(pseudo_label):
                        if lab != 0:
                            ind_.append(i)
                        else:
                            continue

                    rois = pseudo_box.new_zeros((len(ind_), pseudo_box.shape[1]))
                    roi_scores = pseudo_score.new_zeros((len(ind_)))
                    roi_labels = pseudo_label.new_zeros((len(ind_)))#.long
                    roi_ious = pseudo_sem_score.new_zeros((len(ind_)))

                    for ii, idx in enumerate(ind_):
                        rois[ii] = pseudo_box[idx]
                        roi_scores[ii] = pseudo_score[idx]
                        roi_labels[ii] = pseudo_label[idx]
                        roi_ious[ii] = pseudo_sem_score[idx]
                    
                    pseudo_score = roi_scores
                    pseudo_box = rois
                    pseudo_label = roi_labels
                    pseudo_sem_score = roi_ious

                    
                    if len(pseudo_label) == 0:
                        pseudo_boxes.append(pseudo_label.new_zeros((0, 8)).float())
                        continue

                    # conf_thresh = torch.tensor(self.thresh, device=pseudo_label.device).unsqueeze(
                    #     0).repeat(len(pseudo_label), 1).gather(dim=1, index=(pseudo_label-1).unsqueeze(-1))
                    # conf_thresh = torch.tensor(self.thresh, device=torch.cuda.current_device()).unsqueeze(
                    #     0).repeat(len(pseudo_label), 1).gather(dim=1, index=(pseudo_label).unsqueeze(-1))
                    
                    # conf_thresh = torch.tensor(self.thresh, device=torch.cuda.current_device()).unsqueeze(
                    #     0).repeat(len(pseudo_label), 1).gather(dim=1, index=(pseudo_label-1).unsqueeze(-1)).squeeze()
                    conf_thresh = torch.tensor(self.thresh, device=pseudo_label.device).unsqueeze(
                        0).repeat(len(pseudo_label), 1).gather(dim=1, index=(pseudo_label-1).unsqueeze(-1)).squeeze()
                    # valid_inds = pseudo_score > conf_thresh.squeeze()
                    valid_inds = pseudo_score > conf_thresh

                    #valid_inds = valid_inds * (pseudo_sem_score > torch.tensor(self.sem_thresh[0], device=torch.cuda.current_device()))
                    valid_inds = valid_inds * (pseudo_sem_score > self.sem_thresh[0])

                    pseudo_sem_score = pseudo_sem_score[valid_inds]
                    pseudo_box = pseudo_box[valid_inds]
                    pseudo_label = pseudo_label[valid_inds]

                    #  if len(valid_inds) > max_box_num:
                       #  _, inds = torch.sort(pseudo_score, descending=True)
                       #  inds = inds[:max_box_num]
                       #  pseudo_box = pseudo_box[inds]
                       #  pseudo_label = pseudo_label[inds]

                    pseudo_boxes.append(torch.cat([pseudo_box, pseudo_label.view(-1, 1).float()], dim=1))
                    if pseudo_box.shape[0] > max_pseudo_box_num:
                        max_pseudo_box_num = pseudo_box.shape[0]
                    # pseudo_scores.append(pseudo_score)
                    # pseudo_labels.append(pseudo_label)

                max_box_num = batch_dict['gt_boxes'].shape[1]

                # assert max_box_num >= max_pseudo_box_num
                ori_unlabeled_boxes = batch_dict['gt_boxes'][unlabeled_mask, ...]

                if max_box_num >= max_pseudo_box_num:
                    for i, pseudo_box in enumerate(pseudo_boxes):
                        diff = max_box_num - pseudo_box.shape[0]
                        if diff > 0:
                            pseudo_box = torch.cat([pseudo_box, torch.zeros((diff, 8), device=pseudo_box.device)], dim=0)
                        batch_dict['gt_boxes'][unlabeled_mask[i]] = pseudo_box
                else:
                    ori_boxes = batch_dict['gt_boxes']
                    new_boxes = torch.zeros((ori_boxes.shape[0], max_pseudo_box_num, ori_boxes.shape[2]),
                                            device=ori_boxes.device)
                    for i, inds in enumerate(labeled_mask):
                        diff = max_pseudo_box_num - ori_boxes[inds].shape[0]
                        new_box = torch.cat([ori_boxes[inds], torch.zeros((diff, 8), device=ori_boxes[inds].device)], dim=0)
                        new_boxes[inds] = new_box
                    for i, pseudo_box in enumerate(pseudo_boxes):

                        diff = max_pseudo_box_num - pseudo_box.shape[0]
                        if diff > 0:
                            pseudo_box = torch.cat([pseudo_box, torch.zeros((diff, 8), device=pseudo_box.device)], dim=0)
                        new_boxes[unlabeled_mask[i]] = pseudo_box
                    batch_dict['gt_boxes'] = new_boxes

                batch_dict['gt_boxes'][unlabeled_mask, ...] = random_flip_along_x_bbox(
                    batch_dict['gt_boxes'][unlabeled_mask, ...], batch_dict['flip_x'][unlabeled_mask, ...]
                )

                batch_dict['gt_boxes'][unlabeled_mask, ...] = random_flip_along_y_bbox(
                    batch_dict['gt_boxes'][unlabeled_mask, ...], batch_dict['flip_y'][unlabeled_mask, ...]
                )

                batch_dict['gt_boxes'][unlabeled_mask, ...] = global_rotation_bbox(
                    batch_dict['gt_boxes'][unlabeled_mask, ...], batch_dict['rot_angle'][unlabeled_mask, ...]
                )

                batch_dict['gt_boxes'][unlabeled_mask, ...] = global_scaling_bbox(
                    batch_dict['gt_boxes'][unlabeled_mask, ...], batch_dict['scale'][unlabeled_mask, ...]
                )

                pseudo_ious = []
                pseudo_accs = []
                pseudo_fgs = []
                for i, ind in enumerate(unlabeled_mask):
                    'statistics'
                    anchor_by_gt_overlap = iou3d_nms_utils.boxes_iou3d_gpu(
                        batch_dict['gt_boxes'][ind, ...][:, 0:7],
                        ori_unlabeled_boxes[i, :, 0:7])
                    cls_pseudo = batch_dict['gt_boxes'][ind, ...][:, 7]
                    unzero_inds = torch.nonzero(cls_pseudo).squeeze(1).long()
                    cls_pseudo = cls_pseudo[unzero_inds]
                    if len(unzero_inds) > 0:
                        iou_max, asgn = anchor_by_gt_overlap[unzero_inds, :].max(dim=1)
                        pseudo_ious.append(iou_max.unsqueeze(0))
                        acc = (ori_unlabeled_boxes[i][:, 7].gather(dim=0, index=asgn) == cls_pseudo).float().mean()
                        pseudo_accs.append(acc.unsqueeze(0))
                        fg = (iou_max > 0.3).float().sum(dim=0, keepdim=True) / len(unzero_inds)

                        sem_score_fg = (pseudo_sem_score[unzero_inds] * (iou_max > 0.3).float()).sum(dim=0, keepdim=True) \
                                       / torch.clamp((iou_max > 0.3).float().sum(dim=0, keepdim=True), min=1.0)
                        sem_score_bg = (pseudo_sem_score[unzero_inds] * (iou_max < 0.3).float()).sum(dim=0, keepdim=True) \
                                       / torch.clamp((iou_max < 0.3).float().sum(dim=0, keepdim=True), min=1.0)
                        pseudo_fgs.append(fg)

                        'only for 100% label'
                        if self.supervise_mode >= 1:
                            filter = iou_max > 0.3
                            asgn = asgn[filter]
                            batch_dict['gt_boxes'][ind, ...][:] = torch.zeros_like(batch_dict['gt_boxes'][ind, ...][:])
                            batch_dict['gt_boxes'][ind, ...][:len(asgn)] = ori_unlabeled_boxes[i, :].gather(dim=0, index=asgn.unsqueeze(-1).repeat(1, 8))

                            if self.supervise_mode == 2:
                                batch_dict['gt_boxes'][ind, ...][:len(asgn), 0:3] += 0.1 * torch.randn((len(asgn), 3), device=iou_max.device) * \
                                                                                     batch_dict['gt_boxes'][ind, ...][
                                                                                     :len(asgn), 3:6]
                                batch_dict['gt_boxes'][ind, ...][:len(asgn), 3:6] += 0.1 * torch.randn((len(asgn), 3), device=iou_max.device) * \
                                                                                     batch_dict['gt_boxes'][ind, ...][
                                                                                     :len(asgn), 3:6]
                    else:
                        ones = torch.ones((1), device=unlabeled_mask.device)
                        sem_score_fg = ones
                        sem_score_bg = ones
                        pseudo_ious.append(ones)
                        pseudo_accs.append(ones)
                        pseudo_fgs.append(ones)

            if vis_flag:
                for batch in range(batch_dict['gt_boxes'].squeeze(0).cpu().detach().numpy().shape[0]):
                    if batch != 0: #pseudo label save
                        gt_=[]
                        #for GT box visualization in forward 
                        # where, xyz,lwh,heading
                        gt_box = batch_dict['gt_boxes'].squeeze(0).cpu().detach().numpy()[batch]
                        #gt_box = [[gt_box[0][3], gt_box[0][4], gt_box[0][5]], gt_box[0][6], [gt_box[0][0], gt_box[0][1], gt_box[0][2]]] 
                        points = batch_dict['points'].cpu().detach().numpy() 
                        pc_mask = (points[:, 0] == float(batch))
                        points = points[pc_mask]
                        np.save("/home/changwon/detection_task/SSOD/kakao/my_ssda_2/vis_in_model/pc/pseudo_{}.npy".format(batch_dict['frame_id'][batch].item().split(".")[0]), points)
                        file = open("/home/changwon/detection_task/SSOD/kakao/my_ssda_2/vis_in_model/box/pseudo_{}.txt".format(batch_dict['frame_id'][batch].item().split(".")[0]), "w")
                        with open("/home/changwon/detection_task/SSOD/kakao/my_ssda_2/vis_in_model/box/pseudo_{}.txt".format(batch_dict['frame_id'][batch].item().split(".")[0]), "w") as f:
                            for num in range(gt_box.shape[0]):
                                f.writelines("{},{},{},{},{},{},{},".format(gt_box[num][3],gt_box[num][4],gt_box[num][5],gt_box[num][6],gt_box[num][0],gt_box[num][1],gt_box[num][2]))
                                gt_.append([gt_box[num][3],gt_box[num][4],gt_box[num][5],gt_box[num][6],gt_box[num][0],gt_box[num][1],gt_box[num][2]])
                    
                        print(batch_dict['frame_id'][batch].item().split(".")[0])

            for cur_module in self.centerpoint_rcnn.module_list:
                if str(cur_module) == "BEVFeatureExtractorV2()" or str(cur_module) == "PVRCNNHead()":
                        pred_dicts, _ = self.post_processing_for_refine(batch_dict)
                        rois, roi_scores, roi_labels = self.reorder_rois_for_refining(batch_dict['batch_size'], pred_dicts)
                        batch_dict['rois'] = rois
                        batch_dict['roi_scores'] = roi_scores
                        batch_dict['roi_labels'] = roi_labels
                        batch_dict['has_class_labels'] = True
                batch_dict = cur_module(batch_dict)

            pred_dicts = self.post_process(batch_dict) #test 1025
            rois, roi_scores, roi_labels = self.reorder_rois_for_refining(batch_dict['batch_size'], pred_dicts)
            batch_dict['rois'] = rois
            #batch_dict['roi_scores'] = roi_scores
            batch_dict['roi_labels'] = roi_labels
            batch_dict['has_class_labels'] = True
            pred_dicts, recall_dicts = self.post_processing_for_roi__(batch_dict)

            disp_dict = {}
            loss_rpn_cls, loss_rpn_box, tb_dict = self.centerpoint_rcnn.dense_head.get_loss_ssl(scalar=False)
            #loss_point, tb_dict = self.centerpoint_rcnn.point_head.get_loss(tb_dict, scalar=False)
            loss_rcnn_cls, loss_rcnn_box, tb_dict = self.centerpoint_rcnn.roi_head.get_loss(tb_dict, scalar=False)

            if not self.unlabeled_supervise_cls:
                loss_rpn_cls = loss_rpn_cls[labeled_mask, ...].sum()
            else:
                loss_rpn_cls = loss_rpn_cls[labeled_mask, ...].sum() + loss_rpn_cls[unlabeled_mask, ...].sum() * self.unlabeled_weight

            loss_rpn_box = loss_rpn_box[labeled_mask, ...].sum() + loss_rpn_box[unlabeled_mask, ...].sum() * self.unlabeled_weight
            #loss_point = loss_point[labeled_mask, ...].sum()
            loss_rcnn_cls = loss_rcnn_cls[labeled_mask, ...].sum()

            if not self.unlabeled_supervise_refine:
                loss_rcnn_box = loss_rcnn_box[labeled_mask, ...].sum()
            else:
                loss_rcnn_box = loss_rcnn_box[labeled_mask, ...].sum() + loss_rcnn_box[unlabeled_mask, ...].sum() * self.unlabeled_weight

            # loss = loss_rpn_cls + loss_rpn_box + loss_point + loss_rcnn_cls + loss_rcnn_box
            loss = loss_rpn_cls + loss_rpn_box + loss_rcnn_cls + loss_rcnn_box
            tb_dict_ = {}
            for key in tb_dict.keys():
                if 'loss' in key:
                    tb_dict_[key+"_labeled"] = tb_dict[key][labeled_mask, ...].sum()
                    tb_dict_[key + "_unlabeled"] = tb_dict[key][unlabeled_mask, ...].sum()
                elif 'acc' in key:
                    tb_dict_[key+"_labeled"] = tb_dict[key][labeled_mask, ...].sum()
                    tb_dict_[key + "_unlabeled"] = tb_dict[key][unlabeled_mask, ...].sum()
                elif 'point_pos_num' in key:
                    tb_dict_[key + "_labeled"] = tb_dict[key][labeled_mask, ...].sum()
                    tb_dict_[key + "_unlabeled"] = tb_dict[key][unlabeled_mask, ...].sum()
                else:
                    tb_dict_[key] = tb_dict[key]

            tb_dict_['pseudo_ious'] = torch.cat(pseudo_ious, dim=0).mean()
            tb_dict_['pseudo_accs'] = torch.cat(pseudo_accs, dim=0).mean()
            tb_dict_['sem_score_fg'] = sem_score_fg.mean()
            tb_dict_['sem_score_bg'] = sem_score_bg.mean()

            tb_dict_['max_box_num'] = max_box_num
            tb_dict_['max_pseudo_box_num'] = max_pseudo_box_num

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict_, disp_dict

        else:
            for cur_module in self.centerpoint_rcnn.module_list:
                if str(cur_module) == "BEVFeatureExtractorV2()" or str(cur_module) == "PVRCNNHead()":
                    pred_dicts, _ = self.post_processing_for_refine(batch_dict)
                    rois, roi_scores, roi_labels = self.reorder_rois_for_refining(batch_dict['batch_size'], pred_dicts)
                    batch_dict['rois'] = rois
                    batch_dict['roi_scores'] = roi_scores
                    batch_dict['roi_labels'] = roi_labels
                    batch_dict['has_class_labels'] = True
                batch_dict = cur_module(batch_dict)

            # pred_dicts, recall_dicts = self.centerpoint_rcnn.post_processing(batch_dict)
            # pred_dicts, recall_dicts = self.post_processing_for_roi(batch_dict) #roi head의 결과
            #pred_dicts, recall_dicts = self.post_processing(batch_dict)
            #pred_dicts, recall_dicts = self.post_processing_ssl(batch_dict)
            # pred_dicts, recall_dicts = self.centerpoint_rcnn.post_processing_for_roi_ssl(batch_dict)

            pred_dicts = self.post_process(batch_dict) #test 1025
            rois, roi_scores, roi_labels = self.reorder_rois_for_refining(batch_dict['batch_size'], pred_dicts)
            batch_dict['rois'] = rois
            #batch_dict['roi_scores'] = roi_scores
            batch_dict['roi_labels'] = roi_labels
            batch_dict['has_class_labels'] = True
            pred_dicts, recall_dicts = self.post_processing_for_roi__(batch_dict)

            return pred_dicts, recall_dicts, {}

    def get_supervised_training_loss(self):
        disp_dict = {}
        loss_rpn, tb_dict = self.dense_head.get_loss()
        loss_point, tb_dict = self.point_head.get_loss(tb_dict)
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)

        loss = loss_rpn + loss_point + loss_rcnn
        return loss, tb_dict, disp_dict

    def update_global_step(self):
        self.global_step += 1
        alpha = 0.999
        # Use the true average until the exponential average is more correct
        alpha = min(1 - 1 / (self.global_step + 1), alpha)
        for ema_param, param in zip(self.centerpoint_rcnn_ema.parameters(), self.centerpoint_rcnn.parameters()):
            ema_param.data.mul_(alpha).add_((1 - alpha) * param.data)

    def load_params_from_file(self, filename, logger, to_cpu=False):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        model_state_disk = checkpoint['model_state']

        if 'version' in checkpoint:
            logger.info('==> Checkpoint trained from version: %s' % checkpoint['version'])

        update_model_state = {}
        for key, val in model_state_disk.items():
            new_key = 'centerpoint_rcnn.' + key
            if new_key in self.state_dict() and self.state_dict()[new_key].shape == model_state_disk[key].shape:
                update_model_state[new_key] = val
                # logger.info('Update weight %s: %s' % (key, str(val.shape)))
            new_key = 'centerpoint_rcnn_ema.' + key
            if new_key in self.state_dict() and self.state_dict()[new_key].shape == model_state_disk[key].shape:
                update_model_state[new_key] = val
            new_key = key
            if new_key in self.state_dict() and self.state_dict()[new_key].shape == model_state_disk[key].shape:
                update_model_state[new_key] = val

        state_dict = self.state_dict()
        state_dict.update(update_model_state)
        self.load_state_dict(state_dict)

        for key in state_dict:
            if key not in update_model_state:
                logger.info('Not updated weight %s: %s' % (key, str(state_dict[key].shape)))

        logger.info('==> Done (loaded %d/%d)' % (len(update_model_state), len(self.state_dict())))
