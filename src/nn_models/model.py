from typing import Tuple

import pytorch_lightning as pl
import matplotlib.pyplot as plt
import math
import torch
import torchvision
import wandb
import os
import time
import numpy as np
import json
import copy

from datetime import datetime

import torch.nn.functional as F

# externals Models
from torchvision.models.detection.fcos import fcos_resnet50_fpn, FCOS, FCOSHead
from torchvision.models.detection.retinanet import retinanet_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn_v2, _default_anchorgen
from torchvision.models.detection.rpn import RPNHead, RegionProposalNetwork
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.generalized_rcnn import GeneralizedRCNN 
from torchvision.models.detection import FCOS_ResNet50_FPN_Weights, RetinaNet_ResNet50_FPN_V2_Weights, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models import ResNet50_Weights
from torch import Tensor, nn
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR, ReduceLROnPlateau
from ultralytics import YOLO
from torchvision.ops import box_area

from datamodule import DataModule 
from plot_data import fig2rgb_array
from metrics import MetricModule
from log import get_wandb_image_with_labels, get_wandb_image_with_labels_background_unknown_known, get_wandb_image_with_labels_target_known_unknown_background, get_wandb_image_with_proposal, get_wandb_box
from postprocessing_predictions import postprocess_predictions, extract_batched_predictions_with_nms
from postprocessing_predictions import set_tags, set_tags_opti, seperate_predictions_with_threshold_score, seperate_predictions_into_known_and_unknown, get_only_known_targets, get_only_unknown_targets, get_only_background_targets

from nn_models.fcos import my_forward, my_eager_outputs, my_compute_loss, my_head_compute_loss, my_postprocess_detections
from nn_models.my_rpn import compute_loss_iou_for_objecteness, assign_targets_to_anchors_iou, filter_proposals_iou, rpn_forward_that_save_infos_to_be_log, filter_proposals_that_save_infos_to_be_log, my_rcnn_forward, assign_targets_to_anchors_iou_oro, compute_loss_oro_iou_for_objecteness, rpn_forward_oro_that_save_infos_to_be_log, my_rcnn_forward_with_oro, filter_proposals_iou_with_oro, RPNHead_IoU_ORO, RPNHead_IoU
from nn_models.my_rpn import RPNHead_IoU_ORO, compute_loss_oro_iou_for_objecteness_with_negative_sample 
from nn_models.my_roi_heads import my_roi_head_postprocess_detections, _box_inter_union
from nn_models.my_roi_heads import _box_inter_union

from nn_models.my_rpn2 import Multiple_Head_RegionProposalNetwork, RPNHeadSingle
from nn_models.my_rpn import rpn_forward_wich_return_scores
from nn_models.my_faster_rcnn import rcnn_forward_only_rpn, rcnn_forward_with_proposals, eager_outputs_both_losses_and_detections


# CODE FROM https://github.com/hustvl/YOLOP/blob/main/lib/core/general.py#L98 use for yolop
def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

# CODE FROM https://github.com/hustvl/YOLOP/blob/main/lib/core/general.py#L98 use for yolop
def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False):
    """Performs Non-Maximum Suppression (NMS) on inference results

    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """

    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output

def convert_yolop_prediction_format(predictions):

    # from list of yolop results to list of dict with "boxes", "scores", "labels", "score_classe", "score_centerness"
    # boxes is a tensor of (n, 4)
    # labels and scores is a tensor of n
    # doc for yolop results is : https://github.com/hustvl/YOLOP/blob/main/lib/core/general.py
    # the prediction input is a list of n * 6 with 6 being [xyxy, confidence score, classe]


    converted_predictions = []
    for prediction in predictions:
        converted_prediction = {}
        converted_prediction["boxes"] = prediction[:, :4]
        converted_prediction["labels"] = prediction[:, 5].int()
        converted_prediction["scores"] = prediction[:, 4]

        converted_predictions.append(converted_prediction)

    return converted_predictions
                   

def convert_yolo_prediction_format(predictions):

    # from list of yolo results to list of dict with "boxes", "scores", "labels", "score_classe", "score_centerness"
    # boxes is a tensor of (n, 4)
    # labels and scores is a tensor of n
    # doc for yolo results is : https://docs.ultralytics.com/reference/engine/results/#ultralytics.engine.results.Boxes and https://docs.ultralytics.com/reference/engine/results/#ultralytics.engine.results.BaseTensor.to 


    converted_predictions = []
    for prediction in predictions:
        converted_prediction = {}
        converted_prediction["boxes"] = prediction.boxes.xyxy
        converted_prediction["labels"] = prediction.boxes.cls.int()
        converted_prediction["scores"] = prediction.boxes.conf

        converted_predictions.append(converted_prediction)

    return converted_predictions
                    
"""

Create a pytorch lightning module with fcos model from pytorch

"""
class Model(pl.LightningModule):

    def init_modified_fcos(self, cfg_model):

        # Create pytorch implementation of fcos
        if cfg_model.pretrained:
            self.model = fcos_resnet50_fpn(pretrained=True, pretrained_backbone=True)
        else:
            self.model = fcos_resnet50_fpn(num_classes=self.num_classes, _skip_resize=True)

        # Change pytorch implementation toward mine
        FCOS.forward = my_forward
        FCOS.eager_outputs = my_eager_outputs # to get model output at the same time than the losses at training time
        FCOS.compute_loss = my_compute_loss
        FCOS.postprocess_detections = my_postprocess_detections
        FCOSHead.compute_loss = my_head_compute_loss

        # Add parameters to the fcos model to adapte my changes
        self.model.enable_unknown = cfg_model.enable_unknown

        self.model.class_unknown_threshold_inf = self.threshold_score_centerness_unknown
        self.model.centerness_unknown_threshold_supp = self.threshold_score_remove

        self.model.center_sampling = self.center_sampling
        self.model.enable_semantics_classes = cfg_model.enable_semantics_classes
        self.model.enable_semantics_centerness = cfg_model.enable_semantics_centerness
        self.model.head.enable_semantics_classes = cfg_model.enable_semantics_classes
        self.model.head.enable_semantics_centerness = cfg_model.enable_semantics_centerness
        self.model.head.nb_classes = self.num_classes
        self.model.head.class_loss_reduction = cfg_model.class_loss_reduction
        self.model.head.class_loss_factor = cfg_model.class_loss_factor
        self.model.my_scoring_function_filter = cfg_model.my_scoring_function_filter
        self.model.score_thresh = cfg_model.object_score_threshold

    def __init__(self, cfg_model, cfg_metrics, classes_known, classes_names_pred=None, classes_names_gt=None, classes_names_merged=None, batch_size=1, show_image=False, use_custom_scores=False, semantic_segmentation_class_id_label=None ):

        super().__init__()

        self.use_custom_scores = use_custom_scores
        self.scores_cfg = cfg_metrics.scores
        self.considering_known_classes = cfg_metrics.considering_known_classes 
        self.classes_known = classes_known 

        self.percent_warmup_epochs  = 0.05
        self.scheduler = cfg_model.scheduler
        self.scheduler_patience = cfg_model.scheduler_patience
        self.lr_scheduler_interval = cfg_model.lr_scheduler_interval
        self.lr_scheduler_frequency = cfg_model.lr_scheduler_frequency

        self.metrics_module = MetricModule(self.scores_cfg, cfg_metrics, len(classes_names_pred), len(classes_names_gt), classes_known)
        self.train_metrics_module = MetricModule(self.scores_cfg, cfg_metrics, len(classes_names_pred), len(classes_names_gt), classes_known)

        self.num_classes = len(classes_names_pred)

        self.save_predictions_path = cfg_model.save_predictions_path
        self.save_predictions = cfg_model.save_predictions
        self.save_targets = cfg_model.save_targets
        self.is_save_anchors = cfg_model.is_save_anchors
        self.features_is_save = False

        self.threshold_score_centerness_unknown = cfg_model.threshold_score_centerness_unknown
        self.threshold_score_remove = cfg_model.threshold_score_remove 
        self.threshold_score_good = cfg_model.threshold_score_good 

        self.best_map = 0
        self.best_val_raw_coverage = 0

        self.load_path = cfg_model.load_path
        self.is_load = cfg_model.load
        if cfg_model.load:
            self.load(cfg_model.to_load)
            self.model_name_to_load = cfg_model.to_load.replace("/", "_")
            print("Model init is loaded")

        self.multiple_rpn_heads = False
        self.validation_calculate_losses = cfg_model.validation_calculate_losses
        self.test_calculate_losses = cfg_model.test_calculate_losses

        print("Is calculating losses on Validation set :", self.validation_calculate_losses)

        # FCOS
        if cfg_model.name == "fcos":
            if cfg_model.pretrained and not cfg_model.load:
                self.model = torchvision.models.detection.fcos_resnet50_fpn(weights=FCOS_ResNet50_FPN_Weights.COCO_V1)
                print("Using pretrained model on coco")
            else:
                raise NotImplementedError("Fcos modifed is implemented but not up to date with the rest of the code")
                print("Using My fcos modified model")
                self.center_sampling = cfg_model.center_sampling
                self.enable_semantics_centerness = cfg_model.enable_semantics_centerness
                self.log_heatmap = cfg_model.log_heatmap
                self.log_fcos_inside = cfg_model.log_fcos_inside 
                self.init_modified_fcos(cfg_model)

        # Retina Net
        elif cfg_model.name == "retina_net":
            if cfg_model.pretrained and not cfg_model.load:
                self.model = retinanet_resnet50_fpn_v2(weights=RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1)
            else:
                raise NotImplementedError("This function is not yet implemented")
                self.model = retinanet_resnet50_fpn_v2(num_classes=self.num_classes)

        # Faster-RCNN with rpn custom refacto
        elif cfg_model.name == "faster_rcnn":

            if cfg_model.pretrained and not cfg_model.load:
                self.model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1)
                print('model init is COCO detection weight')
            elif not cfg_model.load:
                self.model = fasterrcnn_resnet50_fpn_v2(weights_backbone=ResNet50_Weights.DEFAULT, num_classes=len(classes_names_pred))
                print("model init is Image Net default weight")

            
            if cfg_model.training.freeze_classifier:
                print("\nFreeze Classifier\n")
                for p in self.model.roi_heads.parameters():
                    p.requires_grad = False

            if cfg_model.freeze_backbone:
                # Freeze backbone
                print("\nFreeze Backbone \n")
                for p in self.model.backbone.parameters():
                    p.requires_grad = False


            # Change pytorch classes function to be able to access some data in the process
            GeneralizedRCNN.eager_outputs = eager_outputs_both_losses_and_detections
            RegionProposalNetwork.forward = rpn_forward_wich_return_scores

            self.only_rpn = cfg_model.only_rpn
            self.only_anchors = cfg_model.only_anchors
            if self.only_rpn:
                print("\nOnly rpn is activated in this faster rcnn\n")
                GeneralizedRCNN.forward = rcnn_forward_only_rpn
            else:
                print("\nrpn output is activated in this faster rcnn\n")
                GeneralizedRCNN.forward = rcnn_forward_with_proposals

            self.model.rpn.use_pure_negative_sample = False

            cfg_rpn = cfg_model.rpn
            self.multiple_rpn_heads = cfg_rpn.multiple_rpn_heads
            if isinstance(self.model.rpn, Multiple_Head_RegionProposalNetwork):
                self.multiple_rpn_heads = True
                self.model.rpn.objectness_score_used = cfg_rpn.objectness_score_used

            #print(f"{self.model.rpn.objectness_score_used = }")
            print(f"{self.multiple_rpn_heads = }")

            if cfg_rpn.multiple_rpn_heads:
                print("RPN config :", cfg_rpn)

                # Extract basic head of rpn
                out_channels = self.model.backbone.out_channels
                num_anchors_per_location = self.model.rpn.anchor_generator.num_anchors_per_location()[0]

                self.learning_rate_for_each_heads = {}

                rpn_heads = nn.ModuleDict({})

                rpn_heads["bbox_deltas"] = RPNHeadSingle(out_channels, num_anchors_per_location * 4, loss_function=None, beta=0, conv_depth=2)
                self.learning_rate_for_each_heads["bbox_deltas"] = cfg_rpn.bbox_deltas_lr

                # Add heads
                if cfg_model.rpn.bce:
                    #rpn_heads["BCE"] = self.model.rpn.head
                    rpn_heads["BCE"] = RPNHeadSingle(out_channels, num_anchors_per_location, loss_function=None, beta=0, conv_depth=2)
                    self.learning_rate_for_each_heads["BCE"] = cfg_rpn.bce_lr
                if cfg_model.rpn.bce_proposal:
                    rpn_heads["BCE_proposal"] = RPNHeadSingle(out_channels, num_anchors_per_location, loss_function=cfg_rpn.iou_loss_function, beta=cfg_rpn.iou_reg_beta, conv_depth=2)
                    self.learning_rate_for_each_heads["BCE_proposal"] = cfg_rpn.bce_proposal_lr
                if cfg_model.rpn.iou:
                    rpn_heads["IOU"] = RPNHeadSingle(out_channels, num_anchors_per_location, loss_function=cfg_rpn.iou_loss_function, beta=cfg_rpn.iou_reg_beta, conv_depth=3)
                    self.learning_rate_for_each_heads["IOU"] = cfg_rpn.iou_lr
                if cfg_model.rpn.iou_proposal:
                    rpn_heads["IOU_proposal"] = RPNHeadSingle(out_channels, num_anchors_per_location, loss_function=cfg_rpn.iou_loss_function, beta=cfg_rpn.iou_reg_beta, conv_depth=3)
                    self.learning_rate_for_each_heads["IOU_proposal"] = cfg_rpn.iou_proposal_lr
                if cfg_model.rpn.centerness:
                    rpn_heads["centerness"] = RPNHeadSingle(out_channels, num_anchors_per_location, loss_function=cfg_rpn.centerness_loss_function, beta=cfg_rpn.centerness_reg_beta, conv_depth=3)
                    self.learning_rate_for_each_heads["centerness"] = cfg_rpn.centerness_lr
                if cfg_model.rpn.centerness_proposal:
                    rpn_heads["centerness_proposal"] = RPNHeadSingle(out_channels, num_anchors_per_location, loss_function=cfg_rpn.centerness_loss_function, beta=cfg_rpn.centerness_reg_beta, conv_depth=3)
                    self.learning_rate_for_each_heads["centerness_proposal"] = cfg_rpn.centerness_proposal_lr



                state_dict = self.model.rpn.head.state_dict()

                bbox_pred_w = state_dict.pop("bbox_pred.weight")
                bbox_pred_b = state_dict.pop("bbox_pred.bias")
                state_dict["score.weight"] = state_dict.pop("cls_logits.weight")
                state_dict["score.bias"] = state_dict.pop("cls_logits.bias")

                rpn_heads["BCE"].load_state_dict(state_dict, strict=False)

                state_dict["score.weight"] = bbox_pred_w
                state_dict["score.bias"] = bbox_pred_b
                #state_dict["loss_function.running_var"] = rpn_heads["bbox_deltas"].state_dict()["loss_function.running_var"]
                #state_dict["loss_function.running_mean"] = rpn_heads["bbox_deltas"].state_dict()["loss_function.running_mean"]
                # pop loss function of dict

                rpn_heads["bbox_deltas"].load_state_dict(state_dict, strict=False)


                # Init the multy head RPN
                self.model.rpn = Multiple_Head_RegionProposalNetwork(self.model.rpn.anchor_generator,
                                                                     rpn_heads,
                                                                     cfg_model.rpn,
                                                                     self.model.rpn.proposal_matcher,
                                                                     self.model.rpn.fg_bg_sampler,
                                                                     self.model.rpn._pre_nms_top_n,
                                                                     self.model.rpn._post_nms_top_n,
                                                                     self.model.rpn.nms_thresh,
                                                                     self.model.rpn.score_thresh,
                                                                     )

            # Overwritte hyper param of faster rcnn :
            self.model.rpn._pre_nms_top_n["testing"] = cfg_model.rpn.pre_nms_top_n_test
            self.model.rpn._post_nms_top_n["testing"] = cfg_model.rpn.post_nms_top_n_test
            print("nb_proposals :", self.model.rpn._pre_nms_top_n["testing"])
            print("nb_proposals (should be the same as upward) :",self.model.rpn._post_nms_top_n["testing"])

            print(f"{isinstance(self.model.rpn, Multiple_Head_RegionProposalNetwork) = }")
            print(f"{type(self.model.rpn)}")
            """
            print(f"{self.model.rpn._post_nms_top_n = }")
            print(f"{self.model.rpn._pre_nms_top_n = }")
            print(f"{self.model.rpn.score_thresh = }")
            print(f"{self.model.rpn.nms_thresh= }")
            """


        # Faster-RCNN # TODO remove because DOES NOT GO IN THERE
        elif cfg_model.name == "faster_rcnn":
            exit()
            if cfg_model.pretrained and not cfg_model.load:
                self.model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1)
            elif not cfg_model.load:
                raise NotImplementedError("This function is not yet implemented")

            print(next(self.model.rpn.head.parameters()).is_cuda)

            self.model.roi_heads.detections_per_img = cfg_model.known_detections_per_img
            self.model.roi_heads.known_detections_per_img = cfg_model.known_detections_per_img
            self.model.roi_heads.unknown_detections_per_img = cfg_model.unknown_detections_per_img
            self.model.roi_heads.keep_background = cfg_model.keep_background
            self.model.roi_heads.with_oro = cfg_model.ai_oro
            self.model.roi_heads.unknown_intersection_with_known_threshold = cfg_model.unknown_intersection_with_known_threshold

            RoIHeads.postprocess_detections = my_roi_head_postprocess_detections
            GeneralizedRCNN.forward = my_rcnn_forward

            if cfg_model.keep_background:
                print("\nKeep the background class in post processing of roi head\n")


            if cfg_model.reset_rpn_weight:

                print("\n Reset rpn weight !\n")
                model_random = fasterrcnn_resnet50_fpn_v2()
                self.model.rpn = model_random.rpn

            if cfg_model.freeze_backbone:
                # Freeze backbone and roi heads:
                print("\nFreeze Backbone \n")
                for p in self.model.backbone.parameters():
                    p.requires_grad = False

            if cfg_model.freeze_heads:
                print("\nFreeze ROI Heads\n")
                for p in self.model.roi_heads.parameters():
                    p.requires_grad = False

            if cfg_model.freeze_rpn_heads:

                print("rpn head tyep :", type(self.model.rpn.head))
                if isinstance(self.model.rpn.head, RPNHead_IoU) and cfg_model.ai_oro:
                    print("Changing RPN HEAD IoU by ORO with keeping the weights")
                    out_channels = self.model.backbone.out_channels
                    rpn_anchor_generator = _default_anchorgen()
                    rpn_head = RPNHead_IoU_ORO(out_channels, rpn_anchor_generator.num_anchors_per_location()[0])

                    rpn_head.conv = self.model.rpn.head.conv
                    rpn_head.iou_pred = self.model.rpn.head.iou_pred
                    rpn_head.bbox_pred = self.model.rpn.head.bbox_pred
                    self.model.rpn.head = rpn_head


                print("\nFreeze RPN Heads\n")
                for p in self.model.rpn.head.conv.parameters():
                    p.requires_grad = False
                for p in self.model.rpn.head.iou_pred.parameters():
                    p.requires_grad = False
                for p in self.model.rpn.head.bbox_pred.parameters():
                    p.requires_grad = False

            #if cfg_model.log_rpn:
            RegionProposalNetwork.forward = rpn_forward_that_save_infos_to_be_log
            RegionProposalNetwork.filter_proposals = filter_proposals_that_save_infos_to_be_log
            rpn_batch_size_per_image = 256
            self.model.rpn.batch_size_per_image = rpn_batch_size_per_image
            self.model.rpn.percent_of_negative_sample = cfg_model.percent_of_negative_sample 
            self.model.rpn.add_best_iou_sample = cfg_model.add_best_iou_sample


            if cfg_model.ai_oro:
                print("\nChange objectness from basic classification to iou regression and oro\n")

                GeneralizedRCNN.forward = my_rcnn_forward_with_oro
                RegionProposalNetwork.filter_proposals = filter_proposals_iou_with_oro
                RegionProposalNetwork.assign_targets_to_anchors = assign_targets_to_anchors_iou_oro
                if cfg_model.negative_sample:
                    RegionProposalNetwork.compute_loss = compute_loss_oro_iou_for_objecteness_with_negative_sample
                else:
                    RegionProposalNetwork.compute_loss = compute_loss_oro_iou_for_objecteness
                RegionProposalNetwork.forward = rpn_forward_oro_that_save_infos_to_be_log

                if not cfg_model.load:
                    out_channels = self.model.backbone.out_channels
                    rpn_anchor_generator = _default_anchorgen()
                    rpn_head = RPNHead_IoU_ORO(out_channels, rpn_anchor_generator.num_anchors_per_location()[0])
                    self.model.rpn.head = rpn_head

            elif cfg_model.ai_iou: 

                print("\nChange objectness from basic classification to iou regression\n")

                RegionProposalNetwork.compute_loss = compute_loss_iou_for_objecteness
                RegionProposalNetwork.assign_targets_to_anchors = assign_targets_to_anchors_iou
                RegionProposalNetwork.filter_proposals = filter_proposals_iou
                RegionProposalNetwork.forward = rpn_forward_that_save_infos_to_be_log


            self.model.rpn._pre_nms_top_n["training"] = cfg_model.rpn_pre_nms_training_top_n #2000
            self.model.rpn._pre_nms_top_n["testing"] = cfg_model.rpn_pre_nms_testing_top_n#1000
            self.model.rpn._post_nms_top_n["training"] = cfg_model.rpn_post_nms_training_top_n#2000
            self.model.rpn._post_nms_top_n["testing"] = cfg_model.rpn_post_nms_testing_top_n#1000

            self.model.roi_heads.unknown_roi_head_background_classif_score_threshold = cfg_model.unknown_roi_head_background_classif_score_threshold 
            self.model.roi_heads.unknown_roi_head_oro_score_threshold = cfg_model.unknown_roi_head_oro_score_threshold 
            self.model.roi_heads.unknown_roi_head_iou_score_threshold = cfg_model.unknown_roi_head_iou_score_threshold 
            self.oro_score_threshold = cfg_model.unknown_roi_head_oro_score_threshold

        # Yolov5
        elif cfg_model.name == "yolov5":
            print("Using yolov5 model ")
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        
        # Yolov8
        elif cfg_model.name == "yolov8":
            print("Using yolov8 model ")
            self.model = YOLO('yolov8n.pt')
            self.dataset_path = cfg_model.dataset_path

        # Yolop
        elif cfg_model.name == "yolop":
            print("Using yolop model ")
            if cfg_model.pretrained and not cfg_model.load:
                self.model = torch.hub.load('hustvl/yolop', 'yolop', pretrained=True)
            else:
                raise NotImplementedError("This function is not yet implemented")

        else:
            print("[ERROR] Model name : " + cfg_model.name + " is not handle !")
            return 

        self.model.nb_classes = self.num_classes

        self.batch_size = batch_size
        print("Batch size :", batch_size)
        self.log("batch size", batch_size, batch_size=batch_size)
        self.show_image=show_image
        self.optimizer = cfg_model.optimizer
        self.learning_rate = cfg_model.learning_rate
        self.momentum = cfg_model.momentum
        self.weight_decay = cfg_model.weight_decay
        self.classes_names = classes_names_pred
        self.classes_names_gt = classes_names_gt
        self.classes_names_merged = classes_names_merged
        self.calcul_metric_train_each_nb_batch = cfg_model.calcul_metric_train_each_nb_batch
        self.save_each_epoch = cfg_model.save_each_epoch
        self.enable_unknown = cfg_model.enable_unknown
        self.nms_iou_threshold = cfg_model.nms_iou_threshold


        self.index = 0
        self.config = cfg_model
        self.model_name = cfg_model.name

        # Build dict with label associate to index for wandb
        self.class_id_label = {}
        for i, class_name in enumerate(self.classes_names):
            self.class_id_label[i] = class_name
        self.class_id_label[len(self.classes_names)] = "unknown"

        self.gt_class_id_label = {}
        for i, gt_class_name in enumerate(self.classes_names_gt):
            self.gt_class_id_label[i] = gt_class_name
        print("gt class", self.gt_class_id_label)

        self.merged_class_id_label = {}
        for i, merged_class_name in enumerate(self.classes_names_merged):
            self.merged_class_id_label[i] = merged_class_name
        print("merged id class", self.merged_class_id_label)

        self.semantic_segmentation_class_id_label = semantic_segmentation_class_id_label 

        wandb_alias = "offline" if (wandb.run == None or wandb.run.name == None) else wandb.run.name
        print("wandb :", wandb.run.name)
        self.save_path = cfg_model.save_path + wandb_alias + "_" + cfg_model.name + "_" + datetime.now().strftime("%Y-%m-%d_%H-%M") + "/"
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def load(self, model_name):
        print("\nLoad model : ", model_name)
        self.model = torch.load(self.load_path + model_name, map_location=self.device)
        print("End Loading model\n")

    def forward(self, x):
        return self.model(x)

    def sort_predictions(self, predictions, filter_with_oro=False, nms_unknown_inside_known=True):

        # Select only known
        background_predictions = []
        known_predictions = []
        unknown_predictions = []
        for prediction in predictions:
            nb_pred = len(prediction["labels"])

            known_prediction = {}
            unknown_prediction = {}
            background_prediction = {}

            #filter out only those predicted labels that are recognized and considered valid based on the known classes you have defined.
            over_known_threshold_score_mask = prediction["scores"] >= self.scores_cfg.threshold_score 
            under_minimum_score_mask = prediction["scores"] < self.scores_cfg.threshold_score_minimum 
            classes_known_mask = torch.tensor([label in self.classes_known for label in prediction["labels"]], dtype=torch.bool, device=prediction["labels"].device)
            classes_background_mask = prediction["labels"] == 0

            # Mask with scores and classes
            if self.config.open_low_known_classes_as_unknown:
                background_mask = under_minimum_score_mask
            else:
                background_mask = under_minimum_score_mask | (~classes_background_mask & ~over_known_threshold_score_mask)  # If removing known classes between 0.2 and 0.5
            known_mask = classes_known_mask & over_known_threshold_score_mask
            unknown_mask = ~(background_mask | known_mask)

            if self.config.filter_with_oro:
                # Put unknown that have oro < threshold to background 
                unknown_with_oro_too_low_mask = unknown_mask & (prediction["oro"] < self.oro_score_threshold) #TODO make oro threshold parametable

                #update mask
                unknown_mask = unknown_mask ^ unknown_with_oro_too_low_mask
                background_mask = background_mask | unknown_with_oro_too_low_mask


            if nms_unknown_inside_known and unknown_mask.any() and known_mask.any():
                inter, unions = _box_inter_union(prediction["boxes"][unknown_mask], prediction["boxes"][known_mask])
                area_unknown = box_area(prediction["boxes"][unknown_mask])
                max_inter_values, max_inter_inds = inter.max(dim=1)

                unknown_inside_known_mask = ((area_unknown * 0.4).int() <= (max_inter_values).int())
                if unknown_inside_known_mask.any():
                    background_mask[unknown_mask] = unknown_inside_known_mask
                    unknown_mask[unknown_mask.clone()] = ~unknown_inside_known_mask

            nms_iou_threshold = 0.5  #TODO make in config iou nms thrshold
            kept_boxes_from_nms = torchvision.ops.nms(prediction["boxes"][unknown_mask], prediction['scores'][unknown_mask], nms_iou_threshold)  # NMS
            if kept_boxes_from_nms.any():
                mask_kept_unknown = torch.zeros(unknown_mask.sum(), dtype=bool, device=unknown_mask.device)
                mask_kept_unknown[kept_boxes_from_nms] = True
                background_mask[unknown_mask] = ~mask_kept_unknown
                unknown_mask[unknown_mask.clone()] = mask_kept_unknown 

            for key, value in prediction.items():

                if not torch.is_tensor(value):
                    continue

                known_prediction[key] = value[known_mask]
                unknown_prediction[key] = value[unknown_mask]
                background_prediction[key] = value[background_mask]

            sum_pred = len(known_prediction["labels"]) + len(background_prediction["labels"]) + len(unknown_prediction["labels"])
            if nb_pred != sum_pred:
                print(f"[ERROR] Sorting between K, U and B gone wrong :( sum prediction ({sum_pred}) != nb pred ({nb_pred})")
                print("\n\nknown : ", known_prediction)
                print("\n\nunknown : ",unknown_prediction)
                print("\n\nBackground : ",background_prediction)
                print(f"[ERROR] Sorting between K, U and B gone wrong :( sum prediction ({sum_pred}) != nb pred ({nb_pred})")
                exit()


            known_predictions.append(known_prediction)
            unknown_predictions.append(unknown_prediction)
            background_predictions.append(background_prediction)

        return known_predictions, unknown_predictions, background_predictions

    def get_predictions(self, images, targets, batch_idx, training=False):

        losses = {}

        if "predictions" in targets[0]:
            predictions = []
            for target in targets:
                predictions.append(target["predictions"])
        else:
            if self.model_name == "yolov8":
                raise DeprecationWarning("This model is deprecated and should not be used. (old code that should be verified before use it again)")

                images_names = []
                for i, image in enumerate(images):
                    name = targets[i]["name"]
                    zeros_needed = 12 - len(name)
                    name = self.dataset_path + "images/" + "0" * zeros_needed + name + ".jpg"
                    images_names.append(name)
                
                predictions, losses = self.model(images_names, verbose=False)
                predictions = convert_yolo_prediction_format(predictions)

            elif self.model_name == "yolop":
                raise DeprecationWarning("This modle is deprecated and should not be used. (old code that should be verified before use it again)")
                predictions, da_seg_out, ll_seg_out = self.model(images)
                inference_output, train_out = predictions

                # TODO change hardcode value below !!!!
                predictions = non_max_suppression(inference_output, conf_thres=0.2, iou_thres=0.5)
                predictions = convert_yolop_prediction_format(predictions)

            elif self.model_name == "faster_rcnn":

                if self.only_rpn:
                    predictions, losses = self.model(images, targets)
                    proposals_predictions = None
                else:
                    losses, predictions, proposals_detections = self.model(images, targets)
            else:
                predictions = self.model(images)


        return predictions, losses, proposals_detections

    def log_rpn_sampling(self, transform_images, original_image_sizes, transform_targets, which_score="iou"):

        # TODO refacto to make it work with batch szie more than 1
        if which_score == "iou":
            sampled_anchors = [self.model.rpn.current_iou_anchors_sample]
            sampled_iou_targets = [self.model.rpn.current_iou_targets_sample]
            sampled_iou_predictions = [self.model.rpn.current_iou_pred_sample]
        elif which_score == "centerness":
            sampled_anchors = [self.model.rpn.current_centerness_anchors_sample]
            sampled_iou_targets = [self.model.rpn.current_centerness_targets_sample]
            sampled_iou_predictions = [self.model.rpn.current_centerness_pred_sample]

        else :
            sampled_anchors = [self.model.rpn.current_bce_anchors_sample]
            sampled_iou_targets = [self.model.rpn.current_bce_targets_sample]
            sampled_iou_predictions = [self.model.rpn.current_bce_pred_sample]

        # GT 
        log_targets = []
        for target in transform_targets:
            log_targets.append({"boxes": target["boxes"], "labels": torch.zeros(len(target["boxes"]), device=target["boxes"].device, dtype=int)})

        # objectness target
        anchors_target = []
        anchors_detection = []
        for anchor, target, prediction in zip(sampled_anchors, sampled_iou_targets, sampled_iou_predictions):
            anchors_target.append({"boxes": anchor, "labels": torch.zeros(len(anchor), device=anchor.device, dtype=int), "scores": target})
            anchors_detection.append({"boxes": anchor, "labels": torch.zeros(len(anchor), device=anchor.device, dtype=int), "scores": prediction})

        anchors_detections = self.model.transform.postprocess(anchors_detection, transform_images.image_sizes, original_image_sizes)
        anchors_target = self.model.transform.postprocess(anchors_target, transform_images.image_sizes, original_image_sizes)

        wandb_rpn_sampling_target = []
        wandb_rpn_sampling_predictions = []
        for i, image in enumerate(transform_images.tensors):
            wandb_rpn_sampling_target.append(get_wandb_image_with_labels(image, log_targets[i], anchors_target[i], {0: "target"}, {0: "gt"}))
            wandb_rpn_sampling_predictions.append(get_wandb_image_with_labels(image, log_targets[i], anchors_detections[i], {0: "prediction"}, {0: "gt"}))

        wandb.log({("Images/rpn_sampling_target" + which_score): wandb_rpn_sampling_target})
        wandb.log({("Images/rpn_sampling_predictions" + which_score): wandb_rpn_sampling_predictions})

        if self.config.ai_oro: 

            sampled_oro_targets = self.model.rpn.current_oro_targets 
            sampled_oro_predictions = self.model.rpn.current_oro_predictions
            oro_sampled_anchors = self.model.rpn.current_oro_sampled_anchors

            anchors_target_oro = []
            anchors_detection_oro = []
            for anchor, target, prediction in zip(oro_sampled_anchors, sampled_oro_targets, sampled_oro_predictions):
                anchors_target_oro.append({"boxes": anchor, "labels": torch.zeros(len(anchor), device=anchor.device, dtype=int), "scores": target})
                anchors_detection_oro.append({"boxes": anchor, "labels": torch.zeros(len(anchor), device=anchor.device, dtype=int), "scores": prediction})

            anchors_detections_oro = self.model.transform.postprocess(anchors_detection_oro, transform_images.image_sizes, original_image_sizes)
            anchors_target_oro = self.model.transform.postprocess(anchors_target_oro, transform_images.image_sizes, original_image_sizes)

            wandb_rpn_sampling_target_oro = []
            wandb_rpn_sampling_predictions_oro = []
            for i, image in enumerate(transform_images.tensors):
                wandb_rpn_sampling_target_oro.append(get_wandb_image_with_labels(image, log_targets[i], anchors_target_oro[i], {0: "target"}, {0: "gt"}))
                wandb_rpn_sampling_predictions_oro.append(get_wandb_image_with_labels(image, log_targets[i], anchors_detections_oro[i], {0: "prediction"}, {0: "gt"}))

            wandb.log({("Images/rpn_sampling_target_oro"): wandb_rpn_sampling_target_oro})
            wandb.log({("Images/rpn_sampling_predictions_oro"): wandb_rpn_sampling_predictions_oro})

    def get_rpn_detections(self, transform_images, original_image_sizes):

        proposals = self.model.rpn.current_proposals
        proposal_scores = self.model.rpn.current_scores
        anchors = self.model.rpn.current_anchors
        anchors = self.model.rpn.current_filtered_anchors

        proposals_detections = []
        for proposal, proposal_score in zip(proposals, proposal_scores):
            proposals_detections.append({"boxes": proposal, "labels": torch.zeros(len(proposal), device=proposal.device, dtype=int), "scores": proposal_score})
        proposals_detections = self.model.transform.postprocess(proposals_detections, transform_images.image_sizes, original_image_sizes)

        anchors_detections = []
        for anchor, proposal_score in zip(anchors, proposal_scores): #anchors_labels):
            anchors_detections.append({"boxes": anchor, "labels": torch.zeros(len(anchor), device=anchor.device, dtype=int), "scores": proposal_score})
        anchors_detections = self.model.transform.postprocess(anchors_detections, transform_images.image_sizes, original_image_sizes)

        return proposals_detections, anchors_detections


    def log_rpn_proposals(self, proposals_detections, anchors_detections, transform_images):

        # WANDB LOG object proposal
        wandb_object_proposal_images = []
        for i, image in enumerate(transform_images.tensors):
            wandb_object_proposal_images.append(get_wandb_image_with_labels(image, proposals_detections[i], anchors_detections[i], {0: "anchors"}, {0: "proposals"}))

        wandb.log({("Images/Objects_proposal"): wandb_object_proposal_images})


    # ------ TRAINNING ----------------------------------

    def on_train_epoch_start(self):
        self.train_metrics_module.reset()
        self.train_losses = [] 
        if self.multiple_rpn_heads:
            self.model.rpn.calculate_losses = True


    def training_step(self, batch, batch_idx):

        images, targets = batch
        batch_size = len(images)

        for target in targets:
            if len(target["boxes"]) == 0:
                #print("SKIP BATCH BECAUSE EMPTY TARGETS !!! :", target)
                return None
            if (target["boxes"][:, 0] >= target["boxes"][:, 2]).any() or (target["boxes"][:, 1] >= target["boxes"][:, 3]).any():
                return None


        #print(f"{targets[0]['name'] = }")

        # Remove class we can't predict:
        targets = get_only_known_targets(targets)

        if self.model_name == "fcos":
            (losses, predictions) = self.model(images, targets)

        elif self.model_name == "retina_net":
            # Forward + Loss 
            loss_dict = self.model(images, targets)
            self.log("losses", loss_dict)
            losses = sum(loss for loss in loss_dict.values())

        elif self.model_name == "faster_rcnn":

            # Apply transform on the SS OBD and get transfrom img for logging
            if self.config.ai_oro or self.config.perfect_oro: 
                for target in targets:
                    target["masks"] = target["semantic_segmentation_OBD"].unsqueeze(0)

            transform_images, transform_targets = self.model.transform(images, targets)

            if self.config.ai_oro or self.config.perfect_oro: 
                for target, transform_target in zip(targets, transform_targets):
                    target["semantic_segmentation_OBD"] = transform_target["masks"].squeeze(0)


            # Forward + Loss 
            if self.only_rpn:
                detections, loss_dict = self.model(images, targets)
                proposals_predictions = None
            else:
                loss_dict, detections, proposals_predictions = self.model(images, targets)
           
            self.train_losses.append(loss_dict)
            for loss_name, loss in loss_dict.items():
                self.log(loss_name, loss)
            losses = sum(loss for loss in loss_dict.values())
            self.log("sum_losses", losses)


            if not self.automatic_optimization: # Then their is multiple optimizer
                for optimizer in self.optimizers():
                    optimizer.zero_grad()
                self.manual_backward(losses)
                for optimizer in self.optimizers():
                    optimizer.step()


            self.evaluation_step(detections, images, targets, batch_idx, "Training", proposals_predictions=proposals_predictions)

            # log RPN 
            if batch_idx <= 6: # or batch_idx == 2:

                original_image_sizes: List[Tuple[int, int]] = []
                for img in images:
                    val = img.shape[-2:]
                    torch._assert(
                            len(val) == 2,
                            f"expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}",
                            )
                    original_image_sizes.append((val[0], val[1]))

                #proposals_detections , anchors_detections = self.get_rpn_detections(transform_images, original_image_sizes)
                #self.log_rpn_proposals(proposals_detections, anchors_detections, transform_images)
                if batch_size <= 6:
                    #self.log_rpn_sampling(transform_images, original_image_sizes, transform_targets, which_score="centerness")
                    #self.log_rpn_sampling(transform_images, original_image_sizes, transform_targets, which_score="iou")
                    #self.log_rpn_sampling(transform_images, original_image_sizes, transform_targets, which_score="bce")
                    image = self.model.rpn.current_image_pure_negative
                    sampled_anchors = self.model.rpn.current_iou_anchors_sample
                    sampled_targets = self.model.rpn.current_iou_targets_sample
                    sampling = {"boxes": sampled_anchors, "labels": torch.zeros(len(sampled_anchors), device=sampled_anchors.device, dtype=int), "scores":sampled_targets}
                    box_data = get_wandb_box(sampling, {0: "0"}, (image.shape[1], image.shape[2]), tags="tags_raw")
                    boxes={"ground_truth": {"box_data": box_data, "class_labels": {0: "0"}}}
                    wandb_rpn_sampling = wandb.Image(image, boxes=boxes)
                    wandb.log({("Images/rpn_sampling_iou"): wandb_rpn_sampling})


                    #  Log pure negative score percent object in boxes 
                    if self.model.rpn.use_pure_negative_sample:
                        image = self.model.rpn.current_image_pure_negative
                        sampled_anchors = self.model.rpn.current_anchors_pure_negative
                        sampled_scores = self.model.rpn.current_scores_percent_object_pure_negative
                        mask = self.model.rpn.current_mask_semantic_pure_negative


                        mask = np.clip(mask.cpu().numpy(), 0, 255)

                        pure_negative_sampling = {"boxes": sampled_anchors, "labels": torch.zeros(len(sampled_anchors), device=sampled_anchors.device, dtype=int), "scores":sampled_scores}
                        box_data = get_wandb_box(pure_negative_sampling, {0: "0"}, mask.shape, tags="tags_raw")

                        boxes={"ground_truth": {"box_data": box_data, "class_labels": {0: "0"}}}
                        masks = {"ground_truth": {"mask_data": mask, "class_labels": {0: "background", 1: "object"}}}
                        wandb_rpn_sampling_pure_negative = wandb.Image(image, boxes=boxes, masks=masks)
                        wandb.log({("Images/rpn_sampling_pure_negative_percent_object_score"): wandb_rpn_sampling_pure_negative})

                        sampled_anchors = self.model.rpn.current_iou_anchors_sample
                        sampled_targets = self.model.rpn.current_iou_targets_sample

                        sampling = {"boxes": sampled_anchors, "labels": torch.zeros(len(sampled_anchors), device=sampled_anchors.device, dtype=int), "scores":sampled_targets}
                        box_data = get_wandb_box(sampling, {0: "0"}, mask.shape, tags="tags_raw")

                        boxes={"ground_truth": {"box_data": box_data, "class_labels": {0: "0"}}}
                        masks = {"ground_truth": {"mask_data": mask, "class_labels": {0: "background", 1: "object"}}}
                        wandb_rpn_sampling = wandb.Image(image, boxes=boxes, masks=masks)
                        wandb.log({("Images/rpn_sampling_pure_negative"): wandb_rpn_sampling})

        else:
            raise Exception("not implemented")

        if self.current_epoch % self.save_each_epoch == 0 and self.current_epoch != 0:
            path = self.save_path + "epoch_" + str(self.current_epoch)
            torch.save(self.model, path)


        return losses

    def on_train_epoch_end(self):

        # Calculate the mean of each loss
        mean_losses = {key: sum(loss[key] for loss in self.train_losses) / len(self.train_losses) for key in self.train_losses[0]}
        # Log the mean losses
        for key, value in mean_losses.items():
            self.log("mean " + key, value)

        print()
        print("Training metrics :")
        wandb.log({"Train/metrics": self.train_metrics_module.get_wandb_metrics(with_print=True)})

    def on_train_end(self):

        torch.save(self.model, self.save_path + "Final")
    
    # ------ Evaluation ----------------------------------



    def evaluation_step(self, predictions, images, targets, batch_idx, loging_name, proposals_predictions, with_randoms=False):

        # Match predictions and targets on raw
        if self.metrics_module.is_metrics_on_raw_predictions and predictions != None and predictions != []: 

            # Apply nms on raw detection to not falsify metrics 
            nms_iou_threshold = 0.7 # Same as faster rcnn RPN  #TODO make in config iou nms threshold
            raw_nms_predictions = extract_batched_predictions_with_nms(predictions, nms_iou_threshold)
            #print(f"{raw_nms_predictions[0]['scores'].shape = }")

            for batch_index in range(len(targets)):
                set_tags_opti(raw_nms_predictions, targets, batch_index, self.scores_cfg.iou_threshold, considering_classes=False, tags_name="tags_raw", add_centerness=True)

        # Seperate targets and predictions between known and unknown 
        known_targets = get_only_known_targets(targets)
        unknown_targets = get_only_unknown_targets(targets)
        background_targets = get_only_background_targets(targets)


        if self.metrics_module.is_metrics_on_proposals_predictions and proposals_predictions != None: 

            nms_iou_threshold = 0.7 # Same as faster rcnn RPN  #TODO make in config iou nms threshold
            proposals_nms_predictions = extract_batched_predictions_with_nms(proposals_predictions, nms_iou_threshold)
            #print(f"{proposals_nms_predictions[0]['scores'].shape = }")

            for batch_index in range(len(targets)):
                set_tags_opti(proposals_nms_predictions, targets, batch_index, self.scores_cfg.iou_threshold, considering_classes=False, tags_name="tags_proposals")
                set_tags_opti(proposals_nms_predictions, known_targets, batch_index, self.scores_cfg.iou_threshold, considering_classes=False, tags_name="tags_proposals")
                set_tags_opti(proposals_nms_predictions, unknown_targets, batch_index, self.scores_cfg.iou_threshold, considering_classes=False, tags_name="tags_proposals")

        known_predictions, unknown_predictions, background_predictions = self.sort_predictions(predictions, filter_with_oro=self.config.filter_with_oro)

        # Match predictions and targets
        if predictions != None and predictions != []:
            for batch_index in range(len(targets)):

                set_tags_opti(known_predictions, known_targets, batch_index, self.scores_cfg.iou_threshold, self.considering_known_classes)
                set_tags_opti(unknown_predictions, unknown_targets, batch_index, self.scores_cfg.iou_threshold, considering_classes=False)
                set_tags_opti(known_predictions, unknown_targets, batch_index, self.scores_cfg.iou_threshold, considering_classes=False, tags_name="tags_KP_with_UT")


        # Update with predictions 
        if loging_name == "Training":
            if predictions != None and predictions != []:
                self.train_metrics_module.update(known_targets, unknown_targets, background_targets, known_predictions, unknown_predictions, targets)
                self.train_metrics_module.update_raw(known_targets, unknown_targets, raw_nms_predictions, targets)
            if self.metrics_module.is_metrics_on_proposals_predictions and proposals_predictions != None: 
                self.train_metrics_module.update_proposals(known_targets, unknown_targets, proposals_nms_predictions, targets)
        else:
            self.metrics_module.update(known_targets, unknown_targets, background_targets, known_predictions, unknown_predictions, targets)
            self.metrics_module.update_raw(known_targets, unknown_targets, raw_nms_predictions, targets)
            if self.metrics_module.is_metrics_on_proposals_predictions and proposals_predictions != None: 
                self.metrics_module.update_proposals(known_targets, unknown_targets, proposals_nms_predictions, targets)


        # Loging process
        target_known_unknown_and_background = []
        target_and_random_wandb_images = []
        edge_wandb_images = []

        log_A_OSE = False #TODO put it in config file
        if log_A_OSE and batch_idx <= 2: # or batch_idx == 2:
            A_OSE_wandb_images = []
            for i in range(len(images)):

                if known_predictions[i]["tags_KP_with_UT"].any():

                    A_OSE_wandb_images.append(get_wandb_image_with_labels_background_unknown_known(images[i],
                                                                                                (known_targets[i], unknown_targets[i]),
                                                                                                (known_predictions[i], unknown_predictions[i], background_predictions[i]),
                                                                                                pred_class_id_label=self.class_id_label,
                                                                                                gt_class_id_label=self.gt_class_id_label,
                                                                                                semantic_segmentation_class_id_label=self.semantic_segmentation_class_id_label))
            if A_OSE_wandb_images != []:
                wandb.log({("Images/" + loging_name + "_A_OSE"): A_OSE_wandb_images})

        # If first val, log images and pred
        if batch_idx <= 1 and predictions != []: # or batch_idx == 2:

            nn_wandb_images = []
            raw_nn_wandb_images = []
            for i in range(len(images)):
                raw_nn_wandb_images.append(get_wandb_image_with_labels(images[i], targets[i], predictions[i], self.class_id_label, self.merged_class_id_label))
                nn_wandb_images.append(get_wandb_image_with_labels_background_unknown_known(images[i],
                                                                                            (known_targets[i], unknown_targets[i]),
                                                                                            (known_predictions[i], unknown_predictions[i], background_predictions[i]),
                                                                                            pred_class_id_label=self.class_id_label,
                                                                                            gt_class_id_label=self.merged_class_id_label,
                                                                                            semantic_segmentation_class_id_label=self.semantic_segmentation_class_id_label,
                                                                                            display=False, img_id=str(batch_idx) + str(i)))

            wandb.log({("Images/" + loging_name + "_predictions"): nn_wandb_images})
            wandb.log({("Images/" + loging_name + "_raw_predictions"): raw_nn_wandb_images})











    def plot_kubt(self, known_predictions, unknown_predictions, background_predictions, targets):

        fig, ax = plt.subplots()

        for i in range(len(known_predictions)):

            """
            ax.scatter(known_predictions[i]["scores"].cpu().numpy(), known_predictions[i]["custom_scores"]["color_contrast"].cpu().numpy(), c="tab:blue", label="Known")
            ax.scatter(unknown_predictions[i]["scores"].cpu().numpy(), unknown_predictions[i]["custom_scores"]["color_contrast"].cpu().numpy(), c="tab:red", label="Unknown")
            ax.scatter(background_predictions[i]["scores"].cpu().numpy(), background_predictions[i]["custom_scores"]["color_contrast"].cpu().numpy(), c="tab:green", label="Background")
            """

            ax.scatter(known_predictions[i]["custom_scores"]["edge_density"].cpu().numpy(), known_predictions[i]["custom_scores"]["color_contrast"].cpu().numpy(), c="tab:blue", label="Known")
            ax.scatter(unknown_predictions[i]["custom_scores"]["edge_density"].cpu().numpy(), unknown_predictions[i]["custom_scores"]["color_contrast"].cpu().numpy(), c="tab:red", label="Unknown")
            ax.scatter(background_predictions[i]["custom_scores"]["edge_density"].cpu().numpy(), background_predictions[i]["custom_scores"]["color_contrast"].cpu().numpy(), c="tab:green", label="Background")

        ax.legend()
        ax.grid(True)


    # ------ Validation ----------------------------------

    def on_validation_epoch_start(self):
        self.metrics_module.reset()
        self.losses = [] 
        if self.multiple_rpn_heads:
            self.model.rpn.calculate_losses = self.validation_calculate_losses

        # TODO augmenter le nombre de sample pour la loss de validaiton

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        for target in targets:
            if len(target["boxes"]) == 0:
                return None
            if (target["boxes"][:, 0] >= target["boxes"][:, 2]).any() or (target["boxes"][:, 1] >= target["boxes"][:, 3]).any():
                return None


        if self.only_rpn:
            predictions, losses = self.model(images, targets)
            proposals_predictions = None
        else:
            losses, predictions, proposals_predictions = self.model(images, targets)
            
        self.losses.append(losses)
        for loss_name, loss in losses.items():
            self.log("Val/" + loss_name, loss)


        losses = sum(loss for loss in losses.values())
        self.log("Val/sum_losses", losses)

        self.evaluation_step(predictions, images, targets, batch_idx, "Validation", proposals_predictions=proposals_predictions)

    def on_validation_epoch_end(self):

        wandb.log({"epoch": self.current_epoch})

        # Calculate the mean of each loss
        if len(self.losses) > 0:
            mean_losses = {key: sum(loss[key] for loss in self.losses) / len(self.losses) for key in self.losses[0]}
            # Log the mean losses
            for key, value in mean_losses.items():
                self.log("Val/mean " + key, value)

            if not self.automatic_optimization: # Then their is multiple optimizer
                for i, lr_scheduler in enumerate(self.lr_schedulers()):
                    lr_scheduler.step(mean_losses[self.lr_scheduler_metrics_monitor[i]])


        # Save model
        if self.current_epoch % self.save_each_epoch == 0 and self.current_epoch != 0:
            path = self.save_path + "epoch_" + str(self.current_epoch)
            torch.save(self.model, path)

        print()
        print("Validations metrics :")
        wandb.log({"Val/metrics": self.metrics_module.get_wandb_metrics(with_print=True)})
        val_raw_coverage = self.metrics_module.get_raw_pourcent_coverage()
        self.log("val_coverage", val_raw_coverage)

        if self.metrics_module.cfg.mAP:
            val_map_known = self.metrics_module.current_known_map['map']
            self.log("val_map_known", val_map_known)
            self.log("val_map_unknown", self.metrics_module.current_unknown_map['map'])

            if val_map_known >= self.best_map:
                torch.save(self.model, self.save_path + "epoch_" + str(self.current_epoch) + "Best_map_" + str(val_map_known.item()))
                self.best_map = val_map_known

        if val_raw_coverage >= self.best_val_raw_coverage:
            torch.save(self.model, self.save_path + "epoch_" + str(self.current_epoch) + "Best_map_" + str(val_raw_coverage))
            self.best_val_raw_coverage = val_raw_coverage

    # ------ TESTING ----------------------------------

    def on_test_epoch_start(self):
        self.metrics_module.reset()
        if self.multiple_rpn_heads:
            self.model.rpn.calculate_losses = self.test_calculate_losses

    def test_step(self, batch, batch_idx):

        images, targets = batch
        for target in targets:
            if len(target["boxes"]) == 0:
                return None
            if (target["boxes"][:, 0] >= target["boxes"][:, 2]).any() or (target["boxes"][:, 1] >= target["boxes"][:, 3]).any():
                return None


        if self.only_anchors:
            if self.only_rpn:
                predictions, losses = self.model(images, targets)
                proposals_predictions = None
            else:
                losses, _, proposals_predictions = self.model(images, targets)
            predictions = []
            for anchors in self.model.rpn.current_anchors:
                prediction = {}
                prediction["boxes"] = anchors
                prediction["labels"] = torch.zeros(anchors.shape[0], device=self.device, dtype=int)
                prediction["scores"] = torch.zeros(anchors.shape[0], device=self.device)
                predictions.append(prediction)

        elif self.only_rpn:
            predictions, losses = self.model(images, targets)
            proposals_predictions = None
        else:
            losses, predictions, proposals_predictions = self.model(images, targets)


           
        if self.save_predictions or self.save_targets:
            self.save_predictions_pth(predictions, targets, dataset=self.dataset_split_name)

        self.evaluation_step(predictions, images, targets, batch_idx, "Test", proposals_predictions=proposals_predictions)

        return 


    def on_test_epoch_end(self):

        print()
        print("testing metrics :")
        print()
        wandb.log({"Test/metrics": self.metrics_module.get_wandb_metrics(with_print=True)})
        test_raw_coverage = self.metrics_module.get_raw_pourcent_coverage()
        self.log("test_coverage", test_raw_coverage)
        print()

        if self.metrics_module.cfg.mAP:
            self.log("test_map_known", self.metrics_module.current_known_map['map'])
            self.log("test_map_unknown", self.metrics_module.current_unknown_map['map'])

        self.log("test_precision_known", self.metrics_module.get_known_precision())
        self.log("test_recall_known", self.metrics_module.get_known_recall())
        self.log("test_precision_unknown", self.metrics_module.get_unknown_precision())
        self.log("test_recall_unknown", self.metrics_module.get_unknown_recall())
        self.log("test_A-OSE", self.metrics_module.get_open_set_errors())

    
    # -------------- MISC -------------------------------------------------------

    def save_predictions_pth(self, predictions, targets, dataset="test"):

        if self.is_save_anchors:
            anchors = self.model.rpn.current_anchors
            features = self.model.rpn.current_features

        proposals = self.model.rpn.current_all_proposals
        scores = self.model.rpn.current_all_scores

        batch_size = proposals.shape[0]

        # Reshape into dicts of each image
        scores_formated = []
        for _ in range(batch_size):
            scores_formated.append({})
        for key, value in scores.items():
            value = value.view(batch_size, value.size(0) // batch_size, value.size(1))
            scores_formated.append({})
            for i in range(batch_size):
                scores_formated[i][key] = value[i].half()


        model_name = self.model_name
        if self.is_load:
            model_name = self.model_name_to_load

        for i, (target, prediction, proposal, score) in enumerate(zip(targets, predictions, proposals, scores_formated)):
            image_name = target["name"]

            if self.save_predictions:
                pred_file_path = self.save_predictions_path + "/" + dataset + "/" + model_name + "_" + image_name + ".pth"
                predictions_to_save = {"filtered_predictions": prediction, "proposals": proposal.to(torch.int16), "scores": score}
                torch.save(predictions_to_save, pred_file_path)

            if self.save_targets:
                target_file_path = self.save_predictions_path + "/" + dataset +  "/targets_" + image_name + ".pth"
                torch.save(target, target_file_path)

            if self.is_save_anchors:
                anchors_file_path = self.save_predictions_path + "/" + dataset +  "/anchors_" + image_name + ".pth"
                torch.save(anchors[i], anchors_file_path)

                if not self.features_is_save : 
                    feature_file_path = self.save_predictions_path + "/" + dataset +  "/features.pth"
                    self.features_is_save = True
                    torch.save(features, feature_file_path)

    def configure_optimizers(self):

        if self.optimizer == "SGD":
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum, weight_decay=self.weight_decay)
        elif self.optimizer == "Adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            raise ValueError("Does not hundle this optimizer " + self.optimizer)

        if self.scheduler == "" or self.scheduler == None:
            return optimizer

        if self.scheduler == "linear_warmup_cosine_annealing_lr":

            nb_epoch_warmup = max(1, int(self.percent_warmup_epochs * self.max_epochs))
            lambda_linear_warmup = (lambda epoch: ((epoch + 1) / (nb_epoch_warmup + 1)))
            #print(self.learning_rate, nb_epoch_warmup)
            #print(lambda_linear_warmup(0),lambda_linear_warmup(2),lambda_linear_warmup(5))
            warmup_scheduler = LambdaLR(optimizer, lr_lambda=lambda_linear_warmup)
            schedulers = [warmup_scheduler]

            if self.max_epochs - nb_epoch_warmup == 0:
                return {"optimizer": optimizer, "lr_scheduler": warmup_scheduler}

            cosine_scheduler = CosineAnnealingLR(optimizer, self.max_epochs - nb_epoch_warmup)
            schedulers.append(cosine_scheduler)
            scheduler = SequentialLR(optimizer, schedulers=schedulers, milestones=[nb_epoch_warmup])

            return {"optimizer": optimizer, "lr_scheduler": scheduler}

        elif self.scheduler == "reduce_on_plateau":
            scheduler = ReduceLROnPlateau(optimizer, 'min', patience=self.scheduler_patience)

            return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "Val/sum_losses"}

        elif self.scheduler == "reduce_on_plateau_for_each_heads":

            if not self.multiple_rpn_heads:
                raise ValueError("Can't have lr scheduler for multiple head if model has not multiple head...")
            self.automatic_optimization = False

            optimizers, lr_scheduler_configs, self.lr_scheduler_metrics_monitor = [], [], []

            for head_name, head in self.model.rpn.heads.items():
                parameters = head.parameters()
                learning_rate = self.learning_rate_for_each_heads[head_name]
                monitor = "Val/loss_" + head_name.lower()
                self.lr_scheduler_metrics_monitor.append("loss_" + head_name.lower())
                if self.optimizer == "SGD" and head_name == "bbox_deltas":
                    optimizer = torch.optim.SGD(parameters, lr=learning_rate, momentum=self.momentum, weight_decay=self.weight_decay)
                elif self.optimizer == "SGD":
                    optimizer = torch.optim.SGD(parameters, lr=learning_rate, momentum=self.momentum, weight_decay=self.weight_decay)
                elif self.optimizer == "Adam":
                    optimizer = torch.optim.Adam(parameters, lr=learning_rate, weight_decay=self.weight_decay)
                else:
                    raise ValueError("Does not hundle this optimizer " + self.optimizer)

                if head_name == "bbox_deltas":
                    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=self.scheduler_patience + 4)
                else:
                    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=self.scheduler_patience + 4)

                optimizers.append(optimizer)
                lr_scheduler_config = {
                                    "scheduler": scheduler,
                                    "interval": self.lr_scheduler_interval,
                                    "frequency": self.lr_scheduler_frequency,
                                    "monitor": monitor,
                                    "strict": True,
                                    "name": "lr-" + monitor,
                                }
                lr_scheduler_configs.append(lr_scheduler_config)

            return optimizers, lr_scheduler_configs




    # Gives a mask of anchor points with their centerness classes
    def get_prediction_masks_fcos(self, image_size, batch_size, interval=10):
        masks = {}

        # Mask of centerness
        if self.enable_semantics_centerness:
            foregroud_mask = self.model.head.centerness_foregroud_mask[batch_size - 1]
        else:
            foregroud_mask = self.model.head.foregroud_mask[batch_size - 1]

        pred_centerness = (self.model.head.pred_centerness[batch_size - 1][foregroud_mask]) #[N, HWA]
        gt_centerness = self.model.head.gt_ctrness_targets[batch_size - 1][foregroud_mask]
        anchor_centers = self.model.anchor_centers[foregroud_mask]

        gt_centerness_mask = torch.zeros(image_size).int()
        pred_centerness_mask = torch.zeros(image_size).int()

        for i in range(0, 100, interval):
            all_anchor_in_interval_pred = (pred_centerness >= i/100) & (pred_centerness < (i/100 + interval))
            all_anchor_in_interval_gt = (gt_centerness >= i/100) & (gt_centerness < (i/100 + interval))
            for b in range(-2, 2):
                for j in range(-2, 2):
                    pred_centerness_mask[anchor_centers[all_anchor_in_interval_pred, 1].long() + b, anchor_centers[all_anchor_in_interval_pred, 0].long() + j] = i + interval
                    gt_centerness_mask[anchor_centers[all_anchor_in_interval_gt, 1].long() + b, anchor_centers[all_anchor_in_interval_gt, 0].long() + j] = i + interval

        masks["centerness prediction"] = {"mask_data": pred_centerness_mask.cpu().numpy()}
        masks["centerness ground_truth"] = {"mask_data": gt_centerness_mask.cpu().numpy()}

        return masks

    def get_image_with_fcos_predictions(self, last_image, batch_size):

        image_size = (last_image.shape[1], last_image.shape[2])

        anchor_centers = self.model.anchor_centers
        gt_centers = self.model.gt_centers 
        last_targets = self.model.last_targets
        last_pred = self.model.head.last_pred_boxes
        gt_classes_targets = self.model.head.gt_classes_targets[batch_size - 1]
        cls_logits = self.model.head.cls_logits[batch_size - 1] 
        num_anchors_per_level = self.model.num_anchors_per_level

        name = last_targets["name"]

        #centerness
        pred_centerness = self.model.head.pred_centerness[batch_size - 1].detach()
        gt_centerness = torch.nan_to_num(self.model.head.gt_ctrness_targets[batch_size - 1])

        if self.enable_semantics_centerness:
            foregroud_mask = self.model.head.centerness_foregroud_mask[batch_size - 1]
        else:
            foregroud_mask = self.model.head.foregroud_mask[batch_size - 1]

        gt_box_data = self.get_wandb_box(last_targets, image_size)
        pred_box_data = self.get_wandb_box(last_pred[batch_size -1], image_size)
        boxes={"ground_truth": {"box_data": gt_box_data, "class_labels": self.class_id_label},
               "predictions": {"box_data": pred_box_data, "class_labels": self.class_id_label}}

        levels = [8, 16, 32, 64, 128]
        masks = {}

        anchors_indexes = 0
        for i, level in enumerate(levels):
            if "seg_masks" in last_targets:
                seg_gt_mask = torch.zeros(image_size, device=last_targets["boxes"].device).long()

            centerness_pred_mask = torch.zeros(image_size, device=last_targets["boxes"].device, dtype=pred_centerness.dtype)
            classes_pred_masks = [torch.zeros(image_size, device=last_targets["boxes"].device).int() for i in range(self.model.nb_classes)]

            anchors_indexes_next = anchors_indexes + int(((last_image.shape[2]/level) * last_image.shape[1]/level))
            anchor_centers_x = anchor_centers[anchors_indexes:anchors_indexes_next, 1].long()
            anchor_centers_y = anchor_centers[anchors_indexes:anchors_indexes_next, 0].long()

            for b in range(-2, 2):
                for j in range(-2, 2):

                    nothing_mask = torch.sum(gt_classes_targets[anchors_indexes:anchors_indexes_next], dim=1) == 0

                    if "seg_masks" in last_targets:
                        seg_gt_mask[anchor_centers_x + b, anchor_centers_y + j] = 254 * nothing_mask
                        seg_gt_mask[anchor_centers_x + b,anchor_centers_y  + j] += torch.argmax(gt_classes_targets[anchors_indexes:anchors_indexes_next], dim=1) + 1
                        seg_gt_mask[gt_centers[:, 1].long() + b, gt_centers[:, 0].long() + j] = len(self.classes_names) + 1

                    centerness_pred_mask[anchor_centers_x + b, anchor_centers_y + j] = pred_centerness[anchors_indexes:anchors_indexes_next]

                    for c in range(self.model.nb_classes):
                        classes_pred_masks[c][anchor_centers_x + b, anchor_centers_y + j] = (cls_logits[anchors_indexes:anchors_indexes_next, c] >= 0.5).int() #[H, W] <- [HWA, 11] 


            anchors_indexes = anchors_indexes_next
            if "seg_masks" in last_targets:
                masks['ground_truth_class_level_' + str(i)] = {"mask_data": seg_gt_mask.cpu().numpy(), "class_labels": self.class_labels}
            masks['prediction_centerness_level_' + str(i)] = {"mask_data": centerness_pred_mask.cpu().numpy()}

        masks.update(self.get_prediction_masks_fcos(image_size, batch_size))

        return wandb.Image(last_image.cpu(), masks=masks, boxes=boxes)

    def log_fcos_heatmaps(self, last_image, batch_size):

        image_size = (last_image.shape[1], last_image.shape[2])

        anchor_centers = self.model.anchor_centers
        gt_centers = self.model.gt_centers 
        last_targets = self.model.last_targets
        last_pred = self.model.head.last_pred_boxes
        gt_classes_targets = self.model.head.gt_classes_targets[batch_size - 1]
        cls_logits = self.model.head.cls_logits[batch_size - 1] 
        num_anchors_per_level = self.model.num_anchors_per_level

        name = last_targets["name"]

        #centerness
        pred_centerness = self.model.head.pred_centerness[batch_size - 1].detach()
        gt_centerness = torch.nan_to_num(self.model.head.gt_ctrness_targets[batch_size - 1])

        if self.enable_semantics_centerness:
            foregroud_mask = self.model.head.centerness_foregroud_mask[batch_size - 1]
        else:
            foregroud_mask = self.model.head.foregroud_mask[batch_size - 1]

        levels = [8, 16, 32, 64, 128]

        anchors_indexes = 0
        for i, level in enumerate(levels):

            centerness_pred_mask = torch.zeros(image_size, device=last_targets["boxes"].device, dtype=pred_centerness.dtype)
            classes_pred_masks = [torch.zeros(image_size, device=last_targets["boxes"].device).int() for i in range(self.model.nb_classes)]

            centerness_pred_heatmap = torch.zeros_like(centerness_pred_mask)
            centerness_gt_heatmap = torch.zeros_like(centerness_pred_mask, dtype=gt_centerness.dtype)
            centerness_gt_heatmap_foregroud = torch.zeros_like(centerness_gt_heatmap)
            classes_pred_heatmaps = [torch.zeros_like(centerness_pred_mask) for i in range(self.model.nb_classes)]

            anchors_indexes_next = anchors_indexes + int(((last_image.shape[2]/level) * last_image.shape[1]/level))
            anchor_centers_x = anchor_centers[anchors_indexes:anchors_indexes_next, 1].long()
            anchor_centers_y = anchor_centers[anchors_indexes:anchors_indexes_next, 0].long()

            for b in range(-2, 2):
                for j in range(-2, 2):

                    centerness_pred_heatmap[anchor_centers_x + b, anchor_centers_y + j] = pred_centerness[anchors_indexes:anchors_indexes_next]
                    centerness_gt_heatmap[anchor_centers_x + b, anchor_centers_y + j] = gt_centerness[anchors_indexes:anchors_indexes_next]
                    centerness_gt_heatmap_foregroud[anchor_centers_x + b, anchor_centers_y + j] = gt_centerness[anchors_indexes:anchors_indexes_next] * foregroud_mask[anchors_indexes:anchors_indexes_next].int()
                    for c in range(self.model.nb_classes):
                        classes_pred_heatmaps[c][anchor_centers_x + b, anchor_centers_y + j] = cls_logits[anchors_indexes:anchors_indexes_next, c] #[H, W] <- [HWA, 11] 

            anchors_indexes = anchors_indexes_next

            if i == 0:
                for c in range(self.model.nb_classes):
                    if not torch.all(classes_pred_heatmaps[c] == 0):
                        image_heatmap = np.stack((classes_pred_heatmaps[c].detach().cpu().numpy(),)*3, axis=-1)
                        wandb.log({("Images/inside_train/heatmap/prediction_class_" + self.class_labels[c + 1] + "_level_" + str(i)) : wandb.Image(image_heatmap)}, commit=False)

            image_heatmap = np.stack((centerness_pred_heatmap.cpu().detach().numpy(),)*3, axis=-1)
            wandb.log({("Images/inside_train/heatmap/prediction_centerness_level_" + str(i)) : wandb.Image(image_heatmap)}, commit=False)
            if not torch.all(classes_pred_heatmaps[c] == 0):
                image_heatmap = np.stack((centerness_gt_heatmap_foregroud.cpu().numpy(),)*3, axis=-1)
                wandb.log({("Images/inside_train/heatmap/gt_centerness_foregroud_level_" + str(i)) : wandb.Image(image_heatmap)}, commit=False)

        wandb.log({("Images/inside_train/heatmap/image") : wandb.Image(last_image.cpu()), "Image/inside_train/heatmap/image_name": name})


