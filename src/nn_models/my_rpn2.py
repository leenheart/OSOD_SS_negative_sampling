from typing import Dict, List, Optional, Tuple

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torchvision.ops import boxes as box_ops, Conv2dNormActivation

from torchvision.models.detection import _utils as det_utils

# Import AnchorGenerator to keep compatibility.
from torchvision.models.detection.anchor_utils import AnchorGenerator  # noqa: 401
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.rpn import concat_box_prediction_layers, permute_and_flatten

from postprocessing_predictions import calculate_object_on_drivable_score_v2, calculate_sum_pixel_SS_in_boxes, calculate_centerness_boxes, calculate_centerness_box
from nn_models.adjust_smooth_l1_loss import AdjustSmoothL1Loss 

"""
 Make a custom rpn that can hundle multiple differents heads and score without changing the definition of the functions

 RPN works with one head

"""

class RPNHeadSingle(nn.Module):
    """
    Adds a simple RPN Head with classification and regression heads

    Args:
        in_channels (int): number of channels of the input feature
        num_anchors (int): number of anchors to be predicted
        conv_depth (int, optional): number of convolutions
    """

    _version = 2

    def __init__(self, in_channels: int, num_anchors: int, loss_function: str, beta: int, conv_depth=1) -> None:
        super().__init__()
        convs = []
        for _ in range(conv_depth):
            convs.append(Conv2dNormActivation(in_channels, in_channels, kernel_size=3, norm_layer=None))
        self.conv = nn.Sequential(*convs)
        self.score = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
    
        if loss_function == None:
            self.loss_function = None
        elif loss_function == "adjust_smooth_l1_loss":
            #print("Use adjust smooth l1 loss ")
            self.loss_function = AdjustSmoothL1Loss(1, beta=beta)
        elif loss_function == "smooth_l1_loss":
            #print("Use smooth l1 loss ")
            self.loss_function = nn.modules.loss.SmoothL1Loss(reduction='sum', beta=beta)
        else:
            raise ValueError("Does not handle " + loss_function + ". But handle adjust_smooth_l1_loss and smooth_l1_loss")

        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, std=0.01)  # type: ignore[arg-type]
                if layer.bias is not None:
                    torch.nn.init.constant_(layer.bias, 0)  # type: ignore[arg-type]

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)

        if version is None or version < 2:
            for type in ["weight", "bias"]:
                old_key = f"{prefix}conv.{type}"
                new_key = f"{prefix}conv.0.0.{type}"
                if old_key in state_dict:
                    state_dict[new_key] = state_dict.pop(old_key)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x: List[Tensor]) -> List[Tensor]:
        scores = []
        for feature in x:
            t = self.conv(feature)
            scores.append(self.score(t))
        return scores


def concat_box_prediction_layers_multiple_scores(box_scores: Dict[str, List[Tensor]]) -> Tuple[Dict[str, Tensor], Tensor]:
    box_scores_flattened = {}
    for score_name, scores in box_scores.items():
        box_scores_flattened[score_name] = []

    box_regression_flattened = []
    # for each feature level, permute the outputs to make them be in the
    # same format as the labels. Note that the labels are computed for
    # all feature levels concatenated, so we keep the same representation
    # for the objectness and the box_regression
    for idx_level, box_regression_per_level in enumerate(box_scores["bbox_deltas"]):
        N, AxC, H, W = box_scores["BCE"][idx_level].shape
        Ax4 = box_regression_per_level.shape[1]
        A = Ax4 // 4
        C = AxC // A

        for score_name, scores in box_scores.items():
            box_scores_flattened[score_name].append(permute_and_flatten(scores[idx_level], N, A, C, H, W))

        box_regression_per_level = permute_and_flatten(box_regression_per_level, N, A, 4, H, W)
        box_regression_flattened.append(box_regression_per_level)
    # concatenate on the first dimension (representing the feature levels), to
    # take into account the way the labels were generated (with all feature maps
    # being concatenated as well)
    box_scores = {} 
    for score_name, scores_flatten in box_scores_flattened.items():
        box_scores[score_name] = torch.cat(scores_flatten, dim=1).flatten(0, -2)
    box_regression = torch.cat(box_regression_flattened, dim=1).reshape(-1, 4)
    return box_scores, box_regression

def concat_prediction_layers_multiple_scores(predictions_scores: Dict[str, List[Tensor]]) -> Dict[str, Tensor]:
    outputs_scores_flattened = {}
    size = len(predictions_scores["IOU"])
    for score_name, predictions_score in predictions_scores.items():
        outputs_scores_flattened[score_name] = []
        if size != len(predictions_score):
            raise ValueError("RPN HEADS predictions list should all be the same size !")
    box_regression_flattened = []

    # for each feature level, permute the outputs to make them be in the
    # same format as the labels. Note that the labels are computed for
    # all feature levels concatenated, so we keep the same representation
    # for the objectness and the box_regression
    for idx_by_level, box_regression_per_level in enumerate(box_regression): #, box_regression, oro):
        box_cls_per_level = predictions_scores["BCE"][idx_by_level]
        N, AxC, H, W = box_cls_per_level.shape
        Ax4 = box_regression_per_level.shape[1]
        A = Ax4 // 4
        C = AxC // A
        for score_name, predictions_score in predictions_scores.items():
            predictions_score_flatten = permute_and_flatten(predictions_score[idx_by_level], N, A, C, H, W)
            outputs_scores_flattened[score_name].append(predictions_score_flatten)

        box_regression_per_level = permute_and_flatten(box_regression_per_level, N, A, 4, H, W)
        box_regression_flattened.append(box_regression_per_level)

    # concatenate on the first dimension (representing the feature levels), to
    # take into account the way the labels were generated (with all feature maps
    # being concatenated as well)
    for score_name, predictions_score in predictions_scores.items():
        outputs_scores_flattened[score_name] = torch.cat(outputs_scores_flattened[score_name], dim=1).flatten(0, -2)
    box_regression = torch.cat(box_regression_flattened, dim=1).reshape(-1, 4)
    return outputs_scores_flattened, box_regression

class Multiple_Head_RegionProposalNetwork(torch.nn.Module):
    """
    Implements Region Proposal Network (RPN).

    Args:
        anchor_generator (AnchorGenerator): module that generates the anchors for a set of feature
            maps.
        head (nn.Module): module that computes the objectness and regression deltas
        fg_iou_thresh (float): minimum IoU between the anchor and the GT box so that they can be
            considered as positive during training of the RPN.
        bg_iou_thresh (float): maximum IoU between the anchor and the GT box so that they can be
            considered as negative during training of the RPN.
        batch_size_per_image (int): number of anchors that are sampled during training of the RPN
            for computing the loss
        positive_fraction (float): proportion of positive anchors in a mini-batch during training
            of the RPN
        pre_nms_top_n (Dict[str, int]): number of proposals to keep before applying NMS. It should
            contain two fields: training and testing, to allow for different values depending
            on training or evaluation
        post_nms_top_n (Dict[str, int]): number of proposals to keep after applying NMS. It should
            contain two fields: training and testing, to allow for different values depending
            on training or evaluation
        nms_thresh (float): NMS threshold used for postprocessing the RPN proposals

    """

    __annotations__ = {
        "box_coder": det_utils.BoxCoder,
        "proposal_matcher": det_utils.Matcher,
        "fg_bg_sampler": det_utils.BalancedPositiveNegativeSampler,
    }

    def __init__(
        self,
        anchor_generator: AnchorGenerator,
        heads,#: nn.ModuleDict[nn.Module],
        cfg,
        # Faster-RCNN Training
        proposal_matcher,
        fg_bg_sampler,
        # Faster-RCNN Inference
        pre_nms_top_n: Dict[str, int],
        post_nms_top_n: Dict[str, int],
        nms_thresh: float,
        score_thresh: float = 0.0,
    ) -> None:
        super().__init__()
        self.anchor_generator = anchor_generator
        self.heads = heads
        self.box_coder = det_utils.BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))

        # used during training
        self.box_similarity = box_ops.box_iou

        """
        self.proposal_matcher = det_utils.Matcher(
            fg_iou_thresh,
            bg_iou_thresh,
            allow_low_quality_matches=True,
        )
        """
        self.proposal_matcher = proposal_matcher

        self.fg_bg_sampler =fg_bg_sampler
        # used during testing
        self._pre_nms_top_n = pre_nms_top_n
        self._post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh
        self.score_thresh = score_thresh
        self.min_size = 1e-3

    
        if cfg.bbox_loss_function == "adjust_smooth_l1_loss":
            print("Use adjust smooth l1 loss for bbox rpn regression")
            self.bbox_loss_function = AdjustSmoothL1Loss(4, beta=cfg.bbox_reg_beta)
        elif cfg.bbox_loss_function == "smooth_l1_loss":
            print("Use smooth l1 loss for bbox rpn regression")
            self.bbox_loss_function = nn.modules.loss.SmoothL1Loss(reduction='sum', beta=cfg.bbox_reg_beta)
        else:
            raise ValueError("Does not handle " + cfg.bbox_loss_function + ". But handle adjust_smooth_l1_loss and smooth_l1_loss")

        self.oro = cfg.oro

        self.use_negative_sample = cfg.use_negative_sample
        self.use_pure_negative_sample = cfg.use_pure_negative_sample
        self.objectness_score_used = cfg.objectness_score_used
        print(f"{self.use_negative_sample = }")
        print(f"{self.use_pure_negative_sample = }")
        print(f"{self.objectness_score_used = }")
        self.positive_threshold_classif = cfg.positive_threshold_classif
        self.positive_threshold_reg = cfg.positive_threshold_reg
        self.nb_boxes_sampled_per_img = cfg.nb_boxes_sampled_per_img

    def pre_nms_top_n(self) -> int:
        if self.training:
            return self._pre_nms_top_n["training"]
        return self._pre_nms_top_n["testing"]

    def post_nms_top_n(self) -> int:
        if self.training:
            return self._post_nms_top_n["training"]
        return self._post_nms_top_n["testing"]


    def assign_targets_to_anchors(
        self, anchors: List[Tensor], targets: List[Dict[str, Tensor]]
    ) -> Tuple[List[Tensor], List[Tensor]]:

        labels = []
        matched_gt_boxes = []
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            gt_boxes = targets_per_image["boxes"]

            if gt_boxes.numel() == 0:
                # Background image (negative example)
                device = anchors_per_image.device
                matched_gt_boxes_per_image = torch.zeros(anchors_per_image.shape, dtype=torch.float32, device=device)
                labels_per_image = torch.zeros((anchors_per_image.shape[0],), dtype=torch.float32, device=device)
            else:
                match_quality_matrix = self.box_similarity(gt_boxes, anchors_per_image)
                matched_idxs = self.proposal_matcher(match_quality_matrix)
                # get the targets corresponding GT for each proposal
                # NB: need to clamp the indices because we can have a single
                # GT in the image, and matched_idxs can be -2, which goes
                # out of bounds
                matched_gt_boxes_per_image = gt_boxes[matched_idxs.clamp(min=0)]

                labels_per_image = matched_idxs >= 0
                labels_per_image = labels_per_image.to(dtype=torch.float32)

                # Background (negative examples)
                bg_indices = matched_idxs == self.proposal_matcher.BELOW_LOW_THRESHOLD
                labels_per_image[bg_indices] = 0.0

                # discard indices that are between thresholds
                inds_to_discard = matched_idxs == self.proposal_matcher.BETWEEN_THRESHOLDS
                labels_per_image[inds_to_discard] = -1.0

                #print(f"{(labels_per_image >= 1).sum() = }")

            labels.append(labels_per_image)
            matched_gt_boxes.append(matched_gt_boxes_per_image)
        return labels, matched_gt_boxes

    def assign_targets_to_anchors_multiple_scores(
        self, anchors: List[Tensor], targets: List[Dict[str, Tensor]], proposals = False,
    ) -> Tuple[Dict[str, List[Tensor]], List[Tensor]]:

        suffix = "" if not proposals else "_proposal"


        labels = {}
        labels["BCE" + suffix] = []
        labels["IOU" + suffix] = []
        labels["gt_matches_boxes" + suffix] = []
        labels["best_quality_match_inds"+ suffix] = []
        if self.use_pure_negative_sample:
            labels["SS_binary_objects" + suffix] = []
        matched_gt_boxes = []

        for anchors_per_image, targets_per_image in zip(anchors, targets):
            gt_boxes = targets_per_image["boxes"]

            if gt_boxes.numel() == 0:
                # Background image (negative example)
                device = anchors_per_image.device
                matched_gt_boxes_per_image = torch.zeros(anchors_per_image.shape, dtype=torch.float32, device=device)
                for label_name, _ in labels.items():
                    if label_name == "gt_matches_boxes" + suffix:
                        labels[label_name].append(torch.zeros(anchors_per_image.shape, dtype=torch.float32, device=device))
                    elif label_name == "SS_binary_objects" + suffix:
                        labels[label_name].append(targets_per_image["semantic_segmentation_binary_objects"])
                    elif label_name == "best_quality_match_inds":
                        labels["best_quality_match_inds"+ suffix].append(torch.tensor([], dtype=int, device=device))
                    else:
                        labels[label_name].append(torch.zeros((anchors_per_image.shape[0],), dtype=torch.float32, device=device))
            else:
                match_quality_matrix = self.box_similarity(gt_boxes, anchors_per_image)
                matched_vals, matches = match_quality_matrix.max(dim=0)


                # For each gt, find the prediction with which it has the highest quality
                highest_quality_foreach_gt, _ = match_quality_matrix.max(dim=1)
                # Find the highest quality match available, even if it is low, including ties
                gt_pred_pairs_of_highest_quality = torch.where(match_quality_matrix == highest_quality_foreach_gt[:, None])
                # Example gt_pred_pairs_of_highest_quality:
                # (tensor([0, 1, 1, 2, 2, 3, 3, 4, 5, 5]),
                #  tensor([39796, 32055, 32070, 39190, 40255, 40390, 41455, 45470, 45325, 46390]))
                # Each element in the first tensor is a gt index, and each element in second tensor is a prediction index
                # Note how gt items 1, 2, 3, and 5 each have two ties

                pred_inds_best_quality = gt_pred_pairs_of_highest_quality[1]
                #print(f"{pred_inds_best_quality.numel() = }")
                labels["best_quality_match_inds" + suffix].append(pred_inds_best_quality)

                matched_gt_boxes_per_image = gt_boxes[matches]

                # Add new scores
                labels["IOU" + suffix].append(matched_vals)
                labels["BCE" + suffix].append((matched_vals >= self.positive_threshold_classif).to(dtype=torch.float32)) # TODO make parameter for this
                labels["BCE" + suffix][-1][pred_inds_best_quality] = 1

                matched_gt_boxes_per_img = gt_boxes[matches]
                labels["gt_matches_boxes" + suffix].append(matched_gt_boxes_per_img)
                if self.use_pure_negative_sample and "semantic_segmentation_binary_objects" in targets_per_image:
                    labels["SS_binary_objects" + suffix].append(targets_per_image["semantic_segmentation_binary_objects"])

            matched_gt_boxes.append(matched_gt_boxes_per_image)

        return labels, matched_gt_boxes

    def _get_top_n_idx(self, objectness: Tensor, num_anchors_per_level: List[int]) -> Tensor:
        r = []
        offset = 0
        for ob in objectness.split(num_anchors_per_level, 1):
            num_anchors = ob.shape[1]
            pre_nms_top_n = det_utils._topk_min(ob, self.pre_nms_top_n(), 1)
            _, top_n_idx = ob.topk(pre_nms_top_n, dim=1)
            r.append(top_n_idx + offset)
            offset += num_anchors
        return torch.cat(r, dim=1)

    def filter_proposals(
        self,
        proposals: Tensor,
        objectness: Tensor,
        image_shapes: List[Tuple[int, int]],
        num_anchors_per_level: List[int],
    ) -> Tuple[List[Tensor], List[Tensor]]:

        num_images = proposals.shape[0]
        device = proposals.device
        # do not backprop through objectness
        objectness = objectness.detach()
        objectness = objectness.reshape(num_images, -1)

        levels = [
            torch.full((n,), idx, dtype=torch.int64, device=device) for idx, n in enumerate(num_anchors_per_level)
        ]
        levels = torch.cat(levels, 0)
        levels = levels.reshape(1, -1).expand_as(objectness)

        # select top_n boxes independently per level before applying nms
        top_n_idx = self._get_top_n_idx(objectness, num_anchors_per_level)

        image_range = torch.arange(num_images, device=device)
        batch_idx = image_range[:, None]

        objectness = objectness[batch_idx, top_n_idx]
        levels = levels[batch_idx, top_n_idx]
        proposals = proposals[batch_idx, top_n_idx]
        current_anchors = self.current_anchors[batch_idx, top_n_idx]

        objectness_prob = torch.sigmoid(objectness)

        final_anchors = []
        final_boxes = []
        final_scores = []
        for boxes, scores, lvl, img_shape, anchors in zip(proposals, objectness_prob, levels, image_shapes, current_anchors):
            boxes = box_ops.clip_boxes_to_image(boxes, img_shape)
            anchors = box_ops.clip_boxes_to_image(anchors, img_shape)

            # remove small boxes
            keep = box_ops.remove_small_boxes(boxes, self.min_size)
            boxes, scores, lvl, anchors = boxes[keep], scores[keep], lvl[keep], anchors[keep]

            # remove low scoring boxes
            # use >= for Backwards compatibility
            keep = torch.where(scores >= self.score_thresh)[0]
            boxes, scores, lvl, anchors = boxes[keep], scores[keep], lvl[keep], anchors[keep]

            # non-maximum suppression, independently done per level
            keep = box_ops.batched_nms(boxes, scores, lvl, self.nms_thresh)

            # keep only topk scoring predictions
            keep = keep[: self.post_nms_top_n()]
            boxes, scores, anchors = boxes[keep], scores[keep], anchors[keep]

            final_boxes.append(boxes)
            final_scores.append(scores)
            final_anchors.append(anchors)

        self.current_filtered_anchors = final_anchors

        return final_boxes, final_scores

    # Get sample negative (we are sure they are negative because they have 0 percent object pixel in it thanks to the semantic segmentation)
    def negative_sampled_using_ss(self, anchors: List[Tensor], negative_masks: List[Tensor], OSD_gts) -> List[Tensor]:

        pure_negative_masks = []

        for anchors_per_image, negative_mask, ss_gt in zip(anchors, negative_masks, OSD_gts):

            negative_inds = torch.nonzero(negative_mask)
            # select 1000 boxes to not take too long to calculate the SS scores
            perm_negative_inds = torch.randperm(negative_inds.numel(), device=negative_inds.device)[:self.nb_boxes_sampled_per_img * 3]

            # calculate SS scores on than selected anchors
            #percent_of_object_negative_sample = calculate_sum_pixel_SS_in_boxes(anchors_per_image[perm_negative_inds], ss_gt)
            percent_of_object_negative_sample = calculate_sum_pixel_SS_in_boxes(anchors_per_image[negative_mask][perm_negative_inds], ss_gt)

            self.current_anchors_pure_negative = anchors_per_image[perm_negative_inds]
            self.current_mask_semantic_pure_negative = ss_gt
            self.current_scores_percent_object_pure_negative = percent_of_object_negative_sample


            pure_negative_mask = torch.zeros_like(negative_mask)


            pure_negative_mask_only_on_negative = torch.zeros(negative_mask.sum(), device=negative_mask.device, dtype=bool)
            pure_negative_mask_only_on_negative[perm_negative_inds] = (percent_of_object_negative_sample == 0)
            pure_negative_mask[negative_mask] = pure_negative_mask_only_on_negative 

            pure_negative_masks.append(pure_negative_mask)

        return pure_negative_masks 

    def get_heads_outputs(self, features):

        scores = {}
        for head_name, head in self.heads.items():
            """
            if head_name == "BCE":
                bce_score, pred_bbox_deltas = head(features)
                scores["BCE"] = bce_score
                scores["bbox_deltas"] = pred_bbox_deltas
            else: 
                scores[head_name] = head(features)
            """
            scores[head_name] = head(features)

        return scores

    def sample_proportion_of_two_mask(self, mask_1_batch, mask_2_batch, proportion, nb_sampled):

        sampled_mask_batch, sampled_mask_1_batch, sampled_mask_2_batch = [], [], []
        for mask_1, mask_2 in zip(mask_1_batch, mask_2_batch):
        
            # protect against not enough positive examples
            num_inds_1 = min(mask_1.sum(), int(nb_sampled * proportion))
            num_inds_2 = min(mask_2.sum(), nb_sampled - num_inds_1)
            #print("nb sample  1: ", num_inds_1, " and 2: ", num_inds_2)
            
            indices_1 = torch.nonzero(mask_1)
            indices_2 = torch.nonzero(mask_2)

            # randomly select positive and negative examples
            perm1 = torch.randperm(mask_1.sum(), device=mask_1.device)[:num_inds_1]
            perm2 = torch.randperm(mask_2.sum(), device=mask_2.device)[:num_inds_2]

            # Create new mask
            sampled_mask_1 = torch.zeros_like(mask_1, dtype=bool)
            sampled_mask_2 = torch.zeros_like(mask_2, dtype=bool)
            sampled_mask_1[indices_1[perm1]] = 1
            sampled_mask_2[indices_2[perm2]] = 1
            sampled_mask = sampled_mask_1 | sampled_mask_2

            sampled_mask_1_batch.append(sampled_mask_1)
            sampled_mask_2_batch.append(sampled_mask_2)
            sampled_mask_batch.append(sampled_mask)

        sampled_mask_batch = torch.cat(sampled_mask_batch)
        sampled_mask_1_batch = torch.cat(sampled_mask_1_batch)
        sampled_mask_2_batch = torch.cat(sampled_mask_2_batch)

        return sampled_mask_batch, sampled_mask_1_batch, sampled_mask_2_batch

    def calculate_losses_with_boxes(self, labels, predictions, boxes, concat_boxes, targets, batch_size, suffix=""):

        # Sampling strategies
        ious = labels["IOU" + suffix]
        best_quality_match_inds = labels["best_quality_match_inds" + suffix]
        positive_classif = [iou >= self.positive_threshold_classif for iou in ious]
        positive_reg = [iou >= self.positive_threshold_reg for iou in ious]
        negative = [iou < 0.3 for iou in ious]

        # Set best quality match at positive labels because we do not have better matches for those gt and we want to detect them whatever
        for i, best_quality_match_inds_per_img in enumerate(best_quality_match_inds):
            #print(f"{best_quality_match_inds_per_img.numel() = }")
            positive_reg[i][best_quality_match_inds_per_img] = True
            positive_classif[i][best_quality_match_inds_per_img] = True

        # If we use pure negative thanks to semantic segmentation gt, replace negative by iou with negative with SS
        if self.use_pure_negative_sample:
            negative = self.negative_sampled_using_ss(boxes, negative, labels["SS_binary_objects" + suffix])


        half_pos_half_neg_sample, positive_sampled, negative_sampled = self.sample_proportion_of_two_mask(positive_classif, negative, 0.5, self.nb_boxes_sampled_per_img) #Half positiv and Half negativ
        half_pos_half_neg_sample_reg, positive_sampled_reg, negative_sampled_reg = self.sample_proportion_of_two_mask(positive_reg, negative, 0.5, self.nb_boxes_sampled_per_img) #Half positiv and Half negativ

        # Calculate LOSSES
        losses = {}

        if suffix == "": # we are with anchors for boxes
            # BBOX DELTA loss calculated only on acnhors
            bbox_deltas_mask_sampled = positive_sampled


            regression_targets = labels["bbox_deltas"] 
            pred_bbox_deltas = predictions["bbox_deltas"]
            loss_rpn_box_reg = self.bbox_loss_function(pred_bbox_deltas[bbox_deltas_mask_sampled], regression_targets[bbox_deltas_mask_sampled]) / bbox_deltas_mask_sampled.sum() #(bbox_deltas_mask_sampled.sum())
            losses["loss_bbox_deltas"] = loss_rpn_box_reg


        if "BCE" + suffix in self.heads:

            bce_mask_sampled = half_pos_half_neg_sample


            bce_preds = predictions["BCE" + suffix].flatten()
            bce_labels = torch.cat(labels["BCE" + suffix], dim=0)

            loss_bce = F.binary_cross_entropy_with_logits(bce_preds[bce_mask_sampled], bce_labels[bce_mask_sampled])
            losses["loss_bce" + suffix] = loss_bce
            
            if suffix == "": # we are with anchors for boxes
                
                # For sampling log
                self.current_bce_anchors_sample = concat_boxes[bce_mask_sampled]
                self.current_bce_targets_sample = bce_labels[bce_mask_sampled]
                self.current_bce_pred_sample = bce_preds[bce_mask_sampled]

        if "IOU"+ suffix in self.heads:

            if self.use_negative_sample:
                iou_mask_sampled = half_pos_half_neg_sample_reg
            else:
                iou_mask_sampled = positive_sampled_reg

            iou_preds = predictions["IOU" + suffix].flatten()
            iou_targets = torch.cat(labels["IOU" + suffix], dim=0)
            losses["loss_iou" + suffix] = self.heads["IOU" + suffix].loss_function(iou_preds[iou_mask_sampled], iou_targets[iou_mask_sampled]) / (iou_mask_sampled.sum())

            if suffix == "":
                self.current_iou_anchors_sample = concat_boxes[iou_mask_sampled]
                self.current_iou_targets_sample = iou_targets[iou_mask_sampled]
                self.current_iou_pred_sample = iou_preds[iou_mask_sampled]

        if "centerness"+ suffix in self.heads:
            matched_gt_boxes = torch.cat(labels["gt_matches_boxes" + suffix], dim=0)

            if self.use_negative_sample:
                centerness_mask_sampled = half_pos_half_neg_sample_reg
            else:
                centerness_mask_sampled = positive_sampled_reg

            target_sampled = matched_gt_boxes[centerness_mask_sampled]
            anchor_sampled = concat_boxes[centerness_mask_sampled]
            target_centerness = calculate_centerness_boxes(target_sampled, anchor_sampled)

            pred_centerness = predictions["centerness" + suffix][centerness_mask_sampled]
            losses["loss_centerness" + suffix] = self.heads["centerness" + suffix].loss_function(pred_centerness, target_centerness) / (centerness_mask_sampled.sum())

            if suffix == "":
                self.current_centerness_anchors_sample = anchor_sampled
                self.current_centerness_targets_sample = target_centerness
                self.current_centerness_pred_sample = pred_centerness 

        """
        if self.oro and False: #TODO
            #targets_oro = .append(calculate_object_on_drivable_score_v2(anchors_per_image[sampled_idx_per_image_mask_iou], ss_target, display=False, training=True))
            losses["oro"] = self.compute_loss_oro(scores["oro"], targets_oro, sampled) 
        """

        return losses

    def forward(
        self,
        images: ImageList,
        features: Dict[str, Tensor],
        targets: Optional[List[Dict[str, Tensor]]] = None,
    ) -> Tuple[List[Tensor], Dict[str, Tensor]]:

        """
        Args:
            images (ImageList): images for which we want to compute the predictions
            features (Dict[str, Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (List[Dict[str, Tensor]]): ground-truth boxes present in the image (optional).
                If provided, each element in the dict should contain a field `boxes`,
                with the locations of the ground-truth boxes.

        Returns:
            boxes (List[Tensor]): the predicted boxes from the RPN, one Tensor per
                image.
            losses (Dict[str, Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        # RPN uses all feature maps that are available
        features = list(features.values())
        heads_pred = self.get_heads_outputs(features)
        bce = heads_pred["BCE"]
        pred_bbox_deltas = heads_pred["bbox_deltas"] 
        anchors = self.anchor_generator(images, features)
        self.current_features = features

        num_images = len(anchors)
        num_anchors_per_level_shape_tensors = [o[0].shape for o in bce]
        num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]

        #pred_scores = concat_prediction_layers_multiple_scores(pred_scores)
        #bce, pred_bbox_deltas = concat_box_prediction_layers(bce, pred_bbox_deltas)
        pred_scores, pred_bbox_deltas = concat_box_prediction_layers_multiple_scores(heads_pred)
        # apply pred_bbox_deltas to anchors to obtain the decoded proposals
        # note that we detach the deltas because Faster R-CNN do not backprop through
        # the proposals
        proposals = self.box_coder.decode(pred_bbox_deltas.detach(), anchors)
        concat_proposals = proposals.view(-1, 4)
        proposals = proposals.view(num_images, -1, 4)
        self.current_anchors = torch.stack(anchors)

        self.current_all_proposals = proposals
        self.current_all_scores = pred_scores

        batch_size = len(targets)

        if self.objectness_score_used == "iou":
            objectness = pred_scores["IOU"]
        elif self.objectness_score_used == "centerness":
            objectness = pred_scores["centerness"]
        elif self.objectness_score_used == "bce":
            objectness = pred_scores["BCE"]
        else:
            raise ValueError(f" {self.objectness_score_used = } is not handle, only bce, iou and centerness is ok")

        boxes, scores = self.filter_proposals(proposals, objectness, images.image_sizes, num_anchors_per_level)

        # For access purpose
        self.current_scores = scores
        self.current_proposals = boxes
        self.current_anchors = anchors

        losses = {}
        if self.calculate_losses:

            concat_anchors = torch.cat(anchors, dim=0)

            if targets is None:
                raise ValueError("targets should not be None")

            # Get targets for each head and for each anchors
            labels, matched_gt_boxes = self.assign_targets_to_anchors_multiple_scores(anchors, targets)
            #labels_proposal, matched_gt_boxes_proposal = self.assign_targets_to_anchors_multiple_scores(proposals, targets, proposals=True)
            #labels.update(labels_proposal)
            regression_targets = self.box_coder.encode(matched_gt_boxes, anchors)
            labels["bbox_deltas"] = torch.cat(regression_targets, dim=0)
            pred_scores["bbox_deltas"] = pred_bbox_deltas

            losses = self.calculate_losses_with_boxes(labels, pred_scores, anchors, concat_anchors, targets, batch_size, suffix="")
            self.current_image_pure_negative = images.tensors[-1]
            #losses.update(self.calculate_losses_with_boxes(labels, pred_scores, proposals, concat_proposals, targets, batch_size, suffix="_proposal"))

        return boxes, scores, losses

