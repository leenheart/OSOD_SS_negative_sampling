from typing import Dict, List, Optional, Tuple

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torchvision.ops import boxes as box_ops, Conv2dNormActivation

from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.rpn import concat_box_prediction_layers
#from torchvision.models.detection import FCOS_ResNet50_FPN_Weights

def my_rpn_forward(
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
    objectness, pred_bbox_deltas = self.head(features)
    anchors = self.anchor_generator(images, features)

    num_images = len(anchors)
    num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
    num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]
    objectness, pred_bbox_deltas = concat_box_prediction_layers(objectness, pred_bbox_deltas)
    # apply pred_bbox_deltas to anchors to obtain the decoded proposals
    # note that we detach the deltas because Faster R-CNN do not backprop through
    # the proposals
    proposals = self.box_coder.decode(pred_bbox_deltas.detach(), anchors)
    proposals = proposals.view(num_images, -1, 4)
    boxes, scores = self.filter_proposals(proposals, objectness, images.image_sizes, num_anchors_per_level)

    losses = {}
    if self.training:
        if targets is None:
            raise ValueError("targets should not be None")
        labels, matched_gt_boxes = self.assign_targets_to_anchors(anchors, targets)
        regression_targets = self.box_coder.encode(matched_gt_boxes, anchors)
        loss_objectness, loss_rpn_box_reg = self.compute_loss(
            objectness, pred_bbox_deltas, labels, regression_targets
        )
        losses = {
            "loss_objectness": loss_objectness,
            "loss_rpn_box_reg": loss_rpn_box_reg,
        }

    return boxes, scores
    return boxes, losses
    #return (boxes, scores), losses

def my_rpn_filter_proposals(
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

    objectness_prob = torch.sigmoid(objectness)

    final_boxes = []
    final_scores = []
    for boxes, scores, lvl, img_shape in zip(proposals, objectness_prob, levels, image_shapes):
        boxes = box_ops.clip_boxes_to_image(boxes, img_shape)

        # remove small boxes
        keep = box_ops.remove_small_boxes(boxes, self.min_size)
        boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]

        # remove low scoring boxes
        # use >= for Backwards compatibility
        keep = torch.where(scores >= self.score_thresh)[0]
        boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]

        # non-maximum suppression, independently done per level
        keep = box_ops.batched_nms(boxes, scores, lvl, self.nms_thresh)

        # keep only topk scoring predictions
        keep = keep[: self.post_nms_top_n()]
        boxes, scores = boxes[keep], scores[keep]

        final_boxes.append(boxes)
        final_scores.append(scores)
    return final_boxes, final_scores
