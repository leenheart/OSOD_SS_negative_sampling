from typing import Dict, List, Optional, Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np



def eager_outputs_both_losses_and_detections(self, losses, detections):
    return (detections, losses)

def rcnn_forward_only_rpn(self, images, targets=None):
    # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
    """
    Args:
        images (list[Tensor]): images to be processed
        targets (list[Dict[str, Tensor]]): ground-truth boxes present in the image (optional)

    Returns:
        result (list[BoxList] or dict[Tensor]): the output from the model.
            During training, it returns a dict[Tensor] which contains the losses.
            During testing, it returns list[BoxList] contains additional fields
            like `scores`, `labels` and `mask` (for Mask R-CNN models).

    """
    if self.training:
        if targets is None:
            torch._assert(False, "targets should not be none when in training mode")
        else:
            for target in targets:
                boxes = target["boxes"]
                if isinstance(boxes, torch.Tensor):
                    torch._assert(
                        len(boxes.shape) == 2 and boxes.shape[-1] == 4,
                        f"Expected target boxes to be a tensor of shape [N, 4], got {boxes.shape}.",
                    )
                else:
                    torch._assert(False, f"Expected target boxes to be of type Tensor, got {type(boxes)}.")

    original_image_sizes: List[Tuple[int, int]] = []
    for img in images:
        val = img.shape[-2:]
        torch._assert(
            len(val) == 2,
            f"expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}",
        )
        original_image_sizes.append((val[0], val[1]))


    images_before = images.copy()
    images, targets = self.transform(images, targets)
    image_size = images.tensors[-1].shape[-2:]

    images_before = torch.stack(images_before)
    if len(images_before.shape) != 4:
        images_before = images_before.unsqueeze(0)
    images.tensors = F.interpolate(images_before, size=image_size, mode='bilinear',  align_corners=False)

    # Resize masks
    if "semantic_segmentation" in targets[0].keys():
        for i in range(len(targets)):
            targets[i]["semantic_segmentation_binary_objects"] = F.interpolate(targets[i]["semantic_segmentation_binary_objects"].unsqueeze(0).unsqueeze(0), size=image_size, mode='nearest').squeeze(0).squeeze(0)
            targets[i]["semantic_segmentation"] = F.interpolate(targets[i]["semantic_segmentation"].unsqueeze(0).unsqueeze(0), size=image_size, mode='nearest').squeeze(0).squeeze(0)


    # Check for degenerate boxes
    # TODO: Move this to a function
    if targets is not None:
        for target_idx, target in enumerate(targets):
            boxes = target["boxes"]
            degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
            if degenerate_boxes.any():
                # print the first degenerate box
                bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                degen_bb: List[float] = boxes[bb_idx].tolist()
                torch._assert(
                    False,
                    "All bounding boxes should have positive height and width."
                    f" Found invalid box {degen_bb} for target at index {target_idx}.",
                )

    features = self.backbone(images.tensors)
    if isinstance(features, torch.Tensor):
        features = OrderedDict([("0", features)])
    proposals, proposal_scores, proposal_losses = self.rpn(images, features, targets)


    # Transform proposals into detection :
    detections = []
    for proposal, proposal_score in zip(proposals, proposal_scores):
        detections.append({"boxes": proposal, "labels": torch.zeros(len(proposal), device=proposal.device, dtype=int), "scores": proposal_score})

    #detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
    detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)  # type: ignore[operator]

    losses = {}
    #losses.update(detector_losses)
    losses.update(proposal_losses)

    if torch.jit.is_scripting():
        if not self._has_warned:
            warnings.warn("RCNN always returns a (Losses, Detections) tuple in scripting")
            self._has_warned = True
        return losses, detections
    else:
        return self.eager_outputs(losses, detections)


def rcnn_forward_with_proposals(self, images, targets=None):
    # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
    """
    Args:
        images (list[Tensor]): images to be processed
        targets (list[Dict[str, Tensor]]): ground-truth boxes present in the image (optional)

    Returns:
        result (list[BoxList] or dict[Tensor]): the output from the model.
            During training, it returns a dict[Tensor] which contains the losses.
            During testing, it returns list[BoxList] contains additional fields
            like `scores`, `labels` and `mask` (for Mask R-CNN models).

    """
    if self.training:
        if targets is None:
            torch._assert(False, "targets should not be none when in training mode")
        else:
            for target in targets:
                boxes = target["boxes"]
                if isinstance(boxes, torch.Tensor):
                    torch._assert(
                        len(boxes.shape) == 2 and boxes.shape[-1] == 4,
                        f"Expected target boxes to be a tensor of shape [N, 4], got {boxes.shape}.",
                    )
                else:
                    torch._assert(False, f"Expected target boxes to be of type Tensor, got {type(boxes)}.")

    original_image_sizes: List[Tuple[int, int]] = []
    for img in images:
        val = img.shape[-2:]
        torch._assert(
            len(val) == 2,
            f"expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}",
        )
        original_image_sizes.append((val[0], val[1]))


    images_before = images.copy()
    images, targets = self.transform(images, targets)
    image_size = images.tensors[-1].shape[-2:]

    images_before = torch.stack(images_before)
    if len(images_before.shape) != 4:
        images_before = images_before.unsqueeze(0)
    images.tensors = F.interpolate(images_before, size=image_size, mode='bilinear',  align_corners=False)

    # Resize masks
    if "semantic_segmentation" in targets[0].keys():
        for i in range(len(targets)):
            targets[i]["semantic_segmentation_binary_objects"] = F.interpolate(targets[i]["semantic_segmentation_binary_objects"].unsqueeze(0).unsqueeze(0), size=image_size, mode='nearest').squeeze(0).squeeze(0)
            targets[i]["semantic_segmentation"] = F.interpolate(targets[i]["semantic_segmentation"].unsqueeze(0).unsqueeze(0), size=image_size, mode='nearest').squeeze(0).squeeze(0)


    # Check for degenerate boxes
    # TODO: Move this to a function
    if targets is not None:
        for target_idx, target in enumerate(targets):
            boxes = target["boxes"]
            degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
            if degenerate_boxes.any():
                # print the first degenerate box
                bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                degen_bb: List[float] = boxes[bb_idx].tolist()
                torch._assert(
                    False,
                    "All bounding boxes should have positive height and width."
                    f" Found invalid box {degen_bb} for target at index {target_idx}.",
                )

    features = self.backbone(images.tensors)
    if isinstance(features, torch.Tensor):
        features = OrderedDict([("0", features)])
    proposals, proposal_scores, proposal_losses = self.rpn(images, features, targets)

    detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
    detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)  # type: ignore[operator]


    # Transform proposals into detection :
    proposals_detections = []
    for proposal, proposal_score in zip(proposals, proposal_scores):
        proposals_detections.append({"boxes": proposal, "labels": torch.zeros(len(proposal), device=proposal.device, dtype=int), "scores": proposal_score})
    proposals_detections = self.transform.postprocess(proposals_detections, images.image_sizes, original_image_sizes)  # type: ignore[operator]

    losses = {}
    losses.update(detector_losses)
    losses.update(proposal_losses)

    return losses, detections, proposals_detections

    if torch.jit.is_scripting():
        if not self._has_warned:
            warnings.warn("RCNN always returns a (Losses, Detections) tuple in scripting")
            self._has_warned = True
        return losses, detections
    else:
        return self.eager_outputs(losses, detections)

