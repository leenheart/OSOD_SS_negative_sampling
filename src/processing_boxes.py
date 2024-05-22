import torch
import torchvision
from torch import Tensor
from torchvision.ops import boxes as box_ops, box_area
from torchvision.models.detection import _utils as det_utils

from nn_models.my_roi_heads import _box_inter_union


def _get_top_n_idx(n: int, objectness: Tensor, num_anchors_per_level: list[int]) -> Tensor:
    r = []
    offset = 0
    for ob in objectness.split(num_anchors_per_level, 1):
        num_anchors = ob.shape[1]
        pre_nms_top_n = det_utils._topk_min(ob, n, 1)
        _, top_n_idx = ob.topk(pre_nms_top_n, dim=1)
        r.append(top_n_idx + offset)
        offset += num_anchors
    return torch.cat(r, dim=1)

def filter_proposals(
    proposals: Tensor,
    objectness: Tensor,
    image_shapes: list[tuple[int, int]],
    num_anchors_per_level: list[int],
    use_sigmoid = False,
    pre_nms_top_n = 1000,
    min_size = 0.001, # Same as rpn from pytorch
    score_thresh = 0.0, 
    nms_thresh = 0.7,
    post_nms_top_n = 1000,
    other_scores= {},
) -> tuple[list[Tensor], list[Tensor]]:


    num_images = proposals.shape[0]
    device = proposals.device

    levels = [
        torch.full((n,), idx, dtype=torch.int64, device=device) for idx, n in enumerate(num_anchors_per_level)
    ]
    levels = torch.cat(levels, 0)
    levels = levels.reshape(1, -1).expand_as(objectness)

    # select top_n boxes independently per level before applying nms
    top_n_idx = _get_top_n_idx(pre_nms_top_n, objectness, num_anchors_per_level)

    image_range = torch.arange(num_images, device=device)
    batch_idx = image_range[:, None]


    objectness = objectness[batch_idx, top_n_idx]
    levels = levels[batch_idx, top_n_idx]
    proposals = proposals[batch_idx, top_n_idx]

    for key, value in other_scores.items():
        other_scores[key] = value[batch_idx, top_n_idx]

    if use_sigmoid:
        objectness = torch.sigmoid(objectness)

    final_boxes = []
    final_scores = []

    final_other_scores = {}
    for key in other_scores.keys():
        final_other_scores[key] = []


    for i, (boxes, scores, lvl, img_shape) in enumerate(zip(proposals, objectness, levels, image_shapes)):
        boxes = box_ops.clip_boxes_to_image(boxes, img_shape)

        # remove small boxes
        keep = box_ops.remove_small_boxes(boxes, min_size)
        boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]
        for key, value in other_scores.items():
            final_other_scores[key].append(value[i][keep])

        # remove low scoring boxes
        # use >= for Backwards compatibility
        keep = torch.where(scores >= score_thresh)[0]
        boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]
        for key, value in final_other_scores.items():
            final_other_scores[key][i] = value[i][keep]

        # non-maximum suppression, independently done per level
        keep = box_ops.batched_nms(boxes, scores, lvl, nms_thresh)
        keep = keep[: post_nms_top_n]
        boxes, scores = boxes[keep], scores[keep]

        for key, value in final_other_scores.items():
            final_other_scores[key][i] = value[i][keep]


        final_boxes.append(boxes)
        final_scores.append(scores)


    return final_boxes, final_scores, final_other_scores

def get_KUB_masks(cfg, prediction, classes_as_known):

    #filter out only those predicted labels that are recognized and considered valid based on the known classes you have defined.
    over_known_threshold_score_mask = prediction["scores"] >= cfg.threshold_upper_score
    under_minimum_score_mask = prediction["scores"] < cfg.threshold_under_score_minimum
    classes_known_mask = torch.tensor([label in classes_as_known for label in prediction["labels"]], dtype=torch.bool, device=prediction["labels"].device)
    classes_background_mask = prediction["labels"] == 0

    # Mask with scores and classes
    if cfg.keep_low_known_classes_as_unknown:
        background_mask = under_minimum_score_mask
    else:
        background_mask = under_minimum_score_mask | (~classes_background_mask & ~over_known_threshold_score_mask)  # If removing known classes between 0.2 and 0.5
    known_mask = classes_known_mask & over_known_threshold_score_mask
    unknown_mask = ~(background_mask | known_mask)

    if cfg.remove_under_oro_score:
        # Put unknown that have oro < threshold to background 
        unknown_with_oro_too_low_mask = unknown_mask & (prediction["oro"] < cfg.oro_score_threshold) 

        #update mask
        unknown_mask = unknown_mask ^ unknown_with_oro_too_low_mask
        background_mask = background_mask | unknown_with_oro_too_low_mask

    return known_mask, unknown_mask, background_mask


def sort_predictions_boxes(cfg, predictions, classes_as_known):

    # Select only known
    background_predictions = []
    known_predictions = []
    unknown_predictions = []
    for prediction in predictions:
        nb_pred = len(prediction["labels"])

        known_prediction = {}
        unknown_prediction = {}
        background_prediction = {}

        known_mask, unknown_mask, background_mask = get_KUB_masks(cfg, prediction, classes_as_known)

        # NMS
        if cfg.nms_unknown_inside_known and unknown_mask.any() and known_mask.any():
            inter, unions = _box_inter_union(prediction["boxes"][unknown_mask], prediction["boxes"][known_mask])
            area_unknown = box_area(prediction["boxes"][unknown_mask])
            max_inter_values, max_inter_inds = inter.max(dim=1)

            unknown_inside_known_mask = ((area_unknown * 0.4).int() <= (max_inter_values).int())
            if unknown_inside_known_mask.any():
                background_mask[unknown_mask] = unknown_inside_known_mask
                unknown_mask[unknown_mask.clone()] = ~unknown_inside_known_mask

        kept_boxes_from_nms = torchvision.ops.nms(prediction["boxes"][unknown_mask], prediction['scores'][unknown_mask], cfg.nms_iou_threshold)  # NMS
        if kept_boxes_from_nms.any():
            mask_kept_unknown = torch.zeros(unknown_mask.sum(), dtype=bool, device=unknown_mask.device)
            mask_kept_unknown[kept_boxes_from_nms] = True
            background_mask[unknown_mask] = ~mask_kept_unknown
            unknown_mask[unknown_mask.clone()] = mask_kept_unknown 


        # Seperate predictions

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
