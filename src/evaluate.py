import wandb
import torch
import torchvision
import copy
import numpy as np

from log import get_wandb_box

def remove_less_good_box_inside(index_box, index_to_remove, res, iou_threshold=0.5):

    ious = torch.squeeze(torchvision.ops.box_iou(torch.unsqueeze(res["boxes"][index_box], 0), res["boxes"]))

    for j, score in enumerate(res["scores"]):

        if j == index_box:
            continue

        if torch.all(res["boxes"][j] == res["boxes"][index_box]) and res["labels"][j] == res["labels"][index_box]:
            continue

        if ious[j] > iou_threshold:
            index_to_remove.append(j)

    return index_to_remove

def separate_unknown(results, nb_classes, nms_iou_threshold, threshold_score_centerness_unknown, threshold_score_remove, threshold_score_good, activate_my_custom_nms=False):

    good_preds_batch = []
    unknown_preds_batch = []

    # Change the classe to 3 new classe = ["removes", "zones_unknowns", "unknown"]
    for res in results:
        # save old labels
        res["modified_labels"] = copy.deepcopy(res["labels"])

        first = True
        bad_preds_index = len(res["scores"])
        index_to_remove = []
        for i, score in enumerate(res["scores"]):

            # Ignore index_to_remove
            if i in index_to_remove:
                continue

            if score >= threshold_score_good:
                if activate_my_custom_nms:
                    #remove all inside or near boxes from good one
                    index_to_remove = remove_less_good_box_inside(i, index_to_remove, res, iou_threshold=nms_iou_threshold)
                continue

            if first:
                bad_preds_index = i
                first = False

            if score < threshold_score_remove:
                index_to_remove.append(i)
                continue

            if "scores_centerness" in res:
                if res["scores_centerness"][i] > threshold_score_centerness_unknown:
                    res["labels"][i] = nb_classes - 1
                else:
                    res["labels"][i] = nb_classes - 2

        for i in index_to_remove:
            res["labels"][i] = nb_classes - 3
            res["scores"][i] = 0


        good_preds = {}
        unknown_preds = {}
        mask_not_removed_good_preds = res["labels"][:bad_preds_index] != nb_classes - 3
        mask_not_removed_bad_preds = res["labels"][bad_preds_index:] != nb_classes - 3
        for key in res:
            good_preds[key] = res[key][:bad_preds_index][mask_not_removed_good_preds]
            unknown_preds[key] = res[key][bad_preds_index:][mask_not_removed_bad_preds]


        good_preds_batch.append(good_preds)
        unknown_preds_batch.append(unknown_preds)
            
    return good_preds_batch, unknown_preds_batch, results

def evaluate_batch(batch, model, nb_classes, unknown=False):

    with torch.no_grad():
        results = model(batch[0])

    if unknown:
        results = convert_to_unknown(results)

    return results
