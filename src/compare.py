import json
import torch
import wandb

from torchvision.ops import box_area, box_iou
from log import get_wandb_image_with_labels   

def compare_predictions(predictions_path, models_to_compare, image_name, threshold_score=0.5):

    wandb.init()

    img_file_path = "../model_saves/prediction_saves/faster_rcnn/" + image_name + ".pt"
    loaded_image_tensor = torch.load(img_file_path)

    # Load every predictions
    models_predictions_known = {}
    models_predictions_unknown = {}
    target = None
    labels_id_names = None
    wandb_images_unknown = []
    wandb_images_known = []
    for model in models_to_compare:
        json_file_path = predictions_path + model + "/" + image_name + ".json"
        with open(json_file_path, "r") as json_file:
            loaded_data = json.load(json_file)


        labels_id_names = loaded_data["labels_id_names"]
        labels_id_names = {int(key): value for key, value in labels_id_names.items()}

        # Compare targets
        if target == None:
            target = loaded_data["targets"]
            for key in target.keys():
                target[key] = torch.tensor(target[key])
        else:
            if target != loaded_data["targets"]:
                print("[ERROR] Not same targets !")

        # Load predictions
        model_predictions = loaded_data["predictions"]

        # Get threshold index
        index_threshold = len(model_predictions["scores"])
        for i in range(len(model_predictions["scores"])):
            if model_predictions["scores"][i] < threshold_score:
                print("Found index trehold ! :", model_predictions["scores"][i], threshold_score)
                index_threshold = i
                break

        models_predictions_known[model] = {"scores": torch.tensor(model_predictions["scores"][:index_threshold]),
                                           "labels": torch.tensor(model_predictions["labels"][:index_threshold]),
                                           "boxes": torch.tensor(model_predictions["boxes"][:index_threshold])}
        models_predictions_unknown[model] = {"scores": torch.tensor(model_predictions["scores"][index_threshold:]),
                                             "labels": torch.zeros(len(model_predictions["labels"][index_threshold:]), dtype=torch.int),
                                             "boxes":  torch.tensor(model_predictions["boxes"][index_threshold:])}

        wandb_images_unknown.append(get_wandb_image_with_labels(loaded_image_tensor[0],
                                               target,
                                               models_predictions_unknown[model],
                                               pred_class_id_label={0: "?"},
                                               gt_class_id_label=labels_id_names))

        wandb_images_known.append(get_wandb_image_with_labels(loaded_image_tensor[0],
                                               target,
                                               models_predictions_known[model],
                                               pred_class_id_label=labels_id_names,
                                               gt_class_id_label=labels_id_names))
   
    
    wandb.log({"Images/unknown": wandb_images_unknown})
    wandb.log({"Images/known": wandb_images_known})

    # Compare two by two and keep only boxes matched for all models
    boxes = None 
    scores = None 
    for model in models_to_compare:
        if boxes == None:
            boxes = models_predictions_unknown[model]["boxes"]
            scores = models_predictions_unknown[model]["scores"]
            continue
        new_boxes = []
        new_scores = []
        boxes_iou = box_iou(boxes, models_predictions_unknown[model]["boxes"])
        for prediction_index, boxe in enumerate(boxes):
            best_match_index = torch.argmax(boxes_iou[prediction_index]).detach()
            if boxes_iou[prediction_index][best_match_index] >= 0.5:
                new_boxes.append(boxe)
                new_scores.append(scores[prediction_index])
                new_boxes.append(models_predictions_unknown[model]["boxes"][best_match_index])
                new_scores.append(models_predictions_unknown[model]["scores"][best_match_index])


        print(new_boxes, new_scores)
        boxes = torch.stack(new_boxes)
        scores = torch.stack(new_scores)
        print(boxes, scores)

    concatenate_unknown_predictions = {"boxes": boxes, "scores": scores, "labels": torch.zeros(len(scores), dtype=torch.int)}
    stack_unknown_wandb_image = get_wandb_image_with_labels(loaded_image_tensor[0],
                                               target,
                                               concatenate_unknown_predictions,
                                               pred_class_id_label={0: "?"},
                                               gt_class_id_label=labels_id_names)
    
    wandb.log({"Images/stack_unknown": stack_unknown_wandb_image})

    # Compare two by two and keep every matched boxes
    boxes = [] 
    scores = [] 
    for first_model in models_to_compare:

        for second_model in models_to_compare:

            if first_model == second_model:
                continue

            boxes_iou = box_iou(models_predictions_unknown[first_model]["boxes"], models_predictions_unknown[second_model]["boxes"])

            for prediction_index, boxe in enumerate(models_predictions_unknown[first_model]["boxes"]):

                best_match_index = torch.argmax(boxes_iou[prediction_index]).detach()

                if boxes_iou[prediction_index][best_match_index] >= 0.5:
                    boxes.append(models_predictions_unknown[first_model]["boxes"][prediction_index])
                    scores.append(models_predictions_unknown[first_model]["scores"][prediction_index])
                    boxes.append(models_predictions_unknown[second_model]["boxes"][best_match_index])
                    scores.append(models_predictions_unknown[second_model]["scores"][best_match_index])


    boxes = torch.stack(boxes)
    scores = torch.stack(scores)

    concatenate_unknown_predictions = {"boxes": boxes, "scores": scores, "labels": torch.zeros(len(scores), dtype=torch.int)}
    stack_unknown_wandb_image = get_wandb_image_with_labels(loaded_image_tensor[0],
                                               target,
                                               concatenate_unknown_predictions,
                                               pred_class_id_label={0: "?"},
                                               gt_class_id_label=labels_id_names)
    
    
    wandb.log({"Images/stack_unknown": stack_unknown_wandb_image})


    # Compare two by two and keep every matched boxes and keeps only the highter score
    boxes = [] 
    scores = [] 
    for first_model in models_to_compare:

        for second_model in models_to_compare:

            if first_model == second_model:
                continue

            boxes_iou = box_iou(models_predictions_unknown[first_model]["boxes"], models_predictions_unknown[second_model]["boxes"])

            for prediction_index, boxe in enumerate(models_predictions_unknown[first_model]["boxes"]):

                best_match_index = torch.argmax(boxes_iou[prediction_index]).detach()

                if boxes_iou[prediction_index][best_match_index] >= 0.5:
                    if models_predictions_unknown[first_model]["scores"][prediction_index] >= models_predictions_unknown[second_model]["scores"][best_match_index]:
                        boxes.append(models_predictions_unknown[first_model]["boxes"][prediction_index])
                        scores.append(models_predictions_unknown[first_model]["scores"][prediction_index])
                    else :
                        boxes.append(models_predictions_unknown[second_model]["boxes"][best_match_index])
                        scores.append(models_predictions_unknown[second_model]["scores"][best_match_index])


    boxes = torch.stack(boxes)
    scores = torch.stack(scores)

    concatenate_unknown_predictions = {"boxes": boxes, "scores": scores, "labels": torch.zeros(len(scores), dtype=torch.int)}
    stack_unknown_wandb_image = get_wandb_image_with_labels(loaded_image_tensor[0],
                                               target,
                                               concatenate_unknown_predictions,
                                               pred_class_id_label={0: "?"},
                                               gt_class_id_label=labels_id_names)
    
    
    wandb.log({"Images/stack_unknown": stack_unknown_wandb_image})


    # Compare two by two and keep every matched boxes and keeps only the highter score and remove known boxes with iou > 0.5 of unknown
    filtered_boxes = [] 
    filtered_scores = [] 
    removed_boxes = []
    for model in models_to_compare:

            boxes_iou = box_iou(boxes, models_predictions_known[model]["boxes"])

            for prediction_index, boxe in enumerate(boxes):

                if boxes[prediction_index].tolist() in removed_boxes:
                    continue

                best_match_index = torch.argmax(boxes_iou[prediction_index]).detach()

                if boxes_iou[prediction_index][best_match_index] >= 0.3:
                    removed_boxes.append(boxe.tolist())

    for prediction_index, boxe in enumerate(boxes):
        if boxe.tolist() in removed_boxes:
            continue

        filtered_boxes.append(boxe)
        filtered_scores.append(scores[prediction_index])


    filtered_boxes = torch.stack(filtered_boxes)
    filtered_scores = torch.stack(filtered_scores)
    print("Boxes: ", boxes, scores)
    print("Filtered: ", filtered_boxes, filtered_scores)

    filtered_concatenate_unknown_predictions = {"boxes": filtered_boxes, "scores": filtered_scores, "labels": torch.ones(len(filtered_scores), dtype=torch.int)}
    filtered_stack_unknown_wandb_image = get_wandb_image_with_labels(loaded_image_tensor[0],
                                               target,
                                               filtered_concatenate_unknown_predictions,
                                               pred_class_id_label={1: "?!"},
                                               gt_class_id_label=labels_id_names)
    
    
    wandb.log({"Images/stack_unknown": filtered_stack_unknown_wandb_image})

    print("FINISH")


    






