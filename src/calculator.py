import torch
import wandb
from torch import Tensor
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np
from torchvision.models.detection.anchor_utils import AnchorGenerator
from PIL import Image


from metrics import MetricModule
from postprocessing_predictions import postprocess_predictions, extract_batched_predictions_with_nms
from postprocessing_predictions import set_tags, set_tags_opti, seperate_predictions_with_threshold_score, seperate_predictions_into_known_and_unknown, get_only_known_targets, get_only_unknown_targets, get_only_background_targets
from processing_boxes import sort_predictions_boxes, filter_proposals
from log import get_wandb_image_with_labels

class Calculator(pl.LightningModule):

    def __init__(self, models_name, print_model_names, cfg_metrics, cfg_sort_predictions, classes_as_known, log_randoms=False, image_path="", is_log=False):
        super().__init__()
        self.models_name = models_name
        self.print_model_names = print_model_names 
        self.metrics_modules = {}
        self.scores_cfg = cfg_metrics.scores
        self.sort_prediction_cfg = cfg_sort_predictions
        self.classes_as_known = classes_as_known
        self.considering_known_classes = cfg_metrics.considering_known_classes 

        self.image_path = image_path
        self.log_randoms = log_randoms
        self.is_log = is_log

        # For random generation
        self.num_proposals = 10000
        self.scale = 2

        #self.filtering_config = ["no_change", "bce", "bce_proposal", "iou", "iou_proposal", "centerness", "centerness_proposal", "bce_propXanchors", "mean_BCEs", "sqrt_BCEs", "sqrt_BCE_X_iou"]
        #self.filtering_config = ["bce", "bce_proposal", "iou", "iou_proposal", "centerness", "centerness_proposal", "sqrt_BCEs", "sqrt_BCE_X_iou", "mean_BCE_X_iou", "mean_BCE_X_centerness", "mean_iou_X_centerness", "mean_iou_X_BCE_X_centerness", "BCE_X_iou", "BCE_X_centerness", "iou_X_centerness", "iou_X_BCE_X_centerness", "all_score_multiply" ]
        #self.filtering_config = ["bce", "iou", "centerness", "sqrt_BCE_X_iou", "mean_BCE_X_iou", "mean_BCE_X_centerness", "mean_iou_X_centerness", "BCE_X_iou", "BCE_X_centerness", "iou_X_centerness", "iou_X_BCE_X_centerness", "mean_iou_X_BCE_X_centerness", "sqrt_iou_X_BCE_X_centerness"]
        #self.filtering_config = ["bce", "iou", "centerness", "iou_X_centerness", "iou_X_BCE_X_centerness", "mean_iou_X_BCE_X_centerness", "max_iou_bce_centerness", "mean_iou_X_BCE_X_centerness_normalised", "max_iou_bce_centerness_normalised", "sqrt_iou_X_BCE_X_centerness_normalised", "iou_X_BCE_X_centerness_normalised"]
        self.filtering_config = ["bce", "iou", "centerness"]
        self.filtering_config_base = ["bce", "iou", "centerness"]
        self.alpha_bce = cfg_sort_predictions.alpha_bce
        self.alpha_iou = cfg_sort_predictions.alpha_iou
        self.alpha_centerness = cfg_sort_predictions.alpha_centerness

        for model_name in self.models_name:
            self.metrics_modules[model_name] = {}
            for config in self.filtering_config:
                self.metrics_modules[model_name][config] = MetricModule(self.scores_cfg, cfg_metrics)


        self.anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
        self.aspect_ratios = ((0.5, 1.0, 2.0),) * len(self.anchor_sizes)

        self.class_label = {}
        for i in range(50):
            self.class_label[i] = str(i)

        if "anchors_custom" in models_name:
            #anchor_sizes = ((16,), (32,), (64,), (128,), (256,))
            #anchor_sizes = ((16,), (64,), (128,), (256,), (512,))
            self.anchors_dist_is_evently = False

            if self.anchors_dist_is_evently:
                h, w = (749, 1333)
                self.anchor_sizes = ( (int(0.1*h),), (int(0.2*h),), (int(0.3*h),), (int(0.4*h),), (int(0.5*h),), (int(0.6*h),), (int(0.7*h),), (int(0.8*h),), (int(0.9*h),), (int(h),)) 
                self.aspect_ratios = ((0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9, 1),) * len(self.anchor_sizes)
            else:
                self.anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
                #self.aspect_ratios = ((0.45, 1.0, 2.8),) * len(self.anchor_sizes)

                #self.anchor_sizes = ((23,), (46,), (92,), (184,), (368,))
                #self.aspect_ratios = ((0.5, 1.0, 2),  (0.3, 1.0, 2.6), (0.3, 1.0, 2.6), (0.5, 1.0, 2.0), (0.5, 1.0, 2.0))

            print(" anchors size :", self.anchor_sizes)
            print(" anchors ratio :", self.aspect_ratios)


            self.anchor_generator = AnchorGenerator(self.anchor_sizes, self.aspect_ratios)
            AnchorGenerator.forward = my_anchor_generator_forward
            #AnchorGenerator.grid_anchors = my_grid_anchors

        print(" Models : ", self.models_name)



    def on_test_epoch_start(self):

        for model_name, print_model_name in zip(self.models_name, self.print_model_names):
            for filtering_config in self.filtering_config:
                metrics_module = self.metrics_modules[model_name][filtering_config]
                metrics_module.reset()

        self.pins = 10000
        self.area_targets_counts = None
        self.area_targets_bins = None
        self.width_targets_counts = None
        self.width_targets_bins = None
        self.heigth_targets_counts = None
        self.heigth_targets_bins = None

        self.area_preds_counts = None
        self.area_preds_bins = None
        self.width_preds_counts = None
        self.width_preds_bins = None
        self.heigth_preds_counts = None
        self.heigth_preds_bins = None

    def update_data_saving(self, targets, predictions):

        for target, prediction in zip(targets, predictions):

            nb_channel, img_height, img_width = target["img_size"]

            # Init
            if self.area_targets_counts == None:
                self.area_targets_counts = torch.zeros((self.pins), device=self.device, dtype=int)
                self.area_targets_bins = torch.linspace(0, img_height*img_width, self.pins+1, device=self.device)
                self.width_targets_counts = torch.zeros((self.pins), device=self.device, dtype=int)
                self.width_targets_bins = torch.linspace(0, img_width, self.pins+1, device=self.device)
                self.height_targets_counts = torch.zeros((self.pins), device=self.device, dtype=int)
                self.height_targets_bins = torch.linspace(0, img_height, self.pins+1, device=self.device)

                self.targets_positions_pos = ([], [])
                self.targets_positions_neg = ([], [])
                self.targets_coordinates_neg = ([], [])
                self.anchors_positions = ([], [])
                self.anchors_coordinates = ([], [])

                self.area_preds_counts = torch.zeros((self.pins), device=self.device, dtype=int)
                self.area_preds_bins = torch.linspace(0, img_height*img_width, self.pins+1, device=self.device)
                self.width_preds_counts = torch.zeros((self.pins), device=self.device, dtype=int)
                self.width_preds_bins = torch.linspace(0, img_width, self.pins+1, device=self.device)
                self.height_preds_counts = torch.zeros((self.pins), device=self.device, dtype=int)
                self.height_preds_bins = torch.linspace(0, img_height, self.pins+1, device=self.device)

            # Targets

            boxes = target["boxes"][target["tags_raw"]]
            width = boxes[:, 2] - boxes[:, 0]
            height = boxes[:, 3] - boxes[:, 1]

            # Save boxes sizes
            self.targets_positions_pos[0].append(width.cpu().numpy())
            self.targets_positions_pos[1].append(height.cpu().numpy())

            boxes = target["boxes"][~target["tags_raw"]]
            width = boxes[:, 2] - boxes[:, 0]
            height = boxes[:, 3] - boxes[:, 1]
            area = width * height

            self.targets_positions_neg[0].append(width.cpu().numpy())
            self.targets_positions_neg[1].append(height.cpu().numpy())

            self.targets_coordinates_neg[0].append((boxes[:, 0] + (width / 2)).cpu().numpy())
            self.targets_coordinates_neg[1].append((boxes[:, 1] + (height / 2)).cpu().numpy())

            """
            indices = torch.searchsorted(self.area_targets_bins, area) - 1
            self.area_targets_counts.scatter_add_(0, indices, torch.ones_like(indices))

            indices = torch.searchsorted(self.width_targets_bins, width) - 1
            self.width_targets_counts.scatter_add_(0, indices, torch.ones_like(indices))

            indices = torch.searchsorted(self.height_targets_bins, height) - 1
            self.height_targets_counts.scatter_add_(0, indices, torch.ones_like(indices))
            """

            # Predictions
            boxes = prediction["boxes"]
            width = boxes[:, 2] - boxes[:, 0]
            height = boxes[:, 3] - boxes[:, 1]
            area = width * height

            if len(self.anchors_positions[0]) == 0: 
                self.anchors_positions[0].append(width.cpu().numpy())
                self.anchors_positions[1].append(height.cpu().numpy())
                self.anchors_coordinates[0].append((boxes[:, 0] + (width / 2)).cpu().numpy())
                self.anchors_coordinates[1].append((boxes[:, 1] + (height / 2)).cpu().numpy())

            indices = torch.searchsorted(self.area_preds_bins, area) - 2
            self.area_preds_counts.scatter_add_(0, indices, torch.ones_like(indices))

            indices = torch.searchsorted(self.width_preds_bins, width) - 2
            self.width_preds_counts.scatter_add_(0, indices, torch.ones_like(indices))

            indices = torch.searchsorted(self.height_preds_bins, height) - 2
            self.height_preds_counts.scatter_add_(0, indices, torch.ones_like(indices))

    def get_randoms_predictions(self, targets):
        
        predictions = []
        random_boxes_wandb_images = []
        for i in range(batch_size):
            prediction = {}
            nb_channel, img_height, img_width = targets[i]["img_size"]

            random_boxes = torch.randn((self.num_proposals, 4), device=self.device) #wyhw
            random_boxes.div_(self.scale).add_(0.5).clamp_(0.0, 1.0)
            random_boxes[:, :2].sub_(random_boxes[:, 2:] / 2)
            random_boxes.mul_(torch.tensor([img_width, img_height, img_width, img_height], device=self.device))
            random_boxes[:, [0, 2]].clamp_(min=0.0, max=img_width)
            random_boxes[:, [1, 3]].clamp_(min=0.0, max=img_height)

            prediction["boxes"] = random_boxes

            prediction["labels"] = torch.zeros((self.num_proposals), device=self.device, dtype=torch.uint8)
            prediction["scores"] = torch.randn((self.num_proposals), device=self.device, dtype=torch.float16)
            predictions.append(random_boxes)

        return predictions

    def log_images(self, targets, model_name, predictions, model_name_print, filtering_name=""):

        raw_nn_wandb_images = []
        raw_anchors_wandb_images = []
        raw_target_miss_wandb_imgs = []
        for batch_index, target in enumerate(targets):

            nb_channel, img_height, img_width = target["img_size"]
            white_img = np.ones((img_height, img_width, nb_channel), dtype=np.uint8)


            if (model_name == "anchors" or model_name == "random") and self.is_log:

                positive_anchors_boxes = predictions[batch_index]["boxes"][predictions[batch_index]["tags_raw"]]
                nb_pred = positive_anchors_boxes.shape[0]
                positive_anchors = {"boxes": positive_anchors_boxes, "labels": torch.zeros((nb_pred), device=self.device, dtype=torch.uint8), "scores": torch.zeros((nb_pred), device=self.device)}
                raw_anchors_wandb_images.append(get_wandb_image_with_labels(white_img, target, positive_anchors, self.class_label, img_shape=(img_height, img_width)))

            negative_targets_boxes = target["boxes"][~target["tags_raw"]]

            img_path = self.image_path + target["name"] + ".jpg"
            image = Image.open(img_path)
            img_x, img_y = image.size 
            image = image.resize((img_width, img_height), Image.Resampling.LANCZOS)

            if len(negative_targets_boxes) > 0:
                nb_false_target = negative_targets_boxes.shape[0]
                negative_targets = {"boxes": negative_targets_boxes , "labels": torch.zeros((nb_false_target), device=self.device, dtype=torch.uint8), "scores": torch.zeros((nb_false_target), device=self.device)}
                raw_target_miss_wandb_imgs.append(get_wandb_image_with_labels(image, target, negative_targets, self.class_label, img_shape=(img_height, img_width)))

            raw_nn_wandb_images.append(get_wandb_image_with_labels(image, target, predictions[batch_index], self.class_label, img_shape=(img_height, img_width)))



        if (model_name == "anchors" or model_name == "random") and self.is_log:
            wandb.log({("Images/" + model_name_print + "_raw_predictions_positive"): raw_anchors_wandb_images})
        wandb.log({("Images/" + model_name_print + "_raw_negatives_targets"): raw_target_miss_wandb_imgs})
        wandb.log({("Images/" + model_name_print + "_raw_predictions_" + filtering_name): raw_nn_wandb_images})

    # TODO make it configurable (cfg in init)
    def is_model_without_score(self, model_name):
        return model_name == "anchors" or model_name == "random" or model_name == 'anchors_custom' or model_name == "proposals"

    def get_prediction_model(self, model_name, targets, models_predictions):

        if model_name == "random":
            predictions = self.get_randoms_predictions(targets)

        elif model_name == "anchors_custom":
            _, img_height, img_width = targets[0]["img_size"]
            images_sizes = (img_height, img_width)

            features = []
            device = self.device
            features.append(torch.zeros((2, 256, 184, 336), device=device))
            features.append(torch.zeros((2, 256, 92, 168),  device=device))
            features.append(torch.zeros((2, 256, 46, 92),   device=device))
            features.append(torch.zeros((2, 256, 23, 42),   device=device))
            features.append(torch.zeros((2, 256, 12, 21),   device=device))

            features = [features] * batch_size

            predictions = self.anchor_generator(images_sizes, features, anchors_dist_is_evently=self.anchors_dist_is_evently, clamp=False)

        else:
            predictions = models_predictions[model_name]

        # Reformat 
        if self.is_model_without_score(model_name):
            formated_predictions = [] 
            for prediction in predictions:
                nb_pred = prediction.shape[0]
                formated_predictions.append({"boxes": prediction.int(), "labels": torch.zeros((nb_pred), device=self.device, dtype=torch.uint8), "scores": torch.zeros((nb_pred), device=self.device)})

            predictions = formated_predictions

        return predictions

    def get_filtered_predictions(self, predictions, config, images_shapes):

        if config == "no_change":
            return predictions["filtered_predictions"]
        
        proposals = predictions["proposals"]
        scores = predictions["scores"]
        batch_size = len(proposals)
        num_anchors_per_level = [193536, 48384, 12096, 3024, 756]
        proposals = torch.stack(proposals, dim=0)

        if "BCE" in scores:
            bce = torch.sigmoid(torch.stack(scores["BCE"], dim=0).squeeze(-1)) * self.alpha_bce
            bce_normalized = (bce - bce.min().item()) / (bce.max().item() - bce.min().item()) * self.alpha_bce
        if "BCE_proposal" in scores:
            bce_proposal = torch.sigmoid(torch.stack(scores["BCE_proposal"], dim=0).squeeze(-1))
        if "IOU" in scores:
            iou = torch.stack(scores["IOU"], dim=0).squeeze(-1) * self.alpha_iou
            iou_normalized = (iou - iou.min().item()) / (iou.max().item() - iou.min().item()) * self.alpha_iou
        if "IOU_proposal" in scores:
            iou_proposal = torch.stack(scores["IOU_proposal"], dim=0).squeeze(-1)
        if "centerness" in scores:
            centerness = torch.stack(scores["centerness"], dim=0).squeeze(-1) * self.alpha_centerness
            centerness_normalized = (centerness - centerness.min().item()) / (centerness.max().item() - centerness.min().item()) * self.alpha_centerness
        if "centerness_proposal" in scores:
            centerness_proposal = torch.stack(scores["centerness_proposal"], dim=0).squeeze(-1)

            
        # Unique score
        if config == "bce":
            objectness = bce
        elif config == "bce_proposal":
            objectness = bce_proposal
        elif config == "iou":
            objectness = iou
        elif config == "iou_proposal":
            objectness = iou_proposal
        elif config == "centerness":
            objectness = centerness
        elif config == "centerness_proposal":
            objectness = centerness_proposal

        # Fusion scores

        #   BCEs
        elif config == "bce_propXanchors":
            objectness = bce * bce_proposal
        elif config == "mean_BCEs":
            objectness = (bce + bce_proposal) / 2
        elif config == "sqrt_BCEs":
            objectness = torch.sqrt(bce * bce_proposal)

        elif config == "sqrt_BCE_X_iou":
            objectness = torch.sqrt(bce * iou)
        elif config == "mean_BCE_X_iou":
            objectness = (bce + iou) / 2
        elif config == "mean_BCE_X_centerness":
            objectness = (bce + centerness) / 2
        elif config == "mean_iou_X_centerness":
            objectness = (iou + centerness) / 2
        elif config == "mean_iou_X_BCE_X_centerness":
            objectness = (iou + bce + centerness) / 3
        elif config == "sqrt_iou_X_BCE_X_centerness":
            objectness = torch.pow((iou * bce * centerness), 1/3)
        elif config == "max_iou_bce_centerness":
            objectness = torch.max(torch.max(iou, bce), centerness)


        elif config == "iou_X_BCE_X_centerness":
            objectness = iou * bce * centerness
        elif config == "BCE_X_iou":
            objectness = bce * iou
        elif config == "BCE_X_centerness":
            objectness = bce * centerness
        elif config == "iou_X_centerness":
            objectness = iou * centerness
        elif config == "all_score_multiply":
            objectness = bce * iou * centerness * bce_proposal * iou_proposal * centerness_proposal

        elif config == "mean_iou_X_BCE_X_centerness_normalised":
            objectness = (iou_normalized + bce_normalized + centerness_normalized) / 3
        elif config == "max_iou_bce_centerness_normalised":
            objectness = torch.max(torch.max(iou_normalized, bce_normalized), centerness_normalized)
        elif config == "sqrt_iou_X_BCE_X_centerness_normalised":
            objectness = torch.pow((iou_normalized * bce_normalized * centerness_normalized), 1/3)
        elif config == "iou_X_BCE_X_centerness_normalised":
            objectness = iou_normalized * bce_normalized * centerness_normalized

        else:
            raise ValueError("Does not handle filter config " + config)

        old_proposals = proposals.clone()
        filtered_proposals, filtered_scores, filtered_other_scores = filter_proposals(proposals, objectness, images_shapes, num_anchors_per_level, other_scores={"iou": iou, "centerness": centerness, "bce": bce, "iou_normalized": iou_normalized, "centerness_normalized": centerness_normalized, "bce_normalized": bce_normalized}) 


        filtered_predictions = []
        for i in range(batch_size):
            filtered_predictions.append({"boxes": filtered_proposals[i],
                                         "labels": torch.zeros((len(filtered_scores[i])), device=self.device, dtype=torch.uint8),
                                         "scores": filtered_scores[i],
                                         "iou": filtered_other_scores["iou"][i],
                                         "centerness": filtered_other_scores["centerness"][i],
                                         "bce": filtered_other_scores["bce"][i],
                                         "iou_normalized": filtered_other_scores["iou_normalized"][i],
                                         "centerness_normalized": filtered_other_scores["centerness_normalized"][i],
                                         "bce_normalized": filtered_other_scores["bce_normalized"][i],
                                        })

        return filtered_predictions

    def test_step(self, batch: tuple[list[dict[str, Tensor]], dict[str, list[dict[str, Tensor]]]], batch_index: int):

        if batch_index >= 1:
            self.log_randoms = False
            self.is_log = False

        targets, models_predictions = batch
        batch_size = len(targets)
        images_shapes = [targets[0]["img_size"][-2:]] * batch_size

        # For each models
        for model_name, model_name_print in zip(self.models_name, self.print_model_names):

            # Get Predictions 
            all_predictions = self.get_prediction_model(model_name, targets, models_predictions)

            # For each config
            for filtering_config in self.filtering_config:
                metrics_module = self.metrics_modules[model_name][filtering_config]

                # Filter if we have scores
                if not self.is_model_without_score(model_name):
                    predictions = self.get_filtered_predictions(all_predictions, filtering_config, images_shapes)


                # Match for raw predictions
                for batch_index, target in enumerate(targets):
                    set_tags_opti(predictions, targets, batch_index, self.scores_cfg.iou_threshold, considering_classes=False, tags_name="tags_raw", add_centerness=False)

                # Seperate Known Unknown and background
                known_targets = get_only_known_targets(targets)
                unknown_targets = get_only_unknown_targets(targets)
                background_targets = get_only_background_targets(targets)

                known_predictions, unknown_predictions, background_predictions = sort_predictions_boxes(self.sort_prediction_cfg, predictions, self.classes_as_known)

                # Match predictions and targets
                for batch_index in range(batch_size):
                    set_tags_opti(known_predictions, known_targets, batch_index, self.scores_cfg.iou_threshold, self.considering_known_classes)
                    set_tags_opti(unknown_predictions, unknown_targets, batch_index, self.scores_cfg.iou_threshold, considering_classes=False)
                    set_tags_opti(known_predictions, unknown_targets, batch_index, self.scores_cfg.iou_threshold, considering_classes=False, tags_name="tags_KP_with_UT")

                # Update Metrics
                metrics_module.update_raw(known_targets, unknown_targets, predictions, targets)
                metrics_module.update(known_targets, unknown_targets, background_targets, known_predictions, unknown_predictions, targets)

                # Log
                if self.is_log:
                    self.log_images(targets, model_name, predictions, model_name_print, filtering_name=filtering_config)

                # update for plotting TODO REFAcTO or Remove
                #self.update_data_saving(targets, predictions)

    def plot_histogram(self):

        fig, axs = plt.subplots(2)

        # Print anchors positions :
        nb_ratio = len(self.aspect_ratios[0])
        nb_size = len(self.anchor_sizes)

        anchors_x = np.concatenate(self.anchors_coordinates[0])
        anchors_y = np.concatenate(self.anchors_coordinates[1])
        anchors_width = np.concatenate(self.anchors_positions[0])
        anchors_height = np.concatenate(self.anchors_positions[1])

        # Draw anchors positions
        #axs[1].scatter(anchors_x, anchors_y)

        index = 0
        for i in range(nb_size - 1, -1, -1):
            limit = pow(4, i) * 252 * nb_ratio
            axs[1].scatter(anchors_x[index:index + limit], anchors_y[index:index+limit], s=(40 / (i + 1)))
            index += limit

        # Draw some anchors rectangle
        # for each size print each ratio on the first position pos
        nb_pos_to_print = 150 // nb_ratio #max(30, nb_size * nb_ratio)
        idx_to_print = (np.random.randint(0, len(anchors_y) / nb_ratio, nb_pos_to_print - 2) - 1) * nb_ratio
        idx_to_print[-1] = len(anchors_y) - nb_ratio
        idx_to_print[0] = 0
        idx_to_print = np.round(idx_to_print).astype(int)

        for start_idx in idx_to_print:
            for (x, y, w, h) in zip(anchors_x[start_idx:start_idx + nb_ratio], anchors_y[start_idx: start_idx + nb_ratio], anchors_width[start_idx:start_idx + nb_ratio], anchors_height[start_idx: start_idx + nb_ratio]):
                rectangle = patches.Rectangle((x - w/2, y - h/2), w, h, edgecolor='g', facecolor='none', linewidth=2)
                axs[1].add_patch(rectangle)


        # Draw missing targets 
        nb_miss_targets_to_draw = 400
        targets_neg_x = np.concatenate(self.targets_coordinates_neg[0])
        #print("nb missing targets :", len(targets_neg_x))
        targets_neg_y = np.concatenate(self.targets_coordinates_neg[1])
        targets_neg_width = np.concatenate(self.targets_positions_neg[0])
        targets_neg_height = np.concatenate(self.targets_positions_neg[1])
        for (x, y, w, h) in zip(targets_neg_x[:nb_miss_targets_to_draw], targets_neg_y[:nb_miss_targets_to_draw], targets_neg_width[:nb_miss_targets_to_draw], targets_neg_height[:nb_miss_targets_to_draw]):
            rectangle = patches.Rectangle((x - w/2, y - h/2), w, h, edgecolor='r', facecolor='none', linewidth=5)
            axs[1].add_patch(rectangle)



        # Print absolu position :


        #print(np.unique(anchors_height))
        #print(np.unique(anchors_width))
        combined_coordinates = np.vstack((anchors_width, anchors_height)).T
        anchors = np.unique(combined_coordinates, axis=0)
        #print(anchors)

        """
        strides = [(3,4), (7,8), (15,16), (30, 32), (61, 64)]
        colors = ["hotpink", "deeppink", "mediumvioletred", "darkviolet", "indigo"] * 2000
        #print(f"{colors = }")
        #print('len unique :', len(anchors))
        is_strides = False
        for i, anchor_coodinate in enumerate(anchors):
            if is_strides:
                stride_x, stride_y = strides[i//nb_ratio]
            else:
                stride_x, stride_y = (0, 0)
            #print(stride_x, stride_y)
            x = anchor_coodinate[0]# - stride_x
            y = anchor_coodinate[1]# - stride_y
            x_max = (x - stride_x) *2
            x_min = (x + stride_x) /2
            y_max = (y - stride_y) *2
            y_min = (y + stride_y) /2

            # Draw 

            # Right part of the anchors space ( w <= x <= 2w) :
            x_values = np.linspace(x, x_max, 100)

            y_value_1 = (2 * x * y) / x_values  # y = 2wh/x
            y_value_1 = np.clip(y_value_1, y, y_max) #  h <= y <= 2h
            axs[0].plot(x_values, y_value_1, color='red')

            y_value_2 = - (x * y) / (x_values - (3 * x)) # y = -wh/x-3w
            y_value_2 = np.clip(y_value_2 , y/2, y) # h/2 <= y <= h
            axs[0].plot(x_values, y_value_2, color='red')

            axs[0].fill_between(x_values, y_value_1, y_value_2, alpha=0.3, color=colors[i//nb_ratio])

            # Left part of the anchors space ( w/2 <= x <= w) :
            x_values = np.linspace(x_min, x, 100) 

            y_value_1 = (x * y) / (2 * x_values) # y = wh/2x
            y_value_1 = np.clip(y_value_1 , y_min, y) # h/2 <= y <= h
            axs[0].plot(x_values, y_value_1 , color='red')

            y_value_2 = (3 * y) - ((x * y) / x_values) # y = 3h - xy/x
            y_value_2 = np.clip(y_value_2, y, y_max) #  h <= y <= 2h
            axs[0].plot(x_values, y_value_2, color='red')

            axs[0].fill_between(x_values, y_value_1, y_value_2, alpha=0.3, color=colors[i//nb_ratio])

            #exit()

        """

        # Draw area limit :

        self.minimum_area = 23 * 22
        self.maximum_area = 512 * 1024

        x_values = np.linspace(3, 100, 100)
        y_values = self.minimum_area / x_values
        axs[0].plot(x_values, y_values, color='black')

        x_values = np.linspace(500, 1400, 200)
        y_values = self.maximum_area / x_values
        axs[0].plot(x_values, y_values, color='black')

        x_values = np.linspace(0, 1400, 200)
        y_values = (self.aspect_ratios[0][0] / 2) * x_values
        axs[0].plot(x_values, y_values, color='black')

        x_values = np.linspace(0, 200, 100)
        y_values = (self.aspect_ratios[0][2] * 2) * x_values
        axs[0].plot(x_values, y_values, color='black')



        axs[0].scatter(np.concatenate(self.targets_positions_pos[0]), np.concatenate(self.targets_positions_pos[1]), c="green", s=20)
        targets_width = np.concatenate(self.targets_positions_neg[0])
        targets_height = np.concatenate(self.targets_positions_neg[1])
    
        #filter_size = 15
        #mask = (targets_width > filter_size) & (targets_height > filter_size)
        #mask = (targets_width * targets_height) > (filter_size * filter_size)
        #axs[0].scatter(targets_width[mask], targets_height[mask], c="brown", s=40, marker="x")
        axs[0].scatter(targets_width, targets_height, c="brown", s=20, marker="x")
        #axs[0].scatter(anchors_width, anchors_height, c="red", s=50)



        # Add images size rectangle
        rectangle = patches.Rectangle((0, 0), 1333, 749, edgecolor='black', facecolor='none', linewidth=3)
        axs[0].add_patch(rectangle)
        rectangle = patches.Rectangle((0, 0), 1333, 749, edgecolor='black', facecolor='none', linewidth=3)
        axs[1].add_patch(rectangle)

        axs[0].axis('equal')
        axs[1].axis('equal')
        axs[1].invert_yaxis()
        plt.show()

        return

        # Plot are, width and height

        fig, axs = plt.subplots(2, 4)

        areas = self.area_targets_counts.cpu().numpy()
        areas_bins = self.area_targets_bins.cpu().numpy()
        axs[0][0].stairs(areas, edges=areas_bins, fill=True)

        width = self.width_targets_counts.cpu().numpy()
        width_bins = self.width_targets_bins.cpu().numpy()
        axs[0][1].stairs(width, edges=width_bins, fill=True)

        height = self.height_targets_counts.cpu().numpy()
        height_bins = self.height_targets_bins.cpu().numpy()
        axs[0][2].stairs(height, edges=height_bins, fill=True)

        targets_width = np.concatenate(self.targets_positions_neg[0])
        targets_height = np.concatenate(self.targets_positions_neg[1])

        axs[0][3].scatter(np.concatenate(self.targets_positions_pos[0]), np.concatenate(self.targets_positions_pos[1]), c="green", s=10)
        axs[0][3].scatter(targets_width, targets_height, s=10)
        axs[0][3].scatter(np.concatenate(self.anchors_positions[0]), np.concatenate(self.anchors_positions[1]), c="red")

        areas = self.area_preds_counts.cpu().numpy()
        areas_bins = self.area_preds_bins.cpu().numpy()
        axs[1][0].stairs(areas, edges=areas_bins, fill=True)

        width = self.width_preds_counts.cpu().numpy()
        width_bins = self.width_preds_bins.cpu().numpy()
        axs[1][1].stairs(width, edges=width_bins, fill=True)

        height = self.height_preds_counts.cpu().numpy()
        height_bins = self.height_preds_bins.cpu().numpy()
        axs[1][2].stairs(height, edges=height_bins, fill=True)


        axs[0,0].set_title('Area of missing targets')
        axs[0,1].set_title('width of missing targets')
        axs[0,2].set_title('height of missing targets')
        axs[1,0].set_title('Area of anchors')
        axs[1,1].set_title('width of anchors')
        axs[1,2].set_title('height of anchors')

        plt.show()

    def test_epoch_print_metric(self, metrics_module, model_name, with_print):

        metrics_results = metrics_module.get_wandb_metrics(with_print=with_print)
        wandb.log({"Test/metrics": metrics_results})
        test_raw_coverage = metrics_module.get_raw_pourcent_coverage()
        self.log("test_coverage", test_raw_coverage)

        if metrics_module.cfg.mAP:
            self.log("test_map_known", metrics_module.current_known_map['map'])
            self.log("test_map_unknown", metrics_module.current_unknown_map['map'])

        self.log("test_precision_known", metrics_module.get_known_precision())
        self.log("test_recall_known", metrics_module.get_known_recall())
        self.log("test_precision_unknown", metrics_module.get_unknown_precision())
        self.log("test_recall_unknown", metrics_module.get_unknown_recall())
        self.log("test_A-OSE", metrics_module.get_open_set_errors())

        return metrics_results

    def on_test_epoch_end(self):


        print()
        print("testing metrics :")
        print()
        metrics_latex = {"name" : []}
        metrics_to_load = ["Raw coverage", "Raw coverage known", "Raw coverage unknown"]
        for i, print_model_name in enumerate(self.print_model_names):
            for metric_name in metrics_to_load:
                metrics_latex[print_model_name + "_" + metric_name] = []


        max_coverage_result = 0
        max_coverage_result_name = ''

        for i, (model_name, print_model_name) in enumerate(zip(self.models_name, self.print_model_names)):

            print(f"{model_name=}")
            for filtering_config in self.filtering_config:
                print(f"{filtering_config = }")
                metrics_module = self.metrics_modules[model_name][filtering_config]
                metric_results = self.test_epoch_print_metric(metrics_module, model_name, with_print=True)

                if metric_results["Raw coverage known"] > max_coverage_result:
                    max_coverage_result_name = filtering_config
                    max_coverage_result = metric_results["Raw coverage known"]

                if i == 0:
                    metrics_latex["name"].append(filtering_config)

                for key in metrics_to_load:
                    metrics_latex[print_model_name + "_" + key].append(metric_results[key])

        self.log("max_coverage_result", max_coverage_result)
        self.log("max_coverage_result_" + max_coverage_result_name, max_coverage_result)

        print(f"{metrics_latex = }")

        metrics_df = pd.DataFrame(metrics_latex).T
        latex_table = metrics_df.to_latex(float_format="%.2f").replace('\n', ' ')

        print(f"{latex_table = }")

        # Create a plot
        fig, ax = plt.subplots()

        plt.rc('text', usetex=True)
        plt.rc('text.latex', preamble=r'\usepackage{booktabs} \usepackage{etoolbox} \usepackage{multirow} \usepackage[table]{xcolor} \usepackage{colortbl} \usepackage{siunitx} \usepackage{longtable} \usepackage{graphics}')

        ax.text(0.5,0.5,latex_table, transform=fig.transFigure, horizontalalignment='center', verticalalignment='center', fontsize=20)

        ax.axis('off')
        plt.show()

        return

        self.plot_histogram()


        metrics = {}
        metrics_table = ["Raw Recall", "Raw coverage", "Raw coverage known", "Raw coverage unknown", "Number of raw prediction boxes per imgs"]

        metrics[print_model_name] = {}
        for metric_name in metrics_table:
            metrics[print_model_name][metric_name] = metrics_results[metric_name]
        #metrics[print_model_name]["known mAPs"] = metrics[print_model_name]["known mAPs"]["map"].item()
        #metrics[print_model_name]["unknown mAPs"] = metrics[print_model_name]["unknown mAPs"]["map"].item()

        metrics_df = pd.DataFrame(metrics).T
        latex_table = metrics_df.to_latex(float_format="%.2f").replace('\n', ' ')

        # Display the LaTeX table
        print(latex_table)
        print(type(latex_table))

        # Create a plot
        fig, ax = plt.subplots()

        plt.rc('text', usetex=True)
        plt.rc('text.latex', preamble=r'\usepackage{booktabs} \usepackage{etoolbox} \usepackage{multirow} \usepackage[table]{xcolor} \usepackage{colortbl} \usepackage{siunitx} \usepackage{longtable} \usepackage{graphics}')

        ax.text(0.5,0.5,latex_table, transform=fig.transFigure, horizontalalignment='center', verticalalignment='center', fontsize=20)

        ax.axis('off')
        plt.show()

   
def my_anchor_generator_forward(self, image_size, feature_maps: list[Tensor], anchors_dist_is_evently=False, clamp=False) -> list[Tensor]:

    batch_size = len(feature_maps)
    feature_maps = feature_maps[0]

    if anchors_dist_is_evently:
        grid_sizes = [(10, 10)] * 10 
    else:
        grid_sizes = [feature_map.shape[-2:] for feature_map in feature_maps]


    #grid_sizes = [(368, 672), (184, 336), (92, 168), (46, 84), (23, 42), (12, 21)]
    #grid_sizes = [(192, 336), (96, 168), (48, 84), (24, 42), (12, 21)]
    #image_size = image_size_list.tensors.shape[-2:]

    dtype, device = feature_maps[0].dtype, feature_maps[0].device
    strides = [
     [
         torch.empty((), dtype=torch.int64, device=device).fill_(image_size[0] // g[0]),
         torch.empty((), dtype=torch.int64, device=device).fill_(image_size[1] // g[1]),
     ]
     for g in grid_sizes
    ]
    """
    print(f"{grid_sizes = }")
    print(image_size)
    print(f"{strides = }")
    exit()
    """
    self.set_cell_anchors(dtype, device)
    anchors_over_all_feature_maps = self.grid_anchors(grid_sizes, strides)

    if clamp:
        for i, _ in enumerate(anchors_over_all_feature_maps):
            anchors_over_all_feature_maps[i][:, 0] = torch.clamp(anchors_over_all_feature_maps[i][:, 0], min=0, max=image_size[1])
            anchors_over_all_feature_maps[i][:, 1] = torch.clamp(anchors_over_all_feature_maps[i][:, 1], min=0, max=image_size[0])
            anchors_over_all_feature_maps[i][:, 2] = torch.clamp(anchors_over_all_feature_maps[i][:, 2], min=0, max=image_size[1])
            anchors_over_all_feature_maps[i][:, 3] = torch.clamp(anchors_over_all_feature_maps[i][:, 3], min=0, max=image_size[0])

    anchors: List[List[torch.Tensor]] = []
    for _ in range(batch_size):
     anchors_in_image = [anchors_per_feature_map for anchors_per_feature_map in anchors_over_all_feature_maps]
     anchors.append(anchors_in_image)
    anchors = [torch.cat(anchors_per_image) for anchors_per_image in anchors]
    return anchors


    # For every combination of (a, (g, s), i) in (self.cell_anchors, zip(grid_sizes, strides), 0:2),
    # output g[i] anchors that are s[i] distance apart in direction i, with the same dimensions as a.
def my_grid_anchors(self, grid_sizes: list[list[int]], strides: list[list[Tensor]]) -> list[Tensor]:
    anchors = []
    cell_anchors = self.cell_anchors
    torch._assert(cell_anchors is not None, "cell_anchors should not be None")
    torch._assert(
        len(grid_sizes) == len(strides) == len(cell_anchors),
        "Anchors should be Tuple[Tuple[int]] because each feature "
        "map could potentially have different sizes and aspect ratios. "
        "There needs to be a match between the number of "
        "feature maps passed and the number of sizes / aspect ratios specified.",
    )

    for size, stride, base_anchors in zip(grid_sizes, strides, cell_anchors):
        grid_height, grid_width = size
        stride_height, stride_width = stride
        device = base_anchors.device

        # For output anchor, compute [x_center, y_center, x_center, y_center]
        shifts_x = torch.arange(0, grid_width, dtype=torch.int32, device=device) * stride_width + (stride_width//2)
        shifts_y = torch.arange(0, grid_height, dtype=torch.int32, device=device) * stride_height + (stride_height//2)
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)

        # For every (base anchor, output anchor) pair,
        # offset each zero-centered base anchor by the center of the output anchor.
        anchors.append((shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)).reshape(-1, 4))

    return anchors
