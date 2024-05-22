import os
import torch
import fnmatch
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection.transform import resize_boxes
import re

class PredictionsDataset(Dataset):

    def __init__(self, data_dir, model_names, max_size, preload_targets=True, preload_predictions=True, target_pattern="targets_"):

        self.data_dir = data_dir
        self.model_names = model_names
        self.max_size = max_size
        self.target_pattern = target_pattern

        self.imgs_name = self._find_imgs_name()
        self.preload_targets = preload_targets
        self.preload_predictions = preload_predictions


        
        # Process settings :  TODO make cfg
        self.limit_area = False 
        self.limit_ratio = False 
        self.minimum_ratio = 0.5 / 2  #0.25
        self.maximum_ratio = 2.0  * 2
        self.nb_target_remove_with_limit_area = 0
        self.nb_target_remove_with_limit_ratio = 0
        self.nb_target_boxes = 0
        self.minimum_area = 23 * 22
        self.maximum_area = 512 * 1024

        self.image_resize = (749, 1333)

        if preload_targets:
            self.targets = self._preload_targets()
        if preload_predictions:
            self.predictions = self._preload_predictions()


        """
        if "anchors_custom" in model_names:
            features_path = os.path.join(self.data_dir, f"features.pth")
            self.features = torch.load(features_path)
        """

    def process_targets(self, targets):

        for img_name, target in targets.items():
        
            mask = torch.ones_like(target["labels"], dtype=bool)
            self.nb_target_boxes += mask.sum()

            if self.limit_area:
               boxes = target["boxes"]
               width = boxes[:, 2] - boxes[:, 0]
               height = boxes[:, 3] - boxes[:, 1]
               mask_area = ((width * height) > self.minimum_area) & ((width * height) < self.maximum_area)
               self.nb_target_remove_with_limit_area += (~mask_area).sum()
               mask = mask & mask_area

            if self.limit_ratio:
                if not self.limit_area:
                    boxes = target["boxes"]
                    width = boxes[:, 2] - boxes[:, 0]
                    height = boxes[:, 3] - boxes[:, 1]
                mask_ratio = ((width / height) > self.minimum_ratio) & ((width / height) < self.maximum_ratio)
                self.nb_target_remove_with_limit_ratio += (~mask_ratio).sum()
                mask = mask & mask_ratio

            for key, value in target.items():
                if torch.is_tensor(value):
                    targets[img_name][key] = value[mask]
                else:
                    targets[img_name][key] = value

            self.image_size = target["img_size"][-2:]
            targets[img_name]["boxes"] = resize_boxes(target["boxes"], target["img_size"][-2:], self.image_resize)
            targets[img_name]["img_size"] = (3, self.image_resize[0], self.image_resize[1])


        return targets

    # Resize all
    def process_predictions(self, predictions):

        predictions["filtered_predictions"]["boxes"] = resize_boxes(predictions["filtered_predictions"]["boxes"], self.image_size, self.image_resize).to(int).to(torch.float32)
        #predictions["proposals"] = resize_boxes(predictions["proposals"], self.image_size, self.image_resize)
        predictions["proposals"] = predictions["proposals"].to(torch.float32)

        return predictions

    
    def _find_imgs_name(self):

        imgs_name = []

        print(f"Loading targets in directory :{self.data_dir}")
        for file in os.listdir(self.data_dir):
            if fnmatch.fnmatch(file, f'*{self.target_pattern}*'):
                img_name = file[len(self.target_pattern):]
                number_match = int(re.search(r'\d+', img_name).group())

                if number_match <= self.max_size * 2:
                    imgs_name.append(img_name)
                if len(imgs_name) >= self.max_size:
                    break

        return imgs_name 
    
    def _preload_targets(self):

        targets = {}

        for img_name in self.imgs_name:
            target_path = os.path.join(self.data_dir, self.target_pattern + img_name)
            targets[img_name] = torch.load(target_path)

        targets = self.process_targets(targets)

        return targets

    def _load_img_predictions(self, img_name):

        predictions = {}
        for model_name in self.model_names:
            if model_name == "random" or model_name == "anchors" or model_name == "anchors_custom":
                continue
            model_prediction_path = os.path.join(self.data_dir, f"{model_name}_{img_name}")
            model_predictions = torch.load(model_prediction_path)#, map_location=torch.device('cpu'))

            #process boxes
            if model_name != "proposals":
                model_predictions = self.process_predictions(model_predictions)

            predictions[model_name] = model_predictions

        return predictions
    
    def _preload_predictions(self):

        predictions = {}
        for i, img_name in enumerate(self.imgs_name):
            predictions[img_name] = self._load_img_predictions(img_name)

        return predictions
    
    def __len__(self):
        return len(self.imgs_name)
    
    def __getitem__(self, idx):


        img_name = self.imgs_name[idx]
        target = self.targets[img_name] if self.preload_targets else torch.load(os.path.join(self.data_dir, self.target_pattern, img_name))

        if self.preload_predictions:
            predictions = self.predictions[img_name]
            if "anchors" in self.model_names:
                model_prediction_path = os.path.join(self.data_dir, f"anchors_{img_name}")
                model_predictions = torch.load(model_prediction_path)
                predictions["anchors"] = model_predictions
            """
            if "anchors_custom" in self.model_names:
                predictions["anchors_custom"] = self.features
            """
        else:
            predictions = self._load_img_predictions(img_name)

        return target, predictions 

