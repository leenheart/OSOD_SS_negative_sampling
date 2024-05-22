import json
import torch
from torchvision.io.image import read_image
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection import fcos_resnet50_fpn, FCOS_ResNet50_FPN_Weights
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image

img = read_image("../data/coco/val/images/000000000139.jpg")


# Step 1: Initialize model with the best available weights
weights = FCOS_ResNet50_FPN_Weights.COCO_V1
model = fcos_resnet50_fpn(weights=weights)#, box_score_thresh=0.9)
model.eval()
model = model.cuda()
"""
# Step 2: Initialize the inference transforms
preprocess = weights.transforms()

# Step 3: Apply inference preprocessing transforms
batch = [preprocess(img)]

"""


img_file_path = "../model_saves/prediction_saves/fcos/139.pt"
loaded_image_tensor = torch.load(img_file_path)
loaded_image_tensor[0] = loaded_image_tensor[0]#.to(torch.device("cpu"))
"""
print(loaded_image_tensor)
print(batch)
print(len(batch))
print(loaded_image_tensor[0] == batch[0])
print((loaded_image_tensor[0] == batch[0]).all())
print(type(loaded_image_tensor) == type(batch))
print((loaded_image_tensor == batch).all())

exit()
"""


# Step 4: Use the model and visualize the prediction
#prediction = model(batch)[0]
prediction = model(loaded_image_tensor)[0]

print(prediction)

"""
labels = [weights.meta["categories"][i] for i in prediction["labels"]]
#print("FCOS labels : ", labels)
box = draw_bounding_boxes(img, boxes=prediction["boxes"],
                          labels=labels,
                          colors="red",
                          width=4, font_size=30)
im = to_pil_image(box.detach())
im.show()
"""

# Step 5 Compare with my predictions ! 

# Path to the JSON file
json_file_path = "../model_saves/prediction_saves/fcos/139.json"

# Loading data from JSON file
with open(json_file_path, "r") as json_file:
    loaded_data = json.load(json_file)

print("len equal ? :", len(loaded_data["predictions"]["boxes"]) == len(prediction["boxes"].tolist()))
print("Boxes equal ? :", loaded_data["predictions"]["boxes"] == prediction["boxes"].tolist())
print("Boxes equal :", loaded_data["predictions"]["boxes"], prediction["boxes"].tolist())
print("Labels equal ? :", loaded_data["predictions"]["labels"] == prediction["labels"].tolist())
print("Labels equal ? :", loaded_data["predictions"]["labels"], prediction["labels"].tolist())
print("Scores equal ? :", loaded_data["predictions"]["scores"] == prediction["scores"].tolist())
print("Scores equal ? :", loaded_data["predictions"]["scores"], prediction["scores"].tolist())













"""
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision

preds = [
        dict(
            boxes=torch.tensor([[258.0, 41.0, 606.0, 285.0]]),
            scores=torch.tensor([0.536]),
            labels=torch.tensor([0]),
            )
        ]
target = [
        dict(
            boxes=torch.tensor([[214.0, 41.0, 562.0, 285.0]]),
            labels=torch.tensor([0]),
            )
        ]
metric = MeanAveragePrecision()
metric.update(preds, target)
from pprint import pprint
pprint(metric.compute())
"""
