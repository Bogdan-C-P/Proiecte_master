import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import torch
import torchvision
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import json
import os

import cv2
import matplotlib.pyplot as plt


device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
with open('picked_image_paths.json', 'r') as file:
    training_image_paths = json.load(file)

with open('picked_anno_paths.json', 'r') as file:
    training_annotation_paths = json.load(file)

from torchvision.models.detection import maskrcnn_resnet50_fpn
def get_model_instance_segmentation(num_classes):
    # Load a MobileNetV2-based backbone pre-trained on COCO
    backbone = torchvision.models.mobilenet_v2(pretrained=True).features
    backbone.out_channels = 1280  # Update the output channels

    # Create a Mask R-CNN model with the MobileNetV2 backbone
    model = maskrcnn_resnet50_fpn(pretrained_backbone=backbone)
    for param in model.backbone.parameters():
        param.requires_grad = False
    # Replace the classifier head with a new one
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Replace the mask predictor with a new one
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    return model
model = get_model_instance_segmentation(13)
model.load_state_dict(torch.load("mobilenet_mask_rcnn_3.pth"))
model.eval()
model = model.to(device)
with open('colors.json', 'r') as file:
    color_map = json.load(file)

import pandas as pd

ade20k_labels_info = pd.read_csv('object150_info.csv')
labels_list = list(ade20k_labels_info['Name'])
labels_list.insert(0, 'others')
print(ade20k_labels_info.head())
print(len(labels_list))


color_dict = {}
for i in range(len(labels_list)):
    color_dict[i] = tuple(color_map[i])
print(color_dict)

### testare pe 5 poze din baza de date
#data_loader_validation = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=True, collate_fn=user_scattered_collate, num_workers=4, drop_last=True, pin_memory=True)

def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    Calculam aria suprapusa (IOU)

    Args:
        boxes_preds (tensor): bonding boxurile prezise de dim: (batch size, 4)
        boxes_labels (tensor):  bonding boxurile corecte de dim (batch size, 4)
        box_format (str): midpoint/corners, daca chenarele sunt de forma (x,y,w,h) sau (x1,y1,x2,y2)

    Returns:
        tensor: aria suprapunerii a elementelor din batch
    """

    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2

        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2

        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2

        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    if box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]

        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]

        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]

        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))


    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    return intersection / (box1_area + box2_area - intersection + 1e-6)


def non_max_suppression_rcnn(bboxes, iou_threshold, threshold, box_format="corners"):
    """
    Aplicam supresia non-maximelor pe chenarele oferite

    Args:
        bboxes (list): lista cu toate chenarele de forma [class_pred, prob_score, x1, y1, x2, y2]
        iou_threshold (float): pragul pentru care aria comuna inseamna ca avem acelas element
        threshold (float): pragul pentru care luam in considerare un chenar
        box_format (str): midpoint/corners in functie de formatul chenarelor

    Returns:
        list: chenarele dupa aplicarea algoritmului
    """

    if type(bboxes) != list:
      return []

    bboxes = [box for box in bboxes if box[0][1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[0][1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [
            box for box in bboxes if box[0][0] != chosen_box[0][0] or
            intersection_over_union(torch.tensor(chosen_box[0][2:]), torch.tensor(box[0][2:]),
                                    box_format=box_format) < iou_threshold
            ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms


def img_transform(img):
  normalize = transforms.Normalize(
          mean=[0.485, 0.456, 0.406],
          std=[0.229, 0.224, 0.225])
  # 0-255 to 0-1
  img = np.float32(np.array(img)) / 255.
  img = img.transpose((2, 0, 1))
  img = normalize(torch.from_numpy(img.copy()))
  return img





test_images_paths = training_image_paths
test_images_anno = training_annotation_paths

for idx in range(10):
  idx = 1

  try:

    image_rgb = cv2.cvtColor(cv2.imread(test_images_paths[idx]), cv2.COLOR_BGR2RGB)

    anno_root_path = test_images_anno[idx]
    print(anno_root_path)
    annontation_gray = cv2.imread(anno_root_path, 0)
    print('actual labels', set([labels_list[i] for i in np.unique(annontation_gray)]))

    height, width = annontation_gray.shape
    annontation_rgb = np.zeros((height, width, 3), dtype=np.uint8)

    for unique_value, color_triplet in color_dict.items():
        pixels = annontation_gray == unique_value
        annontation_rgb[pixels] = color_triplet

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].imshow(image_rgb)
    try:
      axs[1].imshow(annontation_rgb)
    except:
      pass


    image_tensor = img_transform(image_rgb)
    image_tensor = image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        model.eval()
        predictions = model(image_tensor)
    boxes = list(predictions[0]['boxes'].cpu())
    masks = list(predictions[0]['masks'].cpu())
    labels = list(predictions[0]['labels'].cpu())
    scores = list(predictions[0]['scores'].cpu())
    boxes_with_labels = []
    for i in range(len(boxes)):
        tmp_list = [labels[i].item(), scores[i].item()] + list(boxes[i])
        boxes_with_labels.append([tmp_list, (masks[i]>0.2).float()])
    #print(nms_boxes)
    new_labels = []
    new_masks = []
    new_scores = []
    nms_boxes = non_max_suppression_rcnn(bboxes=boxes_with_labels, iou_threshold=0.5, threshold=0.05)
    for i in range(len(nms_boxes)):
      new_labels.append(nms_boxes[i][0][0])
      new_masks.append(nms_boxes[i][1])
      new_scores.append(nms_boxes[i][0][1])


    print('predicted labels',set([labels_list[i] for i in new_labels]))

    gray_image = np.zeros((height, width), dtype=np.uint8)
    rgb_image = np.zeros((height, width,3), dtype=np.uint8)
    for label, mask in zip(new_labels, new_masks):
        gray_image[mask[0] != 0] = label
    for unique_value, color_triplet in color_dict.items():
        pixels = gray_image == unique_value
        rgb_image[pixels] = color_triplet
    axs[2].imshow(rgb_image)
    plt.show()
    #break
  except Exception as e:
    print(e)
  break