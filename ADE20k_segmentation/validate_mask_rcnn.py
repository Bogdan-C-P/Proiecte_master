import cv2
import torch
import torchvision
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import json
import numpy as np
import cv2



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

with open('picked_image_paths.json', 'r') as file:
    training_image_paths = json.load(file)

with open('picked_anno_paths.json', 'r') as file:
    training_annotation_paths = json.load(file)

class Custom_dataset(torch.utils.data.Dataset):
    def __init__(self, images, masks):
        self.imgs = images
        self.masks = masks

    def img_transform(self, img):
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
        # 0-255 to 0-1
        img = np.float32(np.array(img)) / 255.
        img = img.transpose((2, 0, 1))
        img = normalize(torch.from_numpy(img.copy()))
        return img

    def segm_transform(self, segm):
        # to tensor, 0 to 150
        segm = torch.from_numpy(np.array(segm)).long()
        return segm

    def __getitem__(self, idx):
        # print(self.imgs[idx])

        #img = Image.open(self.imgs[idx]).convert("RGB")
        # mask = cv2.cv2.imread(self.masks[idx])
        #mask = Image.open(self.masks[idx])
        img = cv2.resize(cv2.imread(self.imgs[idx]), (224, 224))
        mask_orig = cv2.resize(cv2.imread(self.masks[idx], 0), (224, 224), interpolation=cv2.INTER_NEAREST)
        #print(mask_orig.shape)
        mask = self.segm_transform(mask_orig)

        #if not isinstance(img.size[0], int) or not isinstance(img.size[1], int):
            # Round the dimensions to integers
        #    img = img.resize((int(round(img.size[0])), int(round(img.size[1]))))

        # Check if the mask dimensions are integers
        #if not isinstance(mask.shape[0], int) or not isinstance(mask.shape[1], int):
            # Round the dimensions to integers
        #    mask = mask.resize((int(round(mask.size[0])), int(round(mask.size[1]))))

        classes = np.unique(mask)
        # print(classes)
        classes = classes[classes != 0]
        # print(classes)
        num_objs = len(classes)
        mask = np.array(mask)
        masks = np.zeros((num_objs, mask.shape[0], mask.shape[1]))
        # print(mask.shape)
        # print(masks.shape)
        for idx, i in enumerate(classes):
            bool_mask = mask == i
            # print(bool_mask)
            masks[idx][bool_mask] = True
        boxes = []
        labels = []
        for idx, i in enumerate(classes):
            bool_mask = mask == i
            pos = np.where(bool_mask)
            if len(pos[0]) > 0 and len(pos[1]) > 0:  # Check if there are valid elements
                xmin = int(np.min(pos[1]))
                xmax = int(np.max(pos[1]))
                ymin = int(np.min(pos[0]))
                ymax = int(np.max(pos[0]))
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(i)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target['final'] = torch.from_numpy(mask_orig)
        #print(target['final'].size())
        return [self.img_transform(img), target]
        # return [transforms.ToTensor()(img) , target]

    def __len__(self):
        return len(self.imgs)

train_dataset = Custom_dataset(training_image_paths,#[:1024],
                               training_annotation_paths) #[:1024])


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


device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
print(device)

num_classes = 13
model = get_model_instance_segmentation(num_classes)
model.load_state_dict(torch.load("mobilenet_mask_rcnn_3.pth"))
model = model.to(device)


def user_scattered_collate(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]

    # Filter out images with no target boxes
    non_empty_idx = [idx for idx, target in enumerate(targets) if target["boxes"].shape[0] > 0]

    if not non_empty_idx:
        # If no images have target boxes, return an empty batch
        return None

    images = [images[idx] for idx in non_empty_idx]
    targets = [targets[idx] for idx in non_empty_idx]

    return images, targets


batch_size = 1
data_loader_train = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                                collate_fn=user_scattered_collate, drop_last=True,
                                                pin_memory=True)


total_ious = []
pixel_accs = []

def iou(pred, target):
    ious = []
    n_class = 13
    for cls in range(n_class):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = pred_inds[target_inds].sum()
        union = pred_inds.sum() + target_inds.sum() - intersection
        if union == 0:
            ious.append(float('nan'))  # if there is no ground truth, do not include in evaluation
        else:
            ious.append(float(intersection) / max(union, 1))
        # print("cls", cls, pred_inds.sum(), target_inds.sum(), intersection, float(intersection) / max(union, 1))
    return ious


def pixel_acc(pred, target):
    correct = (pred == target).sum()
    total   = (target == target).sum()
    return correct / total
import matplotlib.pyplot as plt
from PIL import Image as PILImage

model.eval()
for iter, batch in enumerate(data_loader_train): #val_loader
    if batch is None:
        continue
    images, targets = batch

    images = list(image.to(device) for image in images)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    mask_final = []
    true_boxes = []
    for i in range(len(targets[0]['labels'])):
        list1 = [targets[0]['labels'][i].item()]
        list2 = [targets[0]['masks'][i]]
        concatenated_list = list1 + list2
        true_boxes.append(concatenated_list)
    mask_final = targets[0]['final']
    #print(mask_final.size())
    predictions = model(images)
    boxes = list(predictions[0]['boxes'])
    masks = list(predictions[0]['masks'])
    labels = list(predictions[0]['labels'])
    scores = list(predictions[0]['scores'])
    boxes_with_labels = []
    for i in range(len(boxes)):
        tmp_list = [labels[i].item(), scores[i].item()] + list(boxes[i])
        boxes_with_labels.append([tmp_list, (masks[i].cpu() > 0.2).float()])

    pred_boxes = non_max_suppression_rcnn(bboxes=boxes_with_labels, iou_threshold=0.5, threshold=0.05)
    new_labels = []
    new_masks = []
    new_scores = []
    for i in range(len(pred_boxes)):
      new_labels.append(pred_boxes[i][0][0])
      new_masks.append(pred_boxes[i][1])
      new_scores.append(pred_boxes[i][0][1])

    #print(new_masks[0].size())
    try:
        _, height, width = new_masks[0].size()
    except:
        continue
    gray_image = np.zeros((height, width), dtype=np.uint8)
    for label, mask in zip(new_labels, new_masks):
        gray_image[mask[0] != 0] = label
    #print(gray_image.shape)
    #print(gray_image)
    #gray_image_pil = PILImage.fromarray(gray_image)

    #plt.imshow(np.array(gray_image_pil), cmap='gray')
    #plt.imshow()


    mask_final = mask_final.cpu().numpy()
    #print(mask_final.shape)
    #plt.imshow(mask_final)
    #plt.show()
    total_ious.append(iou(gray_image, mask_final))

    pixel_accs.append(pixel_acc(gray_image, mask_final))

total_ious = np.array(total_ious).T  # n_class * val_len
ious = np.nanmean(total_ious, axis=1)
pixel_accs = np.array(pixel_accs).mean()
print("pix_acc: {}, meanIoU: {}, IoUs: {}".format(pixel_accs, np.nanmean(ious), ious))