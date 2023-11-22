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

def main():
    # Define the paths to your image and annotation directories
    training_image_dir = 'ADEChallengeData2016/images/training'
    training_annotation_dir = 'ADEChallengeData2016/annotations/training'

    validation_image_dir = 'ADEChallengeData2016/images/validation'
    validation_annotation_dir = 'ADEChallengeData2016/annotations/validation'
    '''
    # Get the list of image and annotation file names
    training_image_files = os.listdir(training_image_dir)
    validation_image_files = os.listdir(validation_image_dir)
    training_annotation_files = os.listdir(training_annotation_dir)
    validation_annotation_files = os.listdir(validation_annotation_dir)

    # Lists for storing valid image and annotation paths
    training_image_paths = []
    training_annotation_paths = []
    validation_image_paths = []
    validation_annotation_paths = []

    # Verify and create correct lists
    def verify_and_create_lists(image_files, annotation_files, image_dir, annotation_dir, valid_image_paths, valid_annotation_paths):
        for img_file in image_files:
            annotation_ext = '.png'  # Adjust to match the annotation file extension

            corresponding_annotation = img_file.replace('.jpg', annotation_ext)

            if corresponding_annotation in annotation_files:
                image_path = os.path.join(image_dir, img_file)
                annotation_path = os.path.join(annotation_dir, corresponding_annotation)

                valid_image_paths.append(image_path)
                valid_annotation_paths.append(annotation_path)


    verify_and_create_lists(training_image_files, training_annotation_files, training_image_dir, training_annotation_dir, training_image_paths, training_annotation_paths)

    verify_and_create_lists(validation_image_files, validation_annotation_files, validation_image_dir, validation_annotation_dir, validation_image_paths, validation_annotation_paths)

    
    # Save lists to JSON files
    with open('training_image_paths.json', 'w') as file:
        json.dump(training_image_paths, file)

    with open('training_annotation_paths.json', 'w') as file:
        json.dump(training_annotation_paths, file)

    with open('validation_image_paths.json', 'w') as file:
        json.dump(validation_image_paths, file)

    with open('validation_annotation_paths.json', 'w') as file:
        json.dump(validation_annotation_paths, file)
    '''

    with open('picked_image_paths.json', 'r') as file:
        training_image_paths = json.load(file)

    with open('picked_anno_paths.json', 'r') as file:
        training_annotation_paths = json.load(file)
    print(len(training_annotation_paths))
    with open('validation_image_paths.json', 'r') as file:
        validation_image_paths = json.load(file)

    with open('validation_annotation_paths.json', 'r') as file:
        validation_annotation_paths = json.load(file)

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
            mask = cv2.resize(cv2.imread(self.masks[idx], 0), (224, 224), interpolation=cv2.INTER_NEAREST)
            mask = self.segm_transform(mask)

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
            return [self.img_transform(img), target]
            # return [transforms.ToTensor()(img) , target]

        def __len__(self):
            return len(self.imgs)

    train_dataset = Custom_dataset(training_image_paths,#[:1024],
                                   training_annotation_paths) #[:1024])

    with open('colors.json', 'r') as file:
        color_map = json.load(file)

    import pandas as pd

    ade20k_labels_info = pd.read_csv('object150_info.csv')
    labels_list = list(ade20k_labels_info['Name'])
    labels_list.insert(0, 'others')
    color_dict = {}
    for i in range(len(labels_list)):
        color_dict[tuple(color_map[i])] = labels_list[i]
    print(color_dict)
    '''
    idx = 1
    tensor_img = train_dataset[idx][0]

    # Convert the PyTorch tensor to a NumPy array
    numpy_array_img = tensor_img.permute(1, 2, 0).numpy()

    # Plot the NumPy array using matplotlib
    plt.imshow(numpy_array_img)
    plt.show()

    tensor_mask = train_dataset[idx][1]['masks']
    numpy_array_mask = tensor_mask.numpy()

    # Plot the NumPy array using matplotlib
    for i in range(len(numpy_array_mask)):
        plt.imshow(numpy_array_mask[i, :, :], cmap='gray')
        plt.title(labels_list[train_dataset[idx][1]['labels'][i].item()])
        plt.show()
    '''
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

    batch_size = 8
    data_loader_train = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                                    collate_fn=user_scattered_collate, drop_last=True,
                                                    pin_memory=True)


    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    import torch.distributed as dist
    def is_dist_avail_and_initialized():
        """
        Verificam dacă calculul distribuit (GPU) este disponibil și inițializat.

        Returns:
            bool: Adevărat dacă calculul distribuit este disponibil și inițializat, altfel False.

        """
        if not dist.is_available():
            return False
        if not dist.is_initialized():
            return False
        return True

    def num_of_proccesses():
        """
        Obținem numărul total de procese din configurația distribuită.

        Returns:
            int: Numărul de procese.
        """
        if not is_dist_avail_and_initialized():
            return 1
        return dist.get_world_size()

    def reduce_dict(input_dict, average=True):
        """
        Reducem valorile din dicționar de la toate procesele astfel încât toate procesele
        să aibă rezultatele medii.

        Args:
            input_dict (dict): Dicționarul ale cărui valori vor fi reduse.
            average (bool): Dacă se va calcula media (True) sau suma (False) valorilor.

        Returns:
            dict: Un dicționar cu aceleași chei ca și input_dict, care conține valorile reduse.
        """
        world_size = num_of_proccesses()
        if world_size < 2:
            return input_dict
        with torch.inference_mode():
            names = []
            values = []
            # sort the keys so that they are consistent across processes
            for k in sorted(input_dict.keys()):
                names.append(k)
                values.append(input_dict[k])
            values = torch.stack(values, dim=0)
            dist.all_reduce(values)
            if average:
                values /= world_size
            reduced_dict = {k: v for k, v in zip(names, values)}
        return reduced_dict

    import math
    epochs = 5
    num_of_batches = math.ceil(len(train_dataset) / batch_size)
    model.train()
    losss_vector = []
    for epoch in range(epochs):
        interm_loss = []
        print(f' epoch {epoch + 1} processing')
        for idx, batch in enumerate(data_loader_train, 1):
            if batch is None:
                continue
            try:
                images, targets = batch

                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                with torch.enable_grad():

                    loss_dict = model(images, targets)
                    print('loss_dict', idx, '/', num_of_batches, loss_dict)

                    if not loss_dict:
                        continue
                    losses = sum(loss for loss in loss_dict.values())
                    # print('losses', losses)
                loss_dict_reduced = reduce_dict(loss_dict)
                # print('loss_dict_reduced', loss_dict_reduced)
                losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                # print('losses_reduced', losses_reduced)
                loss_value = losses_reduced.item()
                # print(loss_value)
                interm_loss.append(loss_value)
                if not math.isfinite(loss_value):
                    print(f"Loss is {loss_value}, stopping training")
                    print(loss_dict_reduced)
                optimizer.zero_grad()
                losses.backward()  # retain_graph=True
                optimizer.step()

                #if idx % 20 == 0:
                #    torch.save(model.state_dict(),
                #               "/content/drive/MyDrive/proiect mlav/fasterrcnn_batch_{}.pth".format(idx))
                del images, targets, loss_value, losses_reduced, loss_dict_reduced, loss_dict, losses
                torch.cuda.empty_cache()
                lr_scheduler.step()
            except Exception as e:
                print(e)

        losss_vector.append(sum(interm_loss) / (len(interm_loss)+0.000001))

        torch.save(model.state_dict(), "mobilenet_mask_rcnn_3.pth")

    torch.save(model.state_dict(), "mobilenet_mask_rcnn_3.pth")
    print(losss_vector)
if __name__ == '__main__':
    main()