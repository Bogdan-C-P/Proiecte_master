from __future__ import print_function

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import random
import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import utils
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models.vgg import VGG
from torch.optim import lr_scheduler
from torch.autograd import Variable


with open('picked_image_paths.json', 'r') as file:
    training_image_paths = json.load(file)

with open('picked_anno_paths.json', 'r') as file:
    training_annotation_paths = json.load(file)

with open('validation_image_paths.json', 'r') as file:
    validation_image_paths = json.load(file)

with open('validation_annotation_paths.json', 'r') as file:
    validation_annotation_paths = json.load(file)

#print(len(training_annotation_paths))

means = np.array([103.939, 116.779, 123.68]) / 255.
from PIL import Image
import cv2
class Custom_dataset(torch.utils.data.Dataset):
    def __init__(self , images , masks, means= means, n_class=13):
        self.imgs = images
        self.masks = masks
        self.means = means
        self.n_class = n_class

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self , idx):
        img_path = self.imgs[idx]
        img = cv2.resize(cv2.imread(img_path),(224,224))
        mask_path = self.masks[idx]
        mask = cv2.resize(cv2.imread(mask_path, 0), (224,224),interpolation=cv2.INTER_NEAREST)

        #img = img[:, :, ::-1]  # switch to BGR
        img = np.transpose(img, (2, 0, 1)) / 255.
        #img[0] -= self.means[0]
        #img[1] -= self.means[1]
        #img[2] -= self.means[2]

        img = torch.from_numpy(img.copy()).float()
        label = torch.from_numpy(mask.copy()).long()

        h, w = label.size()
        target = torch.zeros(self.n_class, h, w)
        for c in range(self.n_class):
            target[c][label == c] = 1

        sample = {'X': img, 'Y': target, 'l': label}

        return sample

def user_scattered_collate(batch):
    images = [item['X'] for item in batch]
    targets = [item['Y'] for item in batch]
    labels = [item['l'] for item in batch]

    # Filter out images with no target boxes
    non_empty_idx = [idx for idx, target in enumerate(targets) if target.shape[0] > 0]

    if not non_empty_idx:
        # If no images have target boxes, return an empty batch
        return None

    images = [images[idx] for idx in non_empty_idx]
    targets = [targets[idx] for idx in non_empty_idx]
    labels = [labels[idx] for idx in non_empty_idx]

    images = torch.stack(images, dim=0)
    targets = torch.stack(targets, dim=0)
    labels = torch.stack(labels, dim=0)
    return {'X': torch.tensor(images), 'Y':torch.tensor(targets), 'l':torch.tensor(labels)}


class FCNs(nn.Module):

    def __init__(self, pretrained_net, n_class):
        super().__init__()
        self.n_class = n_class
        self.pretrained_net = pretrained_net
        self.relu    = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4     = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5     = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)

    def forward(self, x):
        output = self.pretrained_net(x)
        x5 = output['x5']  # size=(N, 512, x.H/32, x.W/32)
        x4 = output['x4']  # size=(N, 512, x.H/16, x.W/16)
        x3 = output['x3']  # size=(N, 256, x.H/8,  x.W/8)
        x2 = output['x2']  # size=(N, 128, x.H/4,  x.W/4)
        x1 = output['x1']  # size=(N, 64, x.H/2,  x.W/2)

        score = self.bn1(self.relu(self.deconv1(x5)))     # size=(N, 512, x.H/16, x.W/16)
        score = score + x4                                # element-wise add, size=(N, 512, x.H/16, x.W/16)
        score = self.bn2(self.relu(self.deconv2(score)))  # size=(N, 256, x.H/8, x.W/8)
        score = score + x3                                # element-wise add, size=(N, 256, x.H/8, x.W/8)
        score = self.bn3(self.relu(self.deconv3(score)))  # size=(N, 128, x.H/4, x.W/4)
        score = score + x2                                # element-wise add, size=(N, 128, x.H/4, x.W/4)
        score = self.bn4(self.relu(self.deconv4(score)))  # size=(N, 64, x.H/2, x.W/2)
        score = score + x1                                # element-wise add, size=(N, 64, x.H/2, x.W/2)
        score = self.bn5(self.relu(self.deconv5(score)))  # size=(N, 32, x.H, x.W)
        score = self.classifier(score)                    # size=(N, n_class, x.H/1, x.W/1)

        return score  # size=(N, n_class, x.H/1, x.W/1)

class VGGNet(VGG):
    def __init__(self, pretrained=True, model='vgg16', requires_grad=True, remove_fc=True, show_params=False):
        super().__init__(make_layers(cfg[model]))
        self.ranges = ranges[model]

        if pretrained:
            exec("self.load_state_dict(models.%s(pretrained=True).state_dict())" % model)

        if not requires_grad:
            for param in super().parameters():
                param.requires_grad = False

        if remove_fc:  # delete redundant fully-connected layer params, can save memory
            del self.classifier

        if show_params:
            for name, param in self.named_parameters():
                print(name, param.size())

    def forward(self, x):
        output = {}

        # get the output of each maxpooling layer (5 maxpool in VGG net)
        for idx in range(len(self.ranges)):
            for layer in range(self.ranges[idx][0], self.ranges[idx][1]):
                x = self.features[layer](x)
            output["x%d"%(idx+1)] = x

        return output

ranges = {
    'vgg11': ((0, 3), (3, 6),  (6, 11),  (11, 16), (16, 21)),
    'vgg13': ((0, 5), (5, 10), (10, 15), (15, 20), (20, 25)),
    'vgg16': ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31)),
    'vgg19': ((0, 5), (5, 10), (10, 19), (19, 28), (28, 37))
}

# cropped version from https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
cfg = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


n_class    = 13

batch_size = 8
epochs     = 10
lr         = 1e-3
momentum   = 0
w_decay    = 1e-5
step_size  = 2
gamma      = 0.5
configs    = "FCNs-BCEWithLogits_batch{}_epoch{}_RMSprop_scheduler-step{}-gamma{}_lr{}_momentum{}_w_decay{}".format(
    batch_size, epochs, step_size, gamma, lr, momentum, w_decay)
print("Configs:", configs)


train_data = Custom_dataset(training_image_paths[:2048], training_annotation_paths[:2048])
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=user_scattered_collate, drop_last=True, pin_memory=True)

vgg_model = VGGNet(requires_grad=True, remove_fc=True)
fcn_model = FCNs(pretrained_net=vgg_model, n_class=n_class)
fcn_model.load_state_dict(torch.load('model_fcn.pth'))
fcn_model.train()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
fcn_model= fcn_model.to(device)
criterion = nn.BCEWithLogitsLoss()
#optimizer = optim.RMSprop(fcn_model.parameters(), lr=lr, momentum=momentum, weight_decay=w_decay)
optimizer = optim.Adam(fcn_model.parameters(), lr=lr, weight_decay=w_decay)

scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

IU_scores    = np.zeros((epochs, n_class))
pixel_scores = np.zeros(epochs)

import time





for epoch in range(epochs):
    scheduler.step()

    ts = time.time()
    for iter, batch in enumerate(train_loader):
        if batch is None:
            continue
        optimizer.zero_grad()

        inputs, labels = Variable(batch['X']), Variable(batch['Y'])

        inputs, labels = inputs.to(device), labels.to(device)
        outputs = fcn_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if iter % 10 == 0:
            print("epoch{}, iter{}, loss: {}".format(epoch+1, iter, loss.item()))
            torch.save(fcn_model.state_dict(), 'model_fcn.pth')
    print("Finish epoch {}, time elapsed {}".format(epoch+1, time.time() - ts))
    torch.save(fcn_model.state_dict(), 'model_fcn.pth')