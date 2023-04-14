import imgaug.augmenters as iaa
import os
import numpy as np
import cv2

# define augmentation pipeline
seq = iaa.Sequential([
    iaa.Affine(rotate=(-25, 25)),
    iaa.Flipud(0.5),
    iaa.Fliplr(0.5),
    iaa.GaussianBlur(sigma=(0, 1.0)),
    iaa.AddToHueAndSaturation((-20, 20))
])

# define path to directory containing original images
originals_dir = 'poze_grayscale\\Open_Eyes'

# define path to directory where augmented images will be saved
augmented_dir = 'poze_augumentate\\Open_Eyes'

# loop through original images and apply augmentations
for file in os.listdir(originals_dir):
    if file.endswith('.jpg') or file.endswith('.png'):
        image = cv2.imread(os.path.join(originals_dir, file))
        mode = image.shape[-1]
        if mode == 1: # grayscale image
            images = [image] * 4 # set number of augmentations per original image
            aug_images = seq(images=images)
            for i, aug_image in enumerate(aug_images):
                new_filename = f"{os.path.splitext(file)[0]}_{i}.jpg"
                aug_image = aug_image.squeeze() # remove singleton dimension
                cv2.imwrite(os.path.join(augmented_dir, new_filename), aug_image)
        elif mode == 3 or mode == 4: # color image
            images = [image] * 4 # set number of augmentations per original image
            aug_images = seq(images=images)
            for i, aug_image in enumerate(aug_images):
                new_filename = f"{os.path.splitext(file)[0]}_{i}.jpg"
                cv2.imwrite(os.path.join(augmented_dir, new_filename), aug_image)
