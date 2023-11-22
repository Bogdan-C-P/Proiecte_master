import json

import numpy as np
from PIL import Image

with open('training_image_paths.json', 'r') as file:
    training_image_paths = json.load(file)
print(len(training_image_paths))
with open('training_annotation_paths.json', 'r') as file:
    training_annotation_paths = json.load(file)


picked_images_list = []
picked_labels_list = []

import os
def are_all_elements_present(arr, element_list):
    # Convert both lists to sets for efficient comparison
    arr = arr.tolist() if isinstance(arr, np.ndarray) else arr
    arr_set = set(arr)
    element_set = set(element_list)

    # Check if all elements in arr are present in element_list
    return arr_set.issubset(element_set)

def filter_non_overlapping_images():

    image_files = training_image_paths # Replace 'images' with your images folder
    for idx, img_path in enumerate(image_files):
        # Load image and corresponding mask
        mask_path = training_annotation_paths[idx]
        mask = Image.open(mask_path)

        classes = np.unique(mask)
        classes = classes[1:]
        # print(classes)
        if are_all_elements_present(classes, [1,2,3,4, 5, 6,7,8,9,10,11,12]):
            picked_images_list.append(img_path)
            picked_labels_list.append(mask_path)

filter_non_overlapping_images()

print(len(picked_images_list))
with open('picked_image_paths.json', 'w') as file:
    json.dump(picked_images_list, file)

with open('picked_anno_paths.json', 'w') as file:
    json.dump(picked_labels_list, file)