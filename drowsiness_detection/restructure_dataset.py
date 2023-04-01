import os
import shutil
import random

dataset_dir = "dataset"
dataset_2_dir = "dataset_2"

if not os.path.exists(dataset_2_dir):
    os.makedirs(dataset_2_dir)

train_dir = os.path.join(dataset_2_dir, "train")
if not os.path.exists(train_dir):
    os.makedirs(train_dir)

val_dir = os.path.join(dataset_2_dir, "val")
if not os.path.exists(val_dir):
    os.makedirs(val_dir)

for class_name in os.listdir(dataset_dir):
    class_dir = os.path.join(dataset_dir, class_name)
    train_class_dir = os.path.join(train_dir, class_name)
    val_class_dir = os.path.join(val_dir, class_name)

    if not os.path.exists(train_class_dir):
        os.makedirs(train_class_dir)

    if not os.path.exists(val_class_dir):
        os.makedirs(val_class_dir)

    files = os.listdir(class_dir)
    random.shuffle(files)

    split_index = int(0.95 * len(files))

    train_files = files[:split_index]
    val_files = files[split_index:]

    for train_file in train_files:
        src_path = os.path.join(class_dir, train_file)
        dst_path = os.path.join(train_class_dir, train_file)
        shutil.copy(src_path, dst_path)

    for val_file in val_files:
        src_path = os.path.join(class_dir, val_file)
        dst_path = os.path.join(val_class_dir, val_file)
        shutil.copy(src_path, dst_path)
