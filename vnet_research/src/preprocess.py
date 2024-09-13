import nibabel as nib
import glob
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import sys

SAVE_ROOT_PATH = "D:/BratsDataset"
ROOT_PATH = "D:/Transfer/MICCAI_BraTS2020_TrainingData"  # This is the issue,

## TODO: Validation does not work at the moment
# VAL_ROOT_PATH = "D:/Transfer/MICCAI_BraTS2020_TrainingData"  

def preprocess(root_path, save_root_path, split="train"):
    
    if split not in ("train", "val", "test"):
        raise ValueError("split must be set to train, val, or test")

    train_path = os.path.join(save_root_path, "train")
    val_path = os.path.join(save_root_path, "validation")

    train_img_path = os.path.join(train_path, "images")
    train_label_path = os.path.join(train_path, "label")

    val_img_path = os.path.join(val_path, "images")
    val_label_path = os.path.join(val_path, "label")

    try:
        os.makedirs(train_path)
        print("Creating train path")
    except:
        print("train path already exists")
    try:
        os.makedirs(val_path)
        print("Creating validation path")
    except:
        print("validation path already exists")
    try:
        os.makedirs(train_img_path)
        print("Creating train image path")
    except:
        print("train image already exists")
    try:
        os.makedirs(train_label_path)
        print("Creating train label path")
    except:
        print("train label already exists")
    try:
        os.makedirs(val_img_path)
        print("Creating validation image path")
    except:
        print("validation image path already exists")
    try:
        os.makedirs(val_label_path)
        print("Creating validation label path")
    except:
        print("validaiton label path already exists")
    
    data_paths = glob.glob(f"BraTS20_Train*", root_dir=root_path)
    dataset_len = len(data_paths)
    
    save_datapaths = data_paths[0 : int(dataset_len * 0.8)]
    save_img_path = train_img_path
    save_label_path = train_label_path

    if split.lower() == "val":
        # Use 20% of the training data
        save_datapaths = data_paths[int(dataset_len * 0.8):-1]
        save_img_path = val_img_path
        save_label_path = val_label_path
    
    counter = 0
    for _, data_path in enumerate(save_datapaths):
        imgs = []
        data_dir = os.path.join(root_path, data_path)
        
        label_paths = glob.glob("*seg.nii*", root_dir=data_dir) 
        feature_paths = glob.glob("*[!g].nii*", root_dir=data_dir)

        # Saving path
        img_file_name = os.path.join(save_img_path, f"img_{counter}")
        label_file_name = os.path.join(save_label_path, f"label_{counter}")

        label_path = os.path.join(data_dir, *label_paths)
        for feature_path in feature_paths:
            img_path = os.path.join(data_dir, feature_path)
            mri_img = nib.load(img_path).get_fdata()
            mri_img = mri_img.astype(np.float32)

            # This min_max_scaler is causing the images to become all zero, 
            # Fixed: Turns out, min_max_scaler takes in a column vector not a row vector in order for it to work
            
            mri_img = MinMaxScaler().fit_transform(mri_img.reshape(-1, 1)).reshape(*mri_img.shape)
            mri_img = mri_img[56:184, 56:184, 5:133]
            imgs.append(mri_img)

        imgs = np.stack(imgs, axis=0)
        
        label = nib.load(label_path).get_fdata()
        label = label.astype(np.float32)
        label = label[56 : 184, 56: 184, 5: 133]
        _, num_of_classes = np.unique(label, return_counts=True)

        if 1 - (num_of_classes / num_of_classes.sum())[0] > 10e-3:
            
            # print("Maximum minimum coordinates for selecting images :", np.where(zeros > 500)[0].min(), np.where(zeros > 500)[0].max())
            print(f"probability distribution of labels that are not background {counter}: {1 - (num_of_classes / num_of_classes.sum())[0]:.5f}")
            np.save(img_file_name, imgs)
            np.save(label_file_name, label)
            counter += 1

# preprocess(ROOT_PATH, SAVE_ROOT_PATH, split="train")
preprocess(ROOT_PATH, SAVE_ROOT_PATH, split="val")


# TO DO: Reshape the image to 128 x 128 x 128, same for the segmentation
# Implement normalization on all the images before loading in data
# Create patches of images 