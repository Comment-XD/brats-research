import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import os
import nibabel as nib
import glob
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def dataloader(args, split="train"):
    if split == "train":
        return DataLoader(TrainDataset(), args.batch_size, num_workers=args.num_workers, shuffle=True)
    if split == "val":
        return


### Take the image and apply normalization
### What type of normalization?, We can apply greyscaling, and remove images that are beyond a certain percentage of labels
### We can also use pytorch normalize

class TrainDataset(Dataset):
    def __init__(self):
        super(Dataset, self).__init__()
        self.root_path = "D:\Transfer\MICCAI_BraTS2020_TrainingData"  

        flair_imgs_path = glob.glob("**\*flair.nii*", root_dir=self.root_path, recursive=True)
        t1_imgs_path = glob.glob("**\*t1.nii*", root_dir=self.root_path, recursive=True)
        t1ce_imgs_path = glob.glob("**\*t1ce.nii*", root_dir=self.root_path, recursive=True)
        t2_imgs_path = glob.glob("**\*t2.nii*", root_dir=self.root_path, recursive=True)

        self.dataset_path = list(zip(flair_imgs_path, t1_imgs_path, t1ce_imgs_path, t2_imgs_path))
        self.label_path = glob.glob("**\*_seg*", root_dir=self.root_path, recursive=True)

    def __len__(self):
        return len(self.label_path)

    def __getitem__(self, idx):
        min_max_scalar = MinMaxScaler()
        seg_mask_path = self.label_path[idx]
        mri_imgs_path = self.dataset_path[idx]

        mri_imgs = []
        seg_mask = nib.load(os.path.join(self.root_path, seg_mask_path)).get_fdata()

        for img_path in mri_imgs_path:
            
            mri_img = nib.load(os.path.join(self.root_path, img_path)).get_fdata()
            mri_img = min_max_scalar.fit_transform(mri_img.reshape(1, -1)).reshape(240,240,155)
            mri_imgs.append(mri_img)

        return mri_imgs, seg_mask