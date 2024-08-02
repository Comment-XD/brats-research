import torch
from torch.utils.data import Dataset
import os
import glob
import numpy as np


class TrainDataset(Dataset):
    def __init__(self, train_dir_path):
        super().__init__()

        # Root path is the train folder, but there is two seperate folders for images and labels
        self.train_img_root = os.path.join(train_dir_path, "images")
        self.train_label_root = os.path.join(train_dir_path, "label")
        
        self.train_img_paths = glob.glob("*.npy", root_dir=self.train_img_root)
        self.train_label_paths = glob.glob("*.npy", root_dir=self.train_label_root)
        
    def __len__(self):
        return len(self.train_label_paths)

    def __getitem__(self, idx):
        img_path = self.train_img_paths[idx]
        img_path = os.path.join(self.train_img_root, img_path)
        
        label_path = self.train_label_paths[idx]
        label_path = os.path.join(self.train_label_root, label_path)
        
        img = np.load(img_path)
        img = torch.from_numpy(img)
        label = torch.from_numpy(np.load(label_path))

        # return img, label     
        return img, label

class ValidationDataset(Dataset):
    def __init__(self, val_dir_path):
        super().__init__()

        # Root path is the train folder, but there is two seperate folders for images and labels
        self.val_img_root = os.path.join(val_dir_path, "images")
        self.val_label_root = os.path.join(val_dir_path, "label")
        
        self.val_img_paths = glob.glob("*.npy", root_dir=self.val_img_root)
        self.val_label_paths = glob.glob("*.npy", root_dir=self.val_label_root)
        
    def __len__(self):
        return len(self.val_label_paths)

    def __getitem__(self, idx):
        img_path = self.val_img_paths[idx]
        img_path = os.path.join(self.val_img_paths, img_path)
        
        label_path = self.train_label_paths[idx]
        label_path = os.path.join(self.train_label_root, label_path)
        
        img = np.load(img_path)
        img = torch.from_numpy(img)
        label = torch.from_numpy(np.load(label_path))

        return img, label
    
class TestDataset(Dataset):
    def __init__(self, val_dir_path):
        super().__init__()

        # Root path is the train folder, but there is two seperate folders for images and labels
        self.val_img_root = os.path.join(val_dir_path, "images")
        self.val_label_root = os.path.join(val_dir_path, "label")
        
        self.val_img_paths = glob.glob("*.npy", root_dir=self.val_img_root)
        self.val_label_paths = glob.glob("*.npy", root_dir=self.val_label_root)
        
    def __len__(self):
        return len(self.val_label_paths)

    def __getitem__(self, idx):
        img_path = self.val_img_paths[idx]
        img_path = os.path.join(self.val_img_paths, img_path)
        
        label_path = self.train_label_paths[idx]
        label_path = os.path.join(self.train_label_root, label_path)
        
        img = np.load(img_path)
        img = torch.from_numpy(img)

        # return img, label     
        return img