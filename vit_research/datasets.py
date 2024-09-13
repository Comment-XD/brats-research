import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from vit_research.transforms import *

train_transformation = transforms.Compose(
    transforms.RandomApply(
        transforms.ColorJitter(0.8, 0.8, 0.8, 0.2),
        p=0.3
    ),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply(
        transforms.GaussianBlur((3,3)),
        p=0.3
    ))

class Train_Dataset(Dataset):
    def __init__(self, args, transform=None):
        """
        
        takes in the train data path and implements augmentation.
        We will follow BYOL augmentation while also incorporating multi-channel conversion
        

        Args:
            args (_type_): _description_
            transform (_type_, optional): _description_. Defaults to None.
        """
        self.transform = transform
        self.g_feature = []
        self.l_feature = []
        
       
    def __len__(self):
        pass
    
    def __getitem__(self, idx):
        """
        Should return global features and local features

        Args:
            idx (_type_): _description_
        """
        return 
        
    

class Val_Dataset(Dataset):
    def __init__(self, args, transform=None):
        self.transform = transform
    
    def __len__(self):
        pass
    
    def __getitem__(self, idx):
        pass
    

class Test_Dataset(Dataset):
    def __init__(self, args, transform=None):
        self.transform = transform
    
    def __len__(self):
        pass
    
    def __getitem__(self, idx):
        pass
    
