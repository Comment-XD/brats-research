import argparse
import os
import torch.nn.grad

from src.datasets import *
from src.utils import *
from src.losses import *
from src.dataloader import dataloader
from models.unet import UNet

import numpy as np
from tqdm import tqdm


def parse_options():
    parser = argparse.ArgumentParser("settings")
    
    ### arguments for loading data
    
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    
    ### arguments for training hyperparameters
    
    parser.add_argument("--optimizer", type=str, default="sgd")
    parser.add_argument("--lr", type=float, default=0.04)
    parser.add_argument("--lr_decay", type=float, default="sgd")
    parser.add_argument("--epochs", type=int, default=10) 
    
    parser.add_argument('--model', type=int, default=50,
                        help='which model to use')
    
    opt = parser.parse_args()
    
    return opt


def main():
    # TODO: Create the file structure so everyone's structure is the same, resulting in consistent results
    # Create a saved models folder, when running models, only use models that had the best 
    # How to train already saved models?
    
    
    args = parse_options()
    train_dataloader = dataloader(args, split="train")
    model = UNet(in_channels=1, )
            
        
    
    
    # data = torch.rand(2, 128, 128, 128)
    # model = UNet(in_channels=1)
    # print(model(data))
    
    

if __name__ == "__main__":
    main()
