import argparse
import os
from src.datasets import *
from src.utils import *
import numpy as np
from models.unet import UNet


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
    
    parser.add_argument("--lr", type=float, default=0.04)
    parser.add_argument("--lr_decay", type=float)
    parser.add_argument("--epochs", type=int, default=10)
    
    opt = parser.parse_args()
    
    return opt


def main():
    # args = parse_options()
    # train_dataloader = dataloader(args, split="train")
    # train_feature, train_label = next(iter(train_dataloader))
    # print(train_feature)

    data = torch.rand(2, 1, 572, 572)
    model = UNet(in_channels=1)
    print(model(data))
    
    

if __name__ == "__main__":
    main()
