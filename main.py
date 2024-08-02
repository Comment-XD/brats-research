import argparse
import os

from src.datasets import *
from src.utils import *
from src.losses import *
from src.dataloader import dataloader
from models.unet import UNet
from models.vnet import VNet


def parse_options():
    parser = argparse.ArgumentParser("settings")
    
    ### arguments for loading data
    
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--print_freq", type=int, default=10,
                        help="print frequency")
    parser.add_argument("--save_freq", type=int, default=50,
                        help="save frequency")
    parser.add_argument("")
    
    ### arguments for training hyperparameters
    
    parser.add_argument("--optimizer", type=str, default="sgd")
    parser.add_argument("--lr", type=float, default=0.04)
    parser.add_argument("--lr_decay", type=int, default=0.01)
    parser.add_argument("--epochs", type=int, default=10) 
    
    parser.add_argument('--model', type=str, default="vnet",
                        help='which model to use')
    parser.add_argument('--loss', type=str, default="dice",
                        help='which loss function to use')
    
    opt = parser.parse_args()
    
    return opt


def main():
    # TODO: Create the file structure so everyone's structure is the same, resulting in consistent results
    # Create a saved models folder, when running models, only use models that had the best 
    # How to train already saved models?
    
    # TODO: Splitting the entire dataset
    
    args = parse_options()
    model = VNet(in_channels=4, out_channels=4)
    train_test_process(args, model)
    
    
    # args = parse_options()
    train_dataloader = dataloader(args, split="train")
    train_feature, train_label = next(iter(train_dataloader))
    
    
    
    
    # print(train_feature.dtype)
    # model(train_feature)
            
    # data = torch.rand(4, 4, 128, 128, 128)
    # print(model(train_feature))
    # model = UNet(in_channels=1)
    # print(model(data))
    
    

if __name__ == "__main__": 
    main()
