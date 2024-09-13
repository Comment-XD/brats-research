import argparse
import os

from monai.networks import VNet
from models.unet import UNet
# from models.vnet import VNet
from src.dataloader import dataloader
from src.datasets import *
from src.losses import *
from src.utils import *

from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter

def parse_options():
    parser = argparse.ArgumentParser("settings")
    
    ### arguments for loading data
    
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--print_freq", type=int, default=10,
                        help="print frequency")
    parser.add_argument("--save_freq", type=int, default=50,
                        help="save frequency")
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

def train_test_process(args, model):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    optimizer = create_optimizer(args, model)
    train_dataloader = dataloader(args, split="train")
    val_dataloader = dataloader(args, split="val")
    
    writer = SummaryWriter(f"runs/{args.model}")
    
    loss_fn = create_loss_fn(args)
    
    for epoch in range(args.epochs):
        print(f"Epoch: {epoch}")
        train_loss = 0
        
        for batch_num, (X, y) in tqdm(enumerate(train_dataloader)):
            
            X = X.to(device)
            y = y.to(device)
            
            optimizer.zero_grad()
            
            y_pred = model(X)
            # print(y_pred.shape)
            loss = loss_fn(y_pred, y)
            train_loss += loss
            loss.backward()
            
            optimizer.step()
            
        
        train_loss /= len(train_dataloader)
        writer.add_scalar("vnet -train_loss/epochs", train_loss, epoch)
        print(f"train loss : {train_loss:.5f}")
        
        model.eval() # Sets the model to evaluation mode for BatchNorm
        
        with torch.no_grad():
            test_loss = 0
            for batch_num, (X, y) in enumerate(val_dataloader):
                
                X = X.to(device)
                y = y.to(device)
                
                y_pred = model(X)
                loss = loss_fn(y_pred, y)
                
                test_loss += loss
                
                ### Implement Save Frequency and checkpoint to for continued training
                # if batch_num % args.save_freq:
                #     pass
                
            test_loss /= len(val_dataloader)
            writer.add_scalar("vnet -train_loss/epochs", test_loss, epoch)
            print(f"test loss : {test_loss:.5f}")


def main():
    # TODO: Create the file structure so everyone's structure is the same, resulting in consistent results
    # Create a saved models folder, when running models, only use models that had the best 
    # How to train already saved models?
    
    args = parse_options()
    model = VNet(in_channels=4, out_channels=4)
    train_test_process(args, model)
    
    

if __name__ == "__main__": 
    main()
