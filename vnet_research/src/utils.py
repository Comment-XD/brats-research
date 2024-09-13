import torch
from src.datasets import *
from src.dataloader import *
from monai.losses import DiceLoss

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def create_loss_fn(args):
    if args.loss.lower() == "dice":
        return DiceLoss(include_background=False, 
                        softmax=True, 
                        reduction="mean")
     
    
    if args.loss.lower() == "focal":
        # This is not finished
        # return FocalLoss()
        return

def create_optimizer(args, model):
    if args.optimizer.lower() == "sgd":
        return torch.optim.SGD(model.parameters(), args.lr)
    if args.optimizer.lower() == "adam":
        return torch.optim.Adam(model.parameters(), args.lr)
        

# def get_mean_std(dataloader):
#     total_mean, total_mean_squared, total_batches = 0, 0, 0
#     for _, (X, y) in enumerate(dataloader):
#         X = X.permute(0, -1, 1, 2, 3)
#         for img in X:
#             # print(torch.mean(img, dim=[0,1,2]))
#             total_mean += torch.mean(img, dim=[0, 1, 2]) 
#             total_mean_squared += torch.mean(img ** 2, dim=[0, 1, 2])
    
#         total_batches += 1
    
#     mean = total_mean / total_batches
#     std = (total_mean_squared / total_batches - mean**2)**0.5
    
#     return mean, std