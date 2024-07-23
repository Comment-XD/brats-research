import torch

def get_mean_std(dataloader):
    total_mean, total_mean_squared, total_batches = 0, 0, 0
    for _, data in enumerate(dataloader):
        for img in data:
            total_mean += torch.mean(img, dim=[0, 1, 2]) 
            total_mean_squared += torch.mean(img ** 2, dim=[0, 1, 2])
    
        total_batches += 1
    
    mean = total_mean / total_batches
    std = (total_mean_squared / total_batches - mean**2)**0.5
    
    return mean, std


def augment(img):
    pass