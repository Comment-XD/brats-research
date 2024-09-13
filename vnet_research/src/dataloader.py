from torch.utils.data import DataLoader

from src.datasets import *

# TRAIN_DATA_PATH = f"{os.getcwd()}/BratsDataset/train"
# VAL_DATA_PATH = f"{os.getcwd()}/BratsDataset/validation"

TRAIN_DATA_PATH = "D:/BratsDataset/train"
VAL_DATA_PATH = "D:/BratsDataset/validation"

print(os.path.exists(TRAIN_DATA_PATH))

def dataloader(args, split="train"):
    if split.lower() not in ["train", "val"]:
        raise ValueError(
            "arguement (split) must be either train, val, or test"
        )
    
    if split == "train":
        return DataLoader(TrainDataset(TRAIN_DATA_PATH),
                          batch_size=args.batch_size, 
                          num_workers=args.num_workers, 
                          shuffle=True, 
                          drop_last=True)
    if split == "val":
        return DataLoader(ValidationDataset(VAL_DATA_PATH), 
                          batch_size=args.batch_size, 
                          num_workers=args.num_workers, 
                          shuffle=False,
                          drop_last=True)
