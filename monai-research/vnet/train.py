import time
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from monai.config import print_config
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.nets import VNet
from monai.transforms import *

from src.datasets import *
from src.transforms import *
from src.utils import *

# from monai.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"

def parser():
    pass

def train_test_process():
    
    # The parameters below can be later added on using the argparser() library
    LR = 4e-3
    WEIGHT_DECAY = 5e-4
    EPOCHS = 300
    SAVE_FREQ = 0
    VALIDAITON_FREQ = 0
    BATCH_SIZE = 4
    DROP_OUT_RATE = 0.1
    CHECKPOINT_ROOT = ""
    

    # TRAIN_DATA_PATH = "D:/BratsDataset/train"
    # VAL_DATA_PATH = "D:/BratsDataset/validation"
    
    TRAIN_DATA_PATH = "/gpfs/SHPC_Data/home/bchen9/research/BratsDataset/train"
    VAL_DATA_PATH = "/gpfs/SHPC_Data/home/bchen9/research/BratsDataset/validation"


    val_ds = ValidationDataset(VAL_DATA_PATH, label_transform())
    train_ds = TrainDataset(TRAIN_DATA_PATH, label_transform())

    train_dataloader = DataLoader(train_ds, batch_size=4, num_workers=0, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_ds, batch_size=4, num_workers=0, drop_last=True)

    model = VNet(
        in_channels=4,
        out_channels=4
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    loss_fn =  DiceLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, softmax=True)

    dice_metric = DiceMetric(include_background=True, reduction="mean")
    # dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")


    writer = SummaryWriter("runs/VNet")
    best_metric = 0
    
    for epoch in tqdm(range(EPOCHS)):
        
        epoch_start = time.time()
        model.train()
        train_loss = 0
        
        
        for batch_num, (data, label) in enumerate(train_dataloader):
            train_start = time.time()
            data, label = (
                data.to(device),
                label.to(device)
            )

            optimizer.zero_grad()
            y_pred = model(data)
            loss = loss_fn(y_pred, label)
            train_loss += loss.item()
            
            loss.backward()
            optimizer.step()

            print(
                f" {batch_num}/{len(train_ds) // train_dataloader.batch_size}"
                f", train_loss: {loss.item():.4f}"
                f", step time: {(time.time() - train_start):.4f}"
            )

        lr_scheduler.step()
        train_loss /= len(train_dataloader)
        writer.add_scalar("vnet -train_loss/epochs", train_loss, epoch)
        
        print(f"epoch {epoch + 1} average loss: {train_loss:.4f}")

        test_loss = 0
        model.eval()
        with torch.no_grad():
            eval_start = time.time()
            for batch_num, (data, y_label) in enumerate(val_dataloader):
                data, y_label = (
                    data.to(device),
                    y_label.to(device)
                )
                y_pred = model(data)
                loss = loss_fn(y_pred, label)
                test_loss += loss.item()
            
                dice_metric(y_pred=y_pred, y=label)


            test_loss /= len(val_dataloader)        
            metric = dice_metric.aggregate().item()

            writer.add_scalar("vnet -test_loss/epochs", test_loss, epoch)
            writer.add_scalar("vnet -dice_metric/epochs", metric, epoch)
            
            # if metric > best_metric:
            #     checkpoint = Checkpoint(model, optimizer, epoch, metric)
            #     checkpoint.save(SAVE_MODEL_PATH)
            #     best_metric = metric
            
            print(
                f"evaludation time consuming: {time.time() - eval_start:.4f}"
                f"current epoch: {epoch + 1} current mean dice: {metric:.4f} average test_loss: {test_loss}"
            )

            dice_metric.reset()

        print(f"time consuming of epoch {epoch + 1} is: {(time.time() - epoch_start):.4f}")  

if __name__ == "__main__":
    train_test_process()