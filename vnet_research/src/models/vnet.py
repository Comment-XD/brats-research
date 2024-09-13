import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm


class Upsample(nn.Module):
    def __init__(self, in_channels:int, out_channels:int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        
        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, 
                               kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(in_channels=out_channels, out_channels=out_channels, 
                               kernel_size=3, padding=1)
        self.batchnorm1 = nn.BatchNorm3d(num_features=out_channels)
        self.batchnorm2 = nn.BatchNorm3d(num_features=out_channels)
        
        self.maxpool = nn.MaxPool3d(kernel_size=2)
        
    
    def forward(self, x):        
        out = F.relu(self.batchnorm1(self.conv1(x)))
        out = F.relu(self.batchnorm2(self.conv2(out)))
        connection = out
        out = self.maxpool(out)
        
        return out, connection
    
    def __repr__(self):
        return f"Upsample(in={self.in_channels}, out={self.out_channels})"


class Downsample(nn.Module):
    def __init__(self, in_channels:int, out_channels:int):
        super().__init__()

        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(in_channels=out_channels, out_channels=out_channels // 2,
                               kernel_size=3, padding=1)
        
        self.conv_transpose1 = nn.ConvTranspose3d(in_channels=out_channels // 2, 
                                                  out_channels=out_channels // 2,
                                                  kernel_size=2,
                                                  stride=2)
        
        self.batchnorm1 = nn.BatchNorm3d(num_features=out_channels)
        self.batchnorm2 = nn.BatchNorm3d(num_features=out_channels // 2)
    
    
    def forward(self, x:torch.Tensor, connection:torch.Tensor, end:bool=False):
        
        # x_width, x_height = (x.shape[-2], x.shape[-1])
        # crop = transforms.RandomCrop((x_width, x_height))
        # crop_connection = crop(connection)
        
        # print(x.size() == connection.size())
        
        out = torch.cat([x, connection], dim=1) # Concatanating them by the channels 
        out = F.relu(self.batchnorm1(self.conv1(out)))
        
        if not end:
            out = F.relu(self.batchnorm2(self.conv2(out)))
            out = self.conv_transpose1(out)

        return out

    def __repr__(self):
        return f"Downsample(in={self.in_channels}, out={self.out_channels})"


class BottleNeck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(in_channels=out_channels, out_channels=out_channels,
                               kernel_size=3, padding=1)
        self.conv_transpose1 = nn.ConvTranspose3d(in_channels=out_channels, 
                                                  out_channels=in_channels,
                                                  kernel_size=2,
                                                  stride=2)
        
        self.batchnorm1 = nn.BatchNorm3d(num_features=out_channels)
        self.batchnorm2 = nn.BatchNorm3d(num_features=out_channels)
        
    def forward(self, x):
        out = F.relu(self.batchnorm1(self.conv1(x)))
        out = F.relu(self.batchnorm2(self.conv2(out)))
        out = self.conv_transpose1(out)
        
        return out
    
    def __repr__(self):
        return f"BottleNeck(in={self.in_channels}, out={self.out_channels})"
        
class VNet(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.u1 = Upsample(in_channels, 16)
        self.u2 = Upsample(16, 32)
        self.u3 = Upsample(32, 64)
        self.u4 = Upsample(64, 128)
        
        self.bottleneck = BottleNeck(128, 256)
        
        self.d1 = Downsample(256, 128)
        self.d2 = Downsample(128, 64)
        self.d3 = Downsample(64, 32)
        self.d4 = Downsample(32, 16)
        
        self.segmenter = nn.Sequential(
            nn.Conv3d(16, 16, kernel_size=3, padding=1), 
            nn.Conv3d(16, out_channels, kernel_size=1)
        )
    
    ### TODO: Could we simplify this u1, u2, u3. u4 to just loops?
    
    def forward(self, x):
        x = x.to(torch.float32) #why do i need to point my data in floating point 32?
        out, c4 = self.u1(x)
        out, c3 = self.u2(out)
        out, c2 = self.u3(out)
        out, c1 = self.u4(out)
        
        out = self.bottleneck(out)

        out = self.d1(out, c1)
        out = self.d2(out, c2)
        out = self.d3(out, c3)
        out = self.d4(out, c4, end=True)
        
        out = self.segmenter(out)
        out = F.softmax(out, dim=1)
        
        return out
    
    def save(self, save_path):
        ### Saves the model after a period of time 
        # torch.save(self.state_dict(), save_path)
        
        pass
    
    def predict(self, x):
        pass
    
        ### TODO: Later on used to compare the segmentation of an output, probably going to use pyplot to begin with
    
    def __repr__(self):
        return f"UNet(in={self.in_channels}, out={self.out_channels})"