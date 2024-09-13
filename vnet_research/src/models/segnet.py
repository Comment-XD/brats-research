import torch
import torch.nn as nn
import torch.nn.functional as F

### --------------------------- ###
# Image size is 128 x 128
# In Encoder, Reduce image size from 128 -> 64 -> 32 -> 16 -> 8
# In Decoder, Increase image size from 8 -> 16 -> 32 -> 64 -> 128
#
### --------------------------- ###

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, num_of_layers):
        super().__init__()
        self.encode_block = self._make_encoding_block(in_channels, out_channels, 
                                                      num_of_layers)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def _make_encoding_block(self, in_channels, out_channels, num_of_layers):
        encoding_block = []
        for _ in range(num_of_layers):
            encoding_block.append(nn.Conv2d(in_channels, out_channels, 
                                            kernel_size=3, padding=1))
            encoding_block.append(nn.BatchNorm2d(out_channels))
            encoding_block.append(nn.ReLU())
        
        return nn.Sequential(*encoding_block)

    def forward(self, x):
        out = self.encode_block(x)
        connection = out
        out = self.maxpool(out)
        
        return out, connection
        
class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, num_of_layers):
        super().__init__()
        self.decode_block = self._make_decode_block(in_channels, out_channels, num_of_layers)
    
    def _make_decode_block(self, in_channels, out_channels, num_of_layers):
        decoding_block = []
        for _ in range(num_of_layers):
            decoding_block.append(nn.Conv2d(out_channels, out_channels, 
                                            kernel_size=3, padding=1))
            decoding_block.append(nn.BatchNorm2d(out_channels))
            decoding_block.append(nn.ReLU())
        
        return nn.Sequential(*decoding_block)
    
    def forward(self, x, connection):
        out = self.decode_block(x)
        
        return out


class SegNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder
        self.decoder
    
    def _make_encoder():
        pass
    def _make_decoder():
        pass
        
        
        
            
            
