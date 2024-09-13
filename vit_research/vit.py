from functools import partial
from typing import Optional, Sequence, List, Union
import math

import torch
import torch.nn as nn

# Embedding_dimensions determines the number of embeds
# Number of patches equal original image dimension (I_h * I_w) / (P_h, P_w)

class DropPath(nn.Module):
    def __init__(self, drop_prob:float=0.0):
        """
        A Stocastic Path method that allows regularization in a model, preventing overfitting

        Args:
            drop_prob (float, optional): The probability of dropping a path. Defaults to 0.0.
        """
        super().__init__()
        self.drop_prob = drop_prob
        
    
    def forward(self, x):
        if self.drop_prob == 0.0 or self.training:
            return x 
        
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask_tensor = keep_prob + torch.rand(size=shape, dtype=x.dtype, device=x.device)
        mask_tensor.floor_()
        
        out = x.div(keep_prob) * mask_tensor
        return out
        
class MLP(nn.Module):
    def __init__(self, in_feats:int, hid_feats:int=None, out_feats:int=None, act_layer=nn.GELU, mlp_drop:float=0.2):
        """
        Multi-Layer Perceptron
        
        Args:
            in_chans (int): input_channels
            hidden_chans (int, optional): hidden_channels. Defaults to None.
            out_chans (int, optional): out_channels. Defaults to None.
            act_layer (_type_, optional): Activation Layer. Defaults to nn.GELU.
            mlp_drop (float, optional): Drop Probability for MLP. Defaults to 0.2.
        """
        
        super().__init__()
        out_features = out_feats or in_feats
        hidden_features = hid_feats or in_feats
        
        self.mlp_block = nn.Sequential(
            nn.Linear(in_features=in_feats, out_features=hidden_features),
            act_layer(),
            nn.Linear(in_features=hidden_features, out_features=out_features),
            nn.Dropout(mlp_drop)
        )
    
    def forward(self, x):
        return self.mlp_block(x)
        

class PatchEmbedding(nn.Module):
    def __init__(self, in_chan:int, patch_dim:int, patch_nums:int, embed_dim:int, patch_drop:float):
        """
        Creates Patch Embeddings of embed dim

        Args:
            in_chan (int): _description_
            patch_dim (int): _description_
            patch_nums (int): _description_
            embed_dim (int): _description_
        """
        
        super().__init__()
        self.flatten = nn.Flatten(2)  
        
        # Creates out embeddings
        self.patcher = nn.Conv2d(in_channels=in_chan, 
                                    out_channels=embed_dim, 
                                    kernel_size=patch_dim,
                                    stride=patch_dim)
        
        self.cls_token = nn.Parameter(torch.randn(size=(1, 1, embed_dim), requires_grad=True))
        self.dropout = nn.Dropout(patch_drop)
        
        # Creates our positional encodings 
        self.positional_encodings = nn.Parameter(torch.randn(size=(1, patch_nums+1, embed_dim), requires_grad=True))
        

    def forward(self, x):
        # Modify Class Token [1, 1, Embed_Dim] -> [Batch_Size, 1, Embed_Dims]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1) # Obtains the batch_size
        
        out = self.patcher(x)
        # dimension: [Batch_Size, Embed_Dims, Num_Patches] -> [Batch_Size, Num_Patches, Embed_Dims]
        out = self.flatten(out).permute(0, 2, 1) 
        
        # Concats ([Batch_Size, Num_Patches, Embed_Dims], [Batch_Size, 1, Embed_Dims], dim=1) -> [Batch_Size, Num_Patches + 1, Embed_Dims]
        out = torch.cat([out, cls_token], dim=1)
        
        # [Batch_Size, Num_Patches + 1, Embed_Dims] + [1, Num_Patches + 1, Embed_Dims]
        out += self.positional_encodings
        out = self.dropout(out)
        
        # Returns dimensions of [Batch_Size, Num_Patches + 1, Embed_Dims]
        return out


class Attention(nn.Module):
    def __init__(self, h:int=4, embed_dim:int=1, attn_drop:float=0.2, proj_drop:int=0.2, qkv_bias:bool=False):
        """Creates the Attention Mechanism for Transformer Models

        Args:
            h (int): number of heads
            embed_dim (int): the embedding dimension 
            drop_prob (float): drop_prob for Dropout
        """
        super().__init__()
        assert embed_dim  % h == 0, "Uneven dimensions for head"
        
        self.h = h
        self.head_dim = embed_dim // h
        
        self.Qw = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.Kw = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.Vw = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.Ow = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        
        self.attn_drop = nn.Dropout(p=attn_drop)
        self.proj_drop = nn.Dropout(p=proj_drop)
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Create the Query. Key, and Value Matrices
        # Dimensions: [seq, ]
        Q = self.Qw(x)
        K = self.Kw(x)
        V = self.Vw(x)
        
        # [Batch_Size, -1, head * head_dim] -> [Batch_size, head, -1, head_dim]
        
        Q = Q.view(batch_size, -1, self.h, self.head_dim).transpose(1,2)
        K = K.view(batch_size, -1, self.h, self.head_dim).transpose(1,2)
        V = V.view(batch_size, -1, self.h, self.head_dim).transpose(1,2)
        # print(f"Q, K, V dimensions: {Q.size()}")
        
        # Tranposes the last dimensions of 
        # K [Batch_size, head, sentence_length, head_dim] -> [Batch_size, head, head_dim, sentence_length]
        scaled_dot_prod = (Q @ K.transpose(2,3)) / math.sqrt(self.head_dim)
        # print(f"scaled_dot_prod dimensions: {scaled_dot_prod.size()}")
        attention_prob = scaled_dot_prod.softmax(dim=-1) @ V
        
        attention_score = self.attn_drop(attention_prob)
        attention_score = attention_score.view(batch_size, -1, self.h * self.head_dim)
        
        A = self.Ow(attention_score)
        A = self.proj_drop(A)
        
        return A, attention_score
        
        
class TransformerBlock(nn.Module):
    def __init__(self, h:int, embed_dim:int, attn_drop:float=0.2, proj_drop:float=0.2, mlp_drop=0.2, drop_path:float=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, 
                 qkv_bias:bool=False, mlp_hid_ratio:float=0.4):
        """_summary_

        Args:
            h (int): _description_
            embed_dim (int): _description_
            attn_drop (float, optional): _description_. Defaults to 0.2.
            proj_drop (float, optional): _description_. Defaults to 0.2.
            mlp_drop (float, optional): _description_. Defaults to 0.2.
            drop_path (float, optional): _description_. Defaults to 0.0.
            act_layer (_type_, optional): _description_. Defaults to nn.GELU.
            norm_layer (_type_, optional): _description_. Defaults to nn.LayerNorm.
            qkv_bias (bool, optional): _description_. Defaults to False.
            mlp_hid_ratio (float, optional): _description_. Defaults to 0.4.
        """
        super().__init__()
        
        # Layer Normalizations
        self.norm1 = norm_layer(embed_dim)
        self.norm2 = norm_layer(embed_dim)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity() 
        
        self.attention = Attention(h=h, embed_dim=embed_dim, attn_drop=attn_drop, proj_drop=proj_drop, qkv_bias=qkv_bias)
        hidden_feats = int(embed_dim * mlp_hid_ratio)
        self.mlp = MLP(embed_dim, hid_feats=hidden_feats, act_layer=act_layer, mlp_drop=mlp_drop)
        
    def forward(self, x, return_att:bool=False):
        out, attention_scores = self.attention(self.norm1(x))
        
        if return_att:
            return attention_scores
        
        x += self.drop_path(out)
        out += self.drop_path(self.mlp(self.norm2(x)))
        
        return out


class VIT(nn.Module):
    def __init__(self, in_chans:int, patch_dim:int, patch_nums:int, embed_dim:int, depth:int=12, drop_decay_rate:float=0.0, h:int=8, attn_drop:float=0.2, proj_drop:float=0.2, patch_drop:int=0.2, 
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, qkv_bias:bool=False, mlp_drop=0.2, mlp_hid_ratio:float=0.4):
        """
        Vision Transformer 

        Args:
            in_chans (int): image input channels
            patch_dim (int): patch dimension for patch embedding
            patch_nums (int): total number of patch embeddings
            embed_dim (int): embed dimension
            depth (int, optional): number of transformer blocks. Defaults to 12.
            drop_decay_rate (float, optional): rate of drop decay for stocastic depth decay. Defaults to 0.0.
            h (int, optional): number of heads. Defaults to 8.
            attn_drop (float, optional): attention drop rate . Defaults to 0.2.
            proj_drop (float, optional): projection drop rate. Defaults to 0.2.
            patch_drop (int, optional): patch_embedding drop rate. Defaults to 0.2.
            act_layer (_type_, optional): activation layer for mlp. Defaults to nn.GELU.
            norm_layer (_type_, optional): layer normalization. Defaults to nn.LayerNorm.
            qkv_bias (bool, optional): query, weight, value biases. Defaults to False.
            mlp_drop (float, optional): mlp drop probability. Defaults to 0.2.
            mlp_hid_ratio (float, optional): mlp hidden layer ratio. Defaults to 0.4.
        """
        super().__init__()

        self.patch_embedder = PatchEmbedding(
            in_chan=in_chans, patch_dim=patch_dim, patch_nums=patch_nums, embed_dim=embed_dim, patch_drop=patch_drop)
        
        # stocastic depth decay
        self.drop_rates = [x.item() for x in torch.linspace(0, drop_decay_rate, depth)]
        
        self.transformer_module = nn.ModuleList([
            TransformerBlock(
                h=h, embed_dim=embed_dim, attn_drop=attn_drop, proj_drop=proj_drop, mlp_drop=mlp_drop, drop_path=self.drop_rates[i], 
                act_layer=act_layer, norm_layer=norm_layer, qkv_bias=qkv_bias, mlp_hid_ratio=mlp_hid_ratio)
            for i in range(depth)])
        
        self.norm = norm_layer(embed_dim)
    
    
    def _init_weights(self):
        pass
    
    def forward(self, x):
        out = self.patch_embedder(x)
        for block in self.transformer_module:
            out = block(out)
        
        out = self.norm(out)
        
        return out


def vit_tiny():
    return VIT(in_chans=3, patch_nums=16, embed_dim=192, depth=12, num_heads=3, mlp_hid_ratio=0.4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))

def vit_small():
    return VIT(in_chans=3, patch_nums=16, embed_dim=384, depth=12, num_heads=6, mlp_hid_ratio=0.4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))

def vit_base():
    return VIT(in_chans=3, patch_nums=16, embed_dim=768, depth=12, num_heads=12, mlp_hid_ratio=0.4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))

vit_dict = {"vit_tiny": [17 * 192, vit_tiny],
            "vit_small": [17 * 384, vit_small],
            "vit_base": [17 * 768, vit_base]}

class VitEncoder(nn.Module):
    def __init__(self, name:str="vit_small", proj_type:str="mlp", output_embed_dim:int=2048):
        """_summary_

        Args:
            name (str, optional): _description_. Defaults to "vit_small".
            proj_type (str, optional): _description_. Defaults to "mlp".
            output_embed_dim (int, optional): _description_. Defaults to 2048.
        """
        super().__init__()
        
        self.model, in_feats = vit_dict[name]
        
        if proj_type == "mlp":
            self.proj_layer = nn.Sequential(
                nn.Linear(in_feats, in_feats),
                nn.Linear(in_feats, output_embed_dim))
        
        if proj_type == "linear":
            self.proj_layer = nn.Sequential(
                nn.Linear(in_feats, output_embed_dim))
        
    def _init_weights(self):
        pass

    def forward(self, x):
        out = self.model(x)
        out = self.proj_layer(out)
        
        return out
        
        
   
def main():   
    pass

def test():
    pass
        
if __name__ == "__main__":
    main()
    # test()
    