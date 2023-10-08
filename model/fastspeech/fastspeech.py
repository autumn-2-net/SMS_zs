import torch
import torch.nn as nn
import torch.nn.functional as F

from model.attention.attention_lay import attn_lay


class fs_encoder(nn.Module):
    def __init__(self,dim,lays, heads=4, dim_head=64,hideen_dim=None,kernel_size=9):
        super().__init__()
        self.attn_lay=nn.ModuleList([attn_lay(dim=dim, heads=heads, dim_head=dim_head,hideen_dim=hideen_dim,kernel_size=kernel_size) for _ in range(lays)])

    def forward(self,x,mask=None):
        for i in self.attn_lay:
            x=i(x,mask)
        return x