#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x
    
class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x
    
class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi


# In[5]:


from .TimeDistributed import TimeDistributed
from .ConvLSTM import ConvLSTM
from .unet_parts import *

class Att_LSTM_UNet(nn.Module):
    def __init__(self, n_channels, n_classes, num_filter = 32, bilinear=False):
        super(Att_LSTM_UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = TimeDistributed(DoubleConv(n_channels, num_filter))
        self.down1 = TimeDistributed(Down(num_filter, num_filter*2))
        self.down2 = TimeDistributed(Down(num_filter*2, num_filter*4))
        self.down3 = TimeDistributed(Down(num_filter*4, num_filter*8))
        factor = 2 if bilinear else 1
        self.down4 = TimeDistributed(Down(num_filter*8, num_filter*16 // factor))
        
        self.lstm = ConvLSTM(input_dim = num_filter*16 // factor,
                    hidden_dim=num_filter*16 // factor,
                    kernel_size=(3,3),
                    num_layers=2,
                    batch_first=True,
                    bias=True,
                    return_all_layers=False)
        
        self.Up5 = up_conv(ch_in=num_filter*16,ch_out=num_filter*8)
        self.Att5 = Attention_block(F_g=num_filter*8,F_l=num_filter*8,F_int=num_filter*4)
        self.Up_conv5 = DoubleConv(in_channels=num_filter*16, out_channels=num_filter*8)

        self.Up4 = up_conv(ch_in=num_filter*8,ch_out=num_filter*4)
        self.Att4 = Attention_block(F_g=num_filter*4,F_l=num_filter*4,F_int=num_filter*2)
        self.Up_conv4 = DoubleConv(in_channels=num_filter*8, out_channels=num_filter*4)
        
        self.Up3 = up_conv(ch_in=num_filter*4,ch_out=num_filter*2)
        self.Att3 = Attention_block(F_g=num_filter*2,F_l=num_filter*2,F_int=num_filter)
        self.Up_conv3 = DoubleConv(in_channels=num_filter*4, out_channels=num_filter*2)
        
        self.Up2 = up_conv(ch_in=num_filter*2,ch_out=num_filter)
        self.Att2 = Attention_block(F_g=num_filter,F_l=num_filter,F_int=num_filter//2)
        self.Up_conv2 = DoubleConv(in_channels=num_filter*2, out_channels=num_filter)

        self.Conv_1x1 = nn.Conv2d(num_filter,n_classes,kernel_size=1,stride=1,padding=0)
        self.decision=nn.Sigmoid()
        
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        _ , [(h, _)] = self.lstm(x5)
        
        # decoding + concat path
        d5 = self.Up5(h)
        x4 = self.Att5(g=d5,x=x4[:,-1,...])
        d5 = torch.cat((x4,d5),dim=1)        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4,x=x3[:,-1,...])
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3,x=x2[:,-1,...])
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2,x=x1[:,-1,...])
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)
        
        logits = self.Conv_1x1(d2)
        logits=self.decision(logits)

        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)