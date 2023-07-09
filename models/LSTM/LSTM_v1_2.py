#!/usr/bin/env python
# coding: utf-8

# In[1]:


from TimeDistributed import TimeDistributed
from ConvLSTM import ConvLSTM
""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, dropout = 0.2):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up_(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

    def forward(self, x):
        return self.up(x)
    
class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class LSTM_UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, num_filter = 64, bilinear=False):
        super(LSTM_UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, num_filter))
        self.down1 = (Down(num_filter, num_filter*2))
        self.down2 = (Down(num_filter*2, num_filter*4))
        self.down3 = (Down(num_filter*4, num_filter*8))
        factor = 2 if bilinear else 1
        self.down4 = (Down(num_filter*8, num_filter*16 // factor))
        
        self.lstm = ConvLSTM(input_dim = num_filter*16 // factor,
                    hidden_dim=num_filter*16 // factor,
                    kernel_size=(3,3),
                    num_layers=2,
                    batch_first=True,
                    bias=True,
                    return_all_layers=False)
        
        self.lf1x1 = nn.Conv2d(num_filter*16, num_filter*8, kernel_size=1)
        self.lstm1x1 = nn.Conv2d(num_filter*16, num_filter*8, kernel_size=1)
        
        self.up1 = (Up(num_filter*16, num_filter*8 // factor, bilinear))
        self.up2 = (Up(num_filter*8, num_filter*4 // factor, bilinear))
        self.up3 = (Up(num_filter*4, num_filter*2 // factor, bilinear))
        self.up4 = (Up(num_filter*2, num_filter, bilinear))
        self.outc = (OutConv(num_filter, n_classes))
        self.decision=nn.Sigmoid()
        
    def forward(self, x):
        x1 = self.inc(x[:,-1,...])
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        with torch.no_grad():
            x1_ = TimeDistributed(self.inc)(x[:,:-1,...].contiguous())
            x2_ = TimeDistributed(self.down1)(x1_)
            x3_ = TimeDistributed(self.down2)(x2_)
            x4_ = TimeDistributed(self.down3)(x3_)
            x5_ = TimeDistributed(self.down4)(x4_)
            
        _ , [(h, _)] = self.lstm(torch.cat((x5_,x5.unsqueeze(1)),1))
        
        x6 = self.lf1x1(x5)
    
        x = self.lstm1x1(h)
        
        x = self.up1(torch.cat([x,x6], dim=1), x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
        
        t1 = self.lstm1(torch.cat((x1_,x1.unsqueeze(1)),1))
        t2 = self.lstm2(torch.cat((x2_,x2.unsqueeze(1)),1))
        t3 = self.lstm3(torch.cat((x3_,x3.unsqueeze(1)),1))
        t4 = self.lstm4(torch.cat((x4_,x4.unsqueeze(1)),1))
        t5 = self.lstm5(torch.cat((x5_,x5.unsqueeze(1)),1))
        
        x = self.up4([x5,t5])
        x = self.up3([x4,t4,x])
        x = self.up2([x3,t3,x])
        x = self.up1([x2,t2,x])

        logits = self.outc(torch.cat([x1,t1,x], dim = 1))
        logits=self.decision(logits)

        return logits