#!/usr/bin/env python
# coding: utf-8

# In[1]:


from torchsummary import summary

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



# In[2]:


class Siam_UNet_train(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, num_filter = 32, bilinear=False):
        super(Siam_UNet_train, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, num_filter))
        self.down1 = (Down(num_filter, num_filter*2))
        self.down2 = (Down(num_filter*2, num_filter*4))
        self.down3 = (Down(num_filter*4, num_filter*8))
        factor = 2 if bilinear else 1
        self.down4 = (Down(num_filter*8, num_filter*16 // factor))
        
        self.bottom = nn.Conv2d(in_channels=1, out_channels=num_filter*16, kernel_size=3, padding='same', bias=True)
        
        self.up1 = (Up(num_filter*16, num_filter*8 // factor, bilinear))
        self.up2 = (Up(num_filter*8, num_filter*4 // factor, bilinear))
        self.up3 = (Up(num_filter*4, num_filter*2 // factor, bilinear))
        self.up4 = (Up(num_filter*2, num_filter, bilinear))
        self.outc = (OutConv(num_filter, n_classes))
        self.decision = nn.Sigmoid()
        
    def forward(self, x_A, x_B):
        x1_A = self.inc(x_A)
        x2_A = self.down1(x1_A)
        x3_A = self.down2(x2_A)
        x4_A = self.down3(x3_A)
        x5_A = self.down4(x4_A)
        
        with torch.no_grad():
            x1_B = self.inc(x_B)
            x2_B = self.down1(x1_B)
            x3_B = self.down2(x2_B)
            x4_B = self.down3(x3_B)
            x5_B = self.down4(x4_B)
        
        b, c, h, w = x5_A.shape
        
        out = F.conv2d(x5_A.reshape(1, b * c, h, w), x5_B, groups = b, padding='same')
        out = out.permute(1, 0, 2, 3)
        
        bottom = self.bottom(out)
        
        x = self.up1(bottom, x4_A)
#         x = self.up1(x5_A, x4_A)
        x = self.up2(x, x3_A)
        x = self.up3(x, x2_A)
        x = self.up4(x, x1_A)
        logits = self.outc(x)
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

# In[3]:


class Siam_UNet_test(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, num_filter = 32, bilinear=False):
        super(Siam_UNet_test, self).__init__()
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, num_filter))
        self.down1 = (Down(num_filter, num_filter*2))
        self.down2 = (Down(num_filter*2, num_filter*4))
        self.down3 = (Down(num_filter*4, num_filter*8))
        factor = 2 if bilinear else 1
        self.down4 = (Down(num_filter*8, num_filter*16 // factor))
        
        self.bottom = nn.Conv2d(in_channels=1, out_channels=num_filter*16, kernel_size=3, padding='same', bias=True)
        
        self.up1 = (Up(num_filter*16, num_filter*8 // factor, bilinear))
        self.up2 = (Up(num_filter*8, num_filter*4 // factor, bilinear))
        self.up3 = (Up(num_filter*4, num_filter*2 // factor, bilinear))
        self.up4 = (Up(num_filter*2, num_filter, bilinear))
        self.outc = (OutConv(num_filter, n_classes))
        self.decision=nn.Sigmoid()

    def forward(self, x):
        d0 = self.inc(x)
        d1 = self.down1(d0)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)

        u4 = self.up1(d4, d3)
        u3 = self.up2(u4, d2)
        u2 = self.up3(u3, d1)
        u1 = self.up4(u2, d0)
        logits = self.outc(u1)
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


# In[4]:


def main():
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Siam_UNet_train(n_channels=1, n_classes=1).to(device)

    summary(model, [(1,448, 336), ([1,80, 128])], batch_size = -1)


# In[4]:


if __name__ == "__main__":
    main()

