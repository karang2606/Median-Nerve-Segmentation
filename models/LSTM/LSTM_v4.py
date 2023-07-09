#!/usr/bin/env python
# coding: utf-8

# In[1]:


from TimeDistributed import TimeDistributed
from ConvLSTM import ConvLSTM
from unet_parts import *

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_ch, out_ch, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_ch, out_ch, in_ch // 2)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_ch, in_ch//2)
        self.conv2 = nn.Conv2d(in_ch//2, out_ch, kernel_size=1)

    def forward(self, x):
        x = torch.cat(x, dim=1)
        x = self.conv(x)
        x = self.up(x)
        return self.conv2(x)
    
class ConvLSTM_BLock(nn.Module):
    
    def __init__(self, in_ch, out_ch):
        super(ConvLSTM_BLock, self).__init__()
        self.layer1 = ConvLSTM(input_dim = in_ch,
                    hidden_dim=in_ch,
                    kernel_size=(3,3),
                    num_layers=1,
                    batch_first=True,
                    bias=True,
                    return_all_layers=False)
        
        self.layer2 = ConvLSTM(input_dim = in_ch,
                    hidden_dim=(in_ch+out_ch)//2,
                    kernel_size=(3,3),
                    num_layers=1,
                    batch_first=True,
                    bias=True,
                    return_all_layers=False)
        
        self.layer3 = ConvLSTM(input_dim = (in_ch+out_ch)//2,
                    hidden_dim=out_ch,
                    kernel_size=(3,3),
                    num_layers=1,
                    batch_first=True,
                    bias=True,
                    return_all_layers=False)
        
    def forward(self, x):

        [x], _ = self.layer1(x)
        [x], _ = self.layer2(x)
        _ , [(h, _)]  = self.layer3(x)
        return h
    
class LSTM_UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, num_filter=64, bilinear=False):
        
        n = [num_filter*2**i for i in range(4)] #[64,128,256,512,512]
        n.append(n[-1])
        t_filter = [2,8,32,128]
        n_filter = [n[i] - t_filter[i] for i in range(4)] #[62,120,224,384]
        
        super(LSTM_UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = TimeDistributed(DoubleConv(1, n_filter[0]))
        self.down1 = TimeDistributed(Down(n_filter[0], n_filter[1]))
        self.down2 = TimeDistributed(Down(n_filter[1], n_filter[2]))
        self.down3 = TimeDistributed(Down(n_filter[2], n_filter[3]))
        factor = 2 if bilinear else 1
        self.down4 = TimeDistributed(Down(n_filter[3], n[4]))
        
        self.lstm1 = ConvLSTM_BLock(n_filter[0], t_filter[0])
        self.lstm2 = ConvLSTM_BLock(n_filter[1], t_filter[1])
        self.lstm3 = ConvLSTM_BLock(n_filter[2], t_filter[2])
        self.lstm4 = ConvLSTM_BLock(n_filter[3], t_filter[3])
        self.lstm5 = ConvLSTM_BLock(n[4], n[4])
        
        # n = [64,128,256,512,512]
        self.up4 = (Up(n[3]*2, n[3], bilinear))
        self.up3 = (Up(n[3]*2, n[2], bilinear))
        self.up2 = (Up(n[2]*2, n[1], bilinear))
        self.up1 = (Up(n[1]*2, n[0], bilinear))
        self.outc = (OutConv(n[0]*2, 1))
        self.decision=nn.Sigmoid()

        
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        t1 = self.lstm1(x1)
        t2 = self.lstm2(x2)
        t3 = self.lstm3(x3)
        t4 = self.lstm4(x4)
        t5 = self.lstm5(x5)
        
        x = self.up4([x5[:,-1,...],t5])
        x = self.up3([x4[:,-1,...],t4,x])
        x = self.up2([x3[:,-1,...],t3,x])
        x = self.up1([x2[:,-1,...],t2,x])

        logits = self.outc(torch.cat([x1[:,-1,...],t1,x], dim = 1))
        logits=self.decision(logits)

        return logits