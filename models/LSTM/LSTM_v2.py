#!/usr/bin/env python
# coding: utf-8

# In[1]:


from TimeDistributed import TimeDistributed
from ConvLSTM import ConvLSTM
from unet_parts import *

class LSTM_UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, bilinear=False):
        super(LSTM_UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = TimeDistributed(DoubleConv(1, 64))
        self.down1 = TimeDistributed(Down(64, 128))
        self.down2 = TimeDistributed(Down(128, 256))
        self.down3 = TimeDistributed(Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = TimeDistributed(Down(512, 512))
        
        self.lstm = ConvLSTM(input_dim = 512,
                    hidden_dim=512,
                    kernel_size=(3,3),
                    num_layers=2,
                    batch_first=True,
                    bias=True,
                    return_all_layers=False)
        
        self.comp1 = nn.Conv2d(64, 62, kernel_size=1) #2
        self.comp2= nn.Conv2d(128, 120, kernel_size=1) #8
        self.comp3= nn.Conv2d(256, 224, kernel_size=1) #32
        self.comp4= nn.Conv2d(512, 384, kernel_size=1) #128
        
        self.lstm_up4 = nn.ConvTranspose2d(512, 128, kernel_size=2, stride=2)
        self.lstm_up3 = nn.ConvTranspose2d(512, 32, kernel_size=4, stride=4)
        self.lstm_up2 = nn.ConvTranspose2d(512, 8, kernel_size=8, stride=8)
        self.lstm_up1 = nn.ConvTranspose2d(512, 2, kernel_size=16, stride=16)

        self.up1 = (Up(1024, 512, bilinear))
        self.up2 = (Up(512, 256, bilinear))
        self.up3 = (Up(256, 128, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, 1))
        self.decision=nn.Sigmoid()

        
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        _ , [(h, _)] = self.lstm(x5)
        
        l1 = self.lstm_up1(h)
        l2 = self.lstm_up2(h)
        l3 = self.lstm_up3(h)
        l4 = self.lstm_up4(h)
        
        c1 = self.comp1(x1[:,-1,...])
        c2 = self.comp2(x2[:,-1,...])
        c3 = self.comp3(x3[:,-1,...])
        c4 = self.comp4(x4[:,-1,...])

        x = self.up1(torch.cat([x5[:,-1,...],h], dim=1), torch.cat([l4,c4], dim=1))
        x = self.up2(x, torch.cat([l3,c3], dim=1))
        x = self.up3(x, torch.cat([l2,c2], dim=1))
        x = self.up4(x, torch.cat([l1,c1], dim=1))
        logits = self.outc(x)
        logits=self.decision(logits)

        return logits