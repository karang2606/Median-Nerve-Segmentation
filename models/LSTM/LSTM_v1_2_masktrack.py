#!/usr/bin/env python
# coding: utf-8

# In[1]:


from .TimeDistributed import TimeDistributed
from .ConvLSTM import ConvLSTM
from .unet_parts import *

class LSTM_UNet(nn.Module):
    def __init__(self, n_channels, n_classes, num_filter = 64, bilinear=False):
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
        
    def forward(self, x, y):
        x1 = self.inc(torch.cat((x[:,-1,...],y[:,-1,...]), dim=1))
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        with torch.no_grad():
            x1_ = TimeDistributed(self.inc)(torch.cat((x[:,:-1,...],y[:,:-1,...]), dim=2))
#                                             contiguous())
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
        logits=self.decision(logits)
        return logits