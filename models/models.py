#!/usr/bin/env python
# coding: utf-8

# In[ ]:
from pathlib import Path
import torch
import random
from models.VisTR.util import misc as utils
from models.VisTR.models import build_model


def get_model(args):
    if args.model_name == 'unet':
        from models.UNet import UNet
        model = UNet(n_channels=1, n_classes=1, num_filter= 64)
        args.output_dir = 'checkpoints_' + args.model_name
    
    elif args.model_name == 'unetpp':
        from models.UNet_plusplus import NestedUNet
        model = NestedUNet(n_channels=1, n_classes=1, num_filter= 64)
        args.output_dir = 'checkpoints_' + args.model_name
    
    elif args.model_name == 'attn_unet':
        from models.Attention_UNet import Att_UNet
        model = Att_UNet(n_channels=1, n_classes=1, num_filter=64)
        args.output_dir = 'checkpoints_' + args.model_name
    
    elif args.model_name == 'siam_unet':
        from models.Siam_UNet import Siam_UNet_train
        model = Siam_UNet_train(n_channels=1, n_classes=1, num_filter=32)
        args.output_dir = 'checkpoints_' + args.model_name
    
    elif args.model_name == 'lstm_unet':
        from models.LSTM.LSTM_v1 import LSTM_UNet
        model = LSTM_UNet(n_channels=1, n_classes=1, num_filter = 32 )
        args.output_dir = 'checkpoints_' + args.model_name
    
    elif args.model_name == 'trans_unet':
        from models.TransUNet.networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
        from models.TransUNet.networks.vit_seg_modeling import VisionTransformer as ViT_seg

        config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
        config_vit.n_classes = 1
        config_vit.n_skip = 3
        config_vit.patches.grid = (int(448 / 16), int(336 / 16))

        model =  ViT_seg(config_vit, img_size = (448,336), num_classes=config_vit.n_classes)

        args.output_dir = 'checkpoints_' + args.model_name
    
    elif args.model_name == 'vistr':
        args.output_dir = 'checkpoints_' + args.model_name
        
        if args.output_dir:
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        device = torch.device(args.device)

        args.pretrained_weights = 'pretrained/r101.pth'
        args.masks = True

        utils.init_distributed_mode(args)

        # fix the seed for reproducibility
        seed = args.seed + utils.get_rank()
        torch.manual_seed(seed)
        random.seed(seed)

        return build_model(args)
    
    else:
#         print(ValueError('Please provide model name.'))
        sys.exit('Please provide a valid model name.')
    return model

