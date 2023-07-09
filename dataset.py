#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import glob
import os
import re
import torch
from torch.utils.data import random_split
import torchvision.transforms as T
from utils.ImagePathDataset import *
from models.VisTR.util import misc as utils


pat=re.compile("(\d+)\D*$")

def key_func(x):
    mat=pat.search(os.path.split(x)[-1]) # match last group of digits
    if mat is None:
        return x
    return "{:>10}".format(mat.group(1)) # right align to 10 digits


def get_dataset(args):
    
    file_dir = glob.glob(args.data_path+'*')
    
    # 3. Create data loaders
    if args.test_batch_size:
        test_kwargs = {'batch_size': args.test_batch_size}
    else:
        train_kwargs = {'batch_size': args.batch_size}
        val_kwargs = {'batch_size': args.val_batch_size}

    use_cuda = torch.cuda.is_available()
    
    # Define any image preprocessing steps
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.2316], [0.2038]), #mean #standard deviation
    ])
        
    if use_cuda:
        cuda_kwargs = {'num_workers': os.cpu_count(),
                       'pin_memory': True}
        
        if args.test_batch_size:
            test_kwargs.update(cuda_kwargs)
            
        else:
            train_kwargs.update(cuda_kwargs)
            val_kwargs.update(cuda_kwargs)

    if args.model_name == 'unet' or args.model_name == 'unetpp' or args.model_name == 'attn_unet' or args.model_name == 'trans_unet':
        image_list = []
        mask_list = []

        for path in file_dir:
            frames = sorted(glob.glob(path+'/images/*.jpg'), key=key_func)
            masks = sorted(glob.glob(path+'/masks/*.png'), key=key_func)
            
            clip_len = 40 if args.wrist else len(frames)
            for i in range(clip_len):
                image_list.append(frames[i])
                mask_list.append(masks[i])
        
        # 2. Split into train / validation partitions
        if args.test_batch_size:
            test_dataset = ImagePathDataset(image_list, mask_list,transform=transform)
            args.n_test = len(test_dataset)
            
        else:
            val_percent = args.val_percent
            n_val = int(len(image_list) * val_percent)
            n_train = len(image_list) - n_val

            train_image_list, val_image_list = random_split(image_list, [n_train, n_val], generator=torch.Generator().manual_seed(0))
            train_mask_list, val_mask_list = random_split(mask_list, [n_train, n_val], generator=torch.Generator().manual_seed(0))

            train_dataset = ImagePathDataset(train_image_list, train_mask_list,
                                             transform=transform,
                                             aug=args.no_aug)
            val_dataset = ImagePathDataset(val_image_list, val_mask_list,
                                            transform=transform)

            args.n_train = len(train_dataset)
            args.n_val = len(val_dataset)

    elif args.model_name == 'siam_unet':
        curr_image_list = []
        prev_image_list = []
        curr_mask_list = []
        prev_mask_list = []

        for path in file_dir:
            frames = sorted(glob.glob(path+'/images/*.jpg'), key=key_func)
            masks = sorted(glob.glob(path+'/masks/*.png'), key=key_func)

            clip_len = 40 if args.wrist else len(frames)
            for i in range(1,clip_len):
                prev_image_list.append(frames[i-1])
                curr_image_list.append(frames[i])
                prev_mask_list.append(masks[i-1])
                curr_mask_list.append(masks[i])
                
        if args.test_batch_size:
            test_dataset = ImagePathDataset_siam(curr_image_list,
                                        prev_image_list,
                                        curr_mask_list,
                                        prev_mask_list,
                                        transform=transform)
            args.n_test = len(test_dataset)
            
        else:
            val_percent = args.val_percent
            n_val = int(len(prev_image_list) * val_percent)
            n_train = len(prev_image_list) - n_val

            train_prev_image_list, val_prev_image_list = random_split(prev_image_list, [n_train, n_val], generator=torch.Generator().manual_seed(0))
            train_curr_image_list, val_curr_image_list = random_split(curr_image_list, [n_train, n_val], generator=torch.Generator().manual_seed(0))
            train_prev_mask_list, val_prev_mask_list = random_split(prev_mask_list, [n_train, n_val], generator=torch.Generator().manual_seed(0))
            train_curr_mask_list, val_curr_mask_list = random_split(curr_mask_list, [n_train, n_val], generator=torch.Generator().manual_seed(0))

            # Create the dataset
            train_dataset = ImagePathDataset_siam(train_curr_image_list,
                                                train_prev_image_list,
                                                train_curr_mask_list,
                                                train_prev_mask_list,
                                                transform=transform,
                                                aug=args.no_aug)

            val_dataset = ImagePathDataset_siam(val_curr_image_list,
                                                val_prev_image_list,
                                                val_curr_mask_list,
                                                val_prev_mask_list,
                                                transform=transform)

            args.n_train = len(train_dataset)
            args.n_val = len(val_dataset)
        
    elif args.model_name == 'lstm_unet':
        num_frames = args.num_frames

        image_list = []
        mask_list = []

        for path in file_dir:
            frames = sorted(glob.glob(path+'/images/*.jpg'), key=key_func)
            masks = sorted(glob.glob(path+'/masks/*.png'), key=key_func)

        #     for i in range(35):
            clip_len = 40 if args.wrist else len(frames)
            for i in range(clip_len - num_frames + 1):
                image_list.append(frames[i:i + num_frames])
                mask_list.append(masks[i + num_frames - 1])

        if args.test_batch_size:
            test_dataset = ImagePathDataset_lstm(image_list,
                                                mask_list,
                                                num_frames = args.num_frames,
                                                transform=transform)
            args.n_test = len(test_dataset)
            
        else:
            val_percent = args.val_percent
            n_val = int(len(image_list) * val_percent)
            n_train = len(image_list) - n_val

            train_image_list, val_image_list = random_split(image_list, [n_train, n_val], generator=torch.Generator().manual_seed(0))
            train_mask_list, val_mask_list = random_split(mask_list, [n_train, n_val], generator=torch.Generator().manual_seed(0))

            train_dataset = ImagePathDataset_lstm(train_image_list,
                                                  train_mask_list,
                                                  num_frames = args.num_frames,
                                                  transform=transform,
                                                  aug=args.no_aug)
            val_dataset = ImagePathDataset_lstm(val_image_list,
                                                val_mask_list,
                                                num_frames = args.num_frames,
                                                transform=transform)
            args.n_train = len(train_dataset)
            args.n_val = len(val_dataset)

    elif args.model_name == 'vistr':
        
        args.batch_size = 1
        num_frames = args.num_frames
        
        image_list = []
        mask_list = []

        for path in file_dir:
            frames = sorted(glob.glob(path+'/images/*.jpg'), key=key_func)
            masks = sorted(glob.glob(path+'/masks/*.png'), key=key_func)

        #     for i in range(35):
            clip_len = 40 if args.wrist else len(frames)
            for i in range(clip_len-num_frames+1):
                image_list.append(frames[i:i+num_frames])
                mask_list.append(masks[i:i+num_frames])
        
        if args.test_batch_size:
            test_dataset = ImagePathDataset_vistr(image_list, mask_list,
                                         num_frames, transform=make_transform(image_set='val'))
            args.n_test = len(test_dataset)
            
        else:
            val_percent = args.val_percent
            n_val = int(len(image_list) * val_percent)
            n_train = len(image_list) - n_val

            train_image_list, val_image_list = random_split(image_list, [n_train, n_val], generator=torch.Generator().manual_seed(0))
            train_mask_list, val_mask_list = random_split(mask_list, [n_train, n_val], generator=torch.Generator().manual_seed(0))

                # no validation ground truth for ytvos dataset

            train_dataset = ImagePathDataset_vistr(train_image_list, train_mask_list,
                                         num_frames, transform=make_transform(image_set='train'))

            sampler_train = torch.utils.data.RandomSampler(train_dataset)

            batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)

            data_loader_train = DataLoader(train_dataset, batch_sampler=batch_sampler_train,
                                       collate_fn=utils.collate_fn, num_workers=args.num_workers)

            args.n_train = len(train_dataset)
    #         args.n_val = len(val_dataset)

            return data_loader_train, None
    
    if args.test_batch_size:
        return DataLoader(test_dataset, shuffle=False, **test_kwargs)
        
    else:
        train_loader = DataLoader(train_dataset, shuffle=True, **train_kwargs)
        val_loader = DataLoader(val_dataset, shuffle=False, **val_kwargs)

        return train_loader, val_loader

