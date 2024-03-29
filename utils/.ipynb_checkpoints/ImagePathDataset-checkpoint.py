#!/usr/bin/env python
# coding: utf-8

# In[ ]:
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from models.VisTR.datasets import transforms as DT
from torchvision.ops import masks_to_boxes
from utils.utils import get_label

class ImagePathDataset(Dataset):
    def __init__(self, image_path, mask_path, transform=None, aug = False):
        self.image_path = image_path
        self.mask_path = mask_path
        self.transform = transform
        self.aug = aug

    def __len__(self):
        if self.aug: return len(self.image_path)*2
        return len(self.image_path)

    def __getitem__(self, idx):

        index = idx//2 if self.aug else idx

        image = Image.open(self.image_path[index])
        mask = T.ToTensor()(Image.open(self.mask_path[index]))

        if self.transform:
            image = self.transform(image)

        if self.aug:
            if idx%2 == 1:
                image = torch.flip(image, dims = [2])
                mask = torch.flip(mask, dims = [2])

        return image, mask

   
class ImagePathDataset_siam(Dataset):
    def __init__(self, curr_frame_path, prev_frame_path,
                 curr_mask_path, prev_mask_path, 
                 transform=None, aug=False, testing = False):
        self.curr_frame_path = curr_frame_path
        self.prev_frame_path = prev_frame_path
        self.curr_mask_path = curr_mask_path
        self.prev_mask_path = prev_mask_path
        self.transform = transform
        self.aug = aug
        self.testing = testing
        

    def __len__(self):
        if self.aug: return len(self.curr_frame_path)*2
        return len(self.curr_frame_path)

    def __getitem__(self, idx):
        
        index = idx//2 if self.aug else idx

        curr_frame = Image.open(self.curr_frame_path[index])
        prev_frame = Image.open(self.prev_frame_path[index])
        curr_mask = T.ToTensor()(Image.open(self.curr_mask_path[index]))
        prev_mask = T.ToTensor()(Image.open(self.prev_mask_path[index]))
            
        if self.transform:
            curr_frame = self.transform(curr_frame)
            prev_frame = self.transform(prev_frame)
            
        if self.aug:
            if idx%2 == 1:
                curr_frame = torch.flip(curr_frame, dims = [2])
                prev_frame = torch.flip(prev_frame, dims = [2])
                curr_mask = torch.flip(curr_mask, dims = [2])
                prev_mask = torch.flip(prev_mask, dims = [2])
                
        xmin, ymin, xmax, ymax = get_label(prev_mask)
        if self.testing: return curr_frame, prev_frame[:,ymin:ymax, xmin:xmax], curr_mask, prev_mask
        return curr_frame, prev_frame[:,ymin:ymax, xmin:xmax], curr_mask
        
class ImagePathDataset_lstm(Dataset):
    def __init__(self, image_path, mask_path, num_frames, transform=None, aug=False):
        self.image_path = image_path
        self.mask_path = mask_path
        self.num_frames = num_frames
        self.transform = transform
        self.aug = aug

    def __len__(self):
        if self.aug: return len(self.image_path)*2
        return len(self.image_path)

    def __getitem__(self, idx):
        index = idx//2 if self.aug else idx
        
        image = [self.transform(Image.open(self.image_path[index][i]))
                          for i in range(self.num_frames)]
        image = torch.stack(image)

        mask = T.ToTensor()(Image.open(self.mask_path[index]))

        if self.aug:
            if idx%2 == 1:
                image = torch.flip(image, dims = [3])
                mask = torch.flip(mask, dims = [2])

        return image, mask
    
def make_transform(image_set):
    normalize = DT.Compose([
        DT.ToTensor(),
        DT.Normalize([0.2316], [0.2038]) #mean #standard deviation
    ])
    if image_set == 'train':
        return DT.Compose([
            # DT.RandomHorizontalFlip(),
            normalize,
        ])

    if image_set == 'val':
        return DT.Compose([normalize])

class ImagePathDataset_vistr(Dataset):
    def __init__(self, image_path, mask_path, num_frames, transform=None, aug=False):
        self.image_path = image_path
        self.mask_path = mask_path
        self.num_frames = num_frames
        self.transform = transform
        self.aug = aug
        

    def __len__(self):
        if self.aug: return len(self.image_path)*2
        return len(self.image_path)
    

    def __getitem__(self, idx):
        
        index = idx//2 if self.aug else idx
        
        images = [Image.open(self.image_path[index][i]) for i in range(self.num_frames)]

        masks = [T.ToTensor()(Image.open(self.mask_path[index][i]))
                for i in range(self.num_frames)]

        if self.aug:
            if idx%2 == 1:
                # Horizontal flip
                images = [image.transpose(Image.FLIP_LEFT_RIGHT) for image in images]
                masks = [torch.flip(mask, dims = [2]) for mask in masks]
                
        targets = {}
        targets['labels'] = torch.ones(self.num_frames).long()
        targets['valid'] = torch.ones(self.num_frames).long()
        targets['masks'] = torch.cat(masks, dim=0)
        targets['boxes'] = self.get_bbox(masks)
        
        if self.transform is not None:
            images, targets = self.transform(images, targets)
        
        images = [image.repeat(3,1,1) for image in images]
            
        return torch.cat(images,dim=0), targets
  
    def get_bbox(self, mask_list):
        return torch.cat([masks_to_boxes(mask) for mask in mask_list], dim=0) 

