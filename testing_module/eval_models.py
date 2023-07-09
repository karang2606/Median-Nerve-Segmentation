#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from PIL import Image
import math
from tqdm import tqdm
import torch
from utils.utils import get_metric, get_label
import torch.nn.functional as F


# In[2]:


def prediction(model, image, true_mask, device, out_threshold = 0.5):
    
    model.eval()
    
    with torch.no_grad():

        image = image.to(device=device, dtype=torch.float32)
        true_mask = true_mask.to(device=device, dtype=torch.long)
#         model.start_flops_count()
        output = model(image).to(device)
#         AVG_flops, params_count = model.compute_average_flops_cost()
#         print('Average flops',flops_to_string(AVG_flops))
#         print('Parameters',params_to_string(params_count))
#         model.stop_flops_count()
        pred_mask = (output > out_threshold).float()
        res = get_metric(pred_mask, true_mask)
        
        pred_mask = pred_mask.cpu().detach().numpy()
        
    return pred_mask, res


# In[3]:


def prediction_siam(model, cur_frame, prev_frame, cur_mask, 
                    prev_mask, device='cpu', out_threshold = 0.5):

    model.eval()
    with torch.no_grad():

        cur_frame = cur_frame.unsqueeze(0).to(device=device,
                                                 dtype=torch.float32, 
                                                 memory_format=torch.channels_last)

        prev_frame = prev_frame.unsqueeze(0).to(device=device,
                                                 dtype=torch.float32, 
                                                 memory_format=torch.channels_last)
        cur_mask = cur_mask.to(device=device, dtype=torch.long)


        xmin, ymin, xmax, ymax = get_label(prev_mask)        
#         model.start_flops_count()
        output = model(cur_frame, prev_frame[...,ymin:ymax, xmin:xmax]).to(device)
#         AVG_flops, params_count = model.compute_average_flops_cost()
#         print('Average flops',flops_to_string(AVG_flops))
#         print('Parameters',params_to_string(params_count))
#         model.stop_flops_count()
        pred_mask = (output > out_threshold).float()
        
        res = get_metric(pred_mask, cur_mask)
        
        pred_mask = pred_mask.cpu().detach().numpy()
        
    return pred_mask, res


# In[4]:


def eval_Siam(args, model, num, frames, true_masks):
    
    device = args.device
    prev_mask = true_masks[0]
    args.pred_masks_seq.append(np.array(Image.open(args.test_msk_path[num][0])))
    for i in tqdm(range(1, args.clip_len), desc='Testing'):
        prev_frame = frames[i-1]
        cur_frame = frames[i]
        cur_mask = true_masks[i]

        pred_mask, res = prediction_siam(model, cur_frame, prev_frame, cur_mask,
                   prev_mask, device, out_threshold = 0.5)
        args.pred_masks_seq.append(pred_mask)

        args.score += res
        _,_,_, curr_ds = res

        if curr_ds > 0.5: prev_mask = pred_masks[0]


# In[5]:


def eval_LSTM(args, model, frames, true_masks):
    
    device = args.device
    n_frames = args.num_frames
    for i in tqdm(range(1, args.clip_len), desc='Testing'):
        if i<n_frames-1:
            img_ = [frames[0] for _ in range(n_frames-i-1)]
            img_ += [frames[j] for j in range(i+1)]
            img_ = torch.stack(img_)

        else:
            img_ = frames[i-n_frames+1:i+1]

        msk_ = true_masks[i]

        pred_mask, res = prediction(model, img_.unsqueeze(0), msk_, device,
                  out_threshold=0.5)
        args.pred_masks_seq.append(pred_mask)
        args.score += res


# In[6]:


def eval_VisTR(args, model, num, frames, true_masks):
    
    device = args.device
    n_frames = args.num_frames

    pred_score_2 = []

    im = Image.open(args.test_img_path[num][0])

    for i in tqdm(range(0, args.clip_len ,n_frames), desc = 'Testing', leave = False):
        start = i
        end = min(i+n_frames, args.clip_len)

        image = frames[start:end]
        if image[0].size()[0] == 1:
            image = [img.repeat(3,1,1).unsqueeze(0).to(device) for img in image]
        else:
            image = [img.unsqueeze(0).to(device) for img in image]
        image = torch.cat(image,dim=0)

        input_len = end - start
        if input_len < n_frames:
            image = torch.cat([image for _ in range(math.ceil(n_frames/input_len))],dim=0)
            image = image[:n_frames]
    #         model.start_flops_count()
        outputs = model(image)
    #         AVG_flops, params_count = model.compute_average_flops_cost()
    #         print('Average flops',flops_to_string(AVG_flops))
    #         print('Parameters',params_to_string(params_count))
    #         model.stop_flops_count()
        # end of model inference
        logits, boxes, masks = outputs['pred_logits'].softmax(-1)[0,:,:-1], outputs['pred_boxes'][0], outputs['pred_masks'][0]
        pred_masks = F.interpolate(masks.reshape(n_frames,args.num_ins,masks.shape[-2],masks.shape[-1]),(im.size[1],im.size[0]),mode="bilinear").sigmoid().cpu().detach().numpy()>0.5
        pred_logits = logits.reshape(n_frames,args.num_ins,logits.shape[-1]).cpu().detach().numpy()
        pred_masks = pred_masks[:input_len]
        pred_logits = pred_logits[:input_len]
        pred_scores = np.max(pred_logits,axis=-1)
        pred_logits = np.argmax(pred_logits,axis=-1)
        temp = []
        for m in range(args.num_ins):
            if pred_masks[:,m].max()==0:
                continue
            score = pred_scores[:,m].mean()
            #category_id = pred_logits[:,m][pred_scores[:,m].argmax()]
            category_id = np.argmax(np.bincount(pred_logits[:,m]))
            instance = {'score':float(score), 'category_id':int(category_id)}
            temp.append(instance)
        pred_score_2.append(temp)
        args.pred_masks_seq.append(pred_masks)
    args.pred_masks_seq = [img[0] for batch in args.pred_masks_seq for img in batch]

    args.score += get_metric(args.pred_masks_seq, true_masks.numpy()) * args.clip_len

    
def eval_UNet(args, model, frames, true_masks):
    
    device = args.device
    for i in tqdm(range(0, args.clip_len, args.test_batch_size), desc='Testing'):
        start = i
        end = min(args.clip_len, i + args.test_batch_size)

        images = frames[start:end]
        masks = true_masks[start:end]

        pred_mask, res = prediction(model, images, masks, device)
        args.pred_masks_seq += [msk[0] for msk in pred_mask]

        args.score += res*len(images)

