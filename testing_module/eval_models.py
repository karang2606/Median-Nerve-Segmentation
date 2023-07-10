import numpy as np
from PIL import Image
import math
from tqdm import tqdm
import torch
from utils.utils import get_metric, get_label
import torch.nn.functional as F
import torchvision.transforms as T


def prediction(model, image, device, out_threshold = 0.5):
    
    model.eval()
    
    with torch.no_grad():

        image = image.to(device=device, dtype=torch.float32)
#         model.start_flops_count()
        output = model(image).to(device)
#         AVG_flops, params_count = model.compute_average_flops_cost()
#         print('Average flops',flops_to_string(AVG_flops))
#         print('Parameters',params_to_string(params_count))
#         model.stop_flops_count()
        pred_mask = (output > out_threshold).float()
        
        pred_mask = pred_mask.cpu().detach().numpy()
        
    return pred_mask


def prediction_siam(model, cur_frame, prev_frame, 
                    prev_mask, device='cpu', out_threshold = 0.5):

    model.eval()
    with torch.no_grad():

        cur_frame = cur_frame.unsqueeze(0).to(device=device,
                                                 dtype=torch.float32, 
                                                 memory_format=torch.channels_last)

        prev_frame = prev_frame.unsqueeze(0).to(device=device,
                                                 dtype=torch.float32, 
                                                 memory_format=torch.channels_last)

        xmin, ymin, xmax, ymax = get_label(prev_mask)        
#         model.start_flops_count()
        output = model(cur_frame, prev_frame[...,ymin:ymax, xmin:xmax]).to(device)
#         AVG_flops, params_count = model.compute_average_flops_cost()
#         print('Average flops',flops_to_string(AVG_flops))
#         print('Parameters',params_to_string(params_count))
#         model.stop_flops_count()
        pred_mask = (output > out_threshold).float()
        
    return pred_mask.cpu().detach().numpy()


def eval_Siam(args, model, frames):
    
    device = args.device
    prev_mask = T.ToTensor()(Image.open(args.sub_masks[0]))
    args.pred_masks_seq.append(prev_mask.numpy()[0])
    for i in tqdm(range(1, args.clip_len), desc='Testing', leave = False):
        prev_frame = frames[i-1]
        cur_frame = frames[i]

        pred_mask = prediction_siam(model, cur_frame, prev_frame,
                   prev_mask, device, out_threshold = 0.5)
        args.pred_masks_seq.append(pred_mask[0,0])

        if len(np.unique(pred_mask)) == 2: prev_mask = torch.from_numpy(pred_mask[0])


def eval_LSTM(args, model, frames):
    
    device = args.device
    n_frames = args.num_frames
    for i in tqdm(range(args.clip_len), desc='Testing', leave = False):
        if i<n_frames-1:
            img_ = [frames[0] for _ in range(n_frames-i-1)]
            img_ += [frames[j] for j in range(i+1)]
            img_ = torch.stack(img_)

        else:
            img_ = frames[i-n_frames+1:i+1]

        pred_mask = prediction(model, img_.unsqueeze(0), device,
                  out_threshold=0.5)
        args.pred_masks_seq.append(pred_mask[0,0])


def eval_VisTR(args, model, frames):
    
    device = args.device
    n_frames = args.num_frames

    pred_score_2 = []

    im = Image.open(args.sub_frames[0])

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
    
def eval_UNet(args, model, frames):
    
    device = args.device
    for i in tqdm(range(0, args.clip_len, args.test_batch_size), desc='Testing', leave = False):
        start = i
        end = min(args.clip_len, i + args.test_batch_size)

        images = frames[start:end]

        pred_mask = prediction(model, images, device)
        args.pred_masks_seq += [msk[0] for msk in pred_mask]
