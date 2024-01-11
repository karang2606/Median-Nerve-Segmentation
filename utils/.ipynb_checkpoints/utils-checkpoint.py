import glob
import re
import logging
from tqdm import tqdm
import numpy as np
from PIL import Image
import os
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchsummary import summary
from utils.dice_metric import dice_loss, dice_coeff
from utils.hausdorff import hausdorff_distance_mask
from torchvision import transforms
import torchvision.transforms as T
import torch.distributions as dist
from torchvision.ops import masks_to_boxes
import time
from mmcv.cnn.utils.flops_counter import add_flops_counting_methods, flops_to_string, params_to_string
from models.VisTR.datasets import transforms as DT

pat=re.compile("(\d+)\D*$")

def key_func(x):
    mat=pat.search(os.path.split(x)[-1]) # match last group of digits
    if mat is None:
        return x
    return "{:>10}".format(mat.group(1)) # right align to 10 digits

# def get_metric(pred, truth):
    
#     if torch.is_tensor(pred): pred = pred.cpu().detach().numpy()
#     if torch.is_tensor(truth): truth = truth.cpu().detach().numpy()
    
#     # Sensitivity == Recall
#     SE = PC = F1 = DC = HD = 0
#     for i in range(len(pred)):
        
#         # SR : Segmentation Result
#         # GT : Ground Truth
        
#         SR, GT = pred[i], truth[i]
        
#         # TP : True Positive
#         TP = ((SR==1)&(GT==1)).sum().item()

#         # FN : False Negative
#         FN = ((SR==0)&(GT==1)).sum().item()

#         # FP : False Positive
#         FP = ((SR==1)&(GT==0)).sum().item()

#         Inter = TP
#         Union = SR.sum().item() + GT.sum().item()
#         SE_ = float(TP)/(float(TP+FN) + 1e-6) #Recall
#         SE += SE_
#         PC_ = float(TP)/(float(TP+FP) + 1e-6) #Precision
#         PC +=PC_
#         F1 += 2*SE_*PC_/(SE_+PC_ + 1e-6) #F1 Score
#         DC += float(2*Inter)/(float(Union) + 1e-6) #Dice Score

#         HD += hausdorff_distance_mask(SR.squeeze(), 
#                                       GT.squeeze(), 
#                                       method = 'standard')
    
#     # return np.array([SE, PC, F1, DC])/len(pred)
#     return np.array([SE, PC, F1, DC, HD])/len(pred)

def get_metric(pred, truth):
    
    if torch.is_tensor(pred): pred = pred.cpu().detach().numpy()
    if torch.is_tensor(truth): truth = truth.cpu().detach().numpy()
    
    # Sensitivity == Recall
    SE = PC = F1 = DC = HD = 0
    valid_frame_count = 0
    
    for i in range(len(pred)):
        
        # SR : Segmentation Result
        # GT : Ground Truth
        
        SR, GT = pred[i], truth[i]
        
        # TP : True Positive
        TP = ((SR==1)&(GT==1)).sum().item()

        # FN : False Negative
        FN = ((SR==0)&(GT==1)).sum().item()

        # FP : False Positive
        FP = ((SR==1)&(GT==0)).sum().item()

        Inter = TP
        Union = SR.sum().item() + GT.sum().item()
        SE_ = float(TP)/(float(TP+FN) + 1e-6) #Recall
        SE += SE_
        PC_ = float(TP)/(float(TP+FP) + 1e-6) #Precision
        PC +=PC_
        F1 += 2*SE_*PC_/(SE_+PC_ + 1e-6) #F1 Score
        DC += float(2*Inter)/(float(Union) + 1e-6) #Dice Score

        HD_ = hausdorff_distance_mask(SR.squeeze(), 
                                      GT.squeeze(), 
                                      method = 'standard')
        if HD_ != np.inf: HD += HD_ ; valid_frame_count += 1
    
    # return np.array([SE, PC, F1, DC])/len(pred)
    if valid_frame_count == 0: return np.array([SE, PC, F1, DC, np.inf])/len(pred)
        
    return np.array([SE, PC, F1, DC, HD*(len(pred)/valid_frame_count)])/len(pred)
    
def get_label(mask):
    boxes = masks_to_boxes(mask)
    xmin, ymin, xmax, ymax = boxes.cpu().numpy().astype('int16')[0]

    xcen = (xmax+xmin)//2
    ycen = (ymax+ymin)//2

    out_width, out_height = (144,80)

    xmin, ymin, xmax, ymax = xcen - out_width//2 , ycen - out_height//2, xcen + out_width//2 , ycen + out_height//2

    # 448, 336
    if xmin<0: xmin = 0; xmax = out_width
    elif xmax>336: xmin = 336 - out_width; xmax = 336

    if ymin<0: ymin = 0; ymax = out_height
    elif ymax>448: ymin = 448 - out_height; ymax = 448

    return xmin, ymin, xmax, ymax


def get_bbox_for_filtered_mask(gray):
    
    bbox=[]
    if not isinstance(gray, np.ndarray): gray = np.array(gray)
    if gray.dtype!='uint8': gray= (gray*255).astype(np.uint8)
        
#     img = cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)
    thresh = cv2.threshold(gray,128,255,cv2.THRESH_BINARY)[1]

    # get contours
#     result = img.copy()
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    for cntr in contours:
        x,y,w,h = cv2.boundingRect(cntr)
        bbox.append([x,y,w,h])
#         cv2.rectangle(result, (x, y), (x+w, y+h), (0, 0, 255), 2)
#         print("x,y,w,h:",x,y,w,h)
    return bbox

def get_filtered_mask(mask_clip):
    bbox =[None]*len(mask_clip)
    cntr = [None]*len(mask_clip)
    area = [None]*len(mask_clip)
    filtered_mask = []
    for i in range(len(mask_clip)):
        frame = mask_clip[i].astype('int')
        temp = np.zeros_like(frame)

        blist = get_bbox_for_filtered_mask(frame)

        if len(blist) == 0:
            temp = filtered_mask[-1]
        elif len(blist) == 1:
            x,y,w,h = blist[0]
            cntr[i] = (x+w/2, y+h/2)
            bbox[i] = (x,y,w,h)
    #         area[i] = frame[x:x+w, y:y+h]
            temp = frame
        elif len(blist) > 1:
            all_cntrs=[(x+w/2, y+h/2) for x,y,w,h in blist]
            area = [w*h for x,y,w,h in blist]
            if cntr[i-1] == None:
                index = np.argmax(area)
                cntr[i] = all_cntrs[index]
                bbox[i] = blist[index]
                x,y,w,h = blist[index]
                temp[y:y+h ,x:x+w] = frame[y:y+h ,x:x+w]
            else:
                x0, y0 = cntr[i-1]
                index = np.argmin([np.sqrt((x0-x)**2 + (y0-y)**2) for x,y in all_cntrs])
                cntr[i] = all_cntrs[index]
                bbox[i] = blist[index]
                x,y,w,h = blist[index]
                temp[y:y+h ,x:x+w] = frame[y:y+h ,x:x+w]
        filtered_mask.append(temp)
        
    return filtered_mask


def DrawContours(img, msk, type_ = None):
    if not isinstance(img, np.ndarray): img = np.array(img)
    if not isinstance(msk, np.ndarray): msk = np.array(msk)
    if img.dtype!='uint8': img= (img*255).astype(np.uint8)
    if msk.dtype!='uint8': msk= (msk*255).astype(np.uint8)
    if(len(img.shape)<3): img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    
    cnts = cv2.findContours(msk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    if type_ == 'GT':
        for c in cnts:
            cv2.drawContours(img, [c], -1, (0, 255, 0), thickness=2)
    if type_ == 'Pred':
        for c in cnts:
            cv2.drawContours(img, [c], -1, (0, 0, 255), thickness=2)
    return img


def put_text(frame, name):
    cv2.putText(frame,
            name, (0, 440),
            0,
            1,
            color =(255, 255, 255),
            thickness=2,
            lineType=cv2.LINE_AA)


def get_frames(path):
    # Read the video from specified path
    frame_list = []
    cam = cv2.VideoCapture(path)
    nw = []
    while(True):
        # reading from frame
        ret,frame = cam.read()
        if ret:
            frame_list.append(frame)
        else:
            break
    return frame_list

def create_clip(args):
    
    clip = args.sub_frames
    print(args.no_GT)
    print(len(clip), len(args.pred_masks_seq))

    frameSize = Image.open(clip[0]).size # width x height
    args.frameRate = 5
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(f'./segmented_clips/{args.sub_name}_{args.model_name}.avi', fourcc, args.frameRate, frameSize)

    for i in range(args.clip_len):

        img = Image.open(clip[i]) #frame

        if args.no_GT:
            msk = np.array(Image.open(args.sub_masks[i])) #ground Truth
            img = DrawContours(img, msk, 'GT') #Green
        
        if len(np.unique(args.pred_masks_seq[i])) == 2:
            out_frame = DrawContours(img, args.pred_masks_seq[i], 'Pred') #Red
    #     out_frame = out_frame[:224]

        #  area mm2/ a pixel
        const = (30/448)**2
        area = np.sum(args.pred_masks_seq[i])*const

        put_text(out_frame, f'CSA={round(area,2)}')
        out.write(out_frame)

    out.release()

def load_model(model, args):
    if not args.load_from:
        args.load_from = sorted(glob.glob(args.output_dir + '/*.pth'), key=key_func)[-1]

    if args.model_name == 'vistr':
        state_dict = torch.load(args.load_from,  map_location='cpu')['model']
    else:
        state_dict = torch.load(args.load_from, map_location='cpu')
        
    model.load_state_dict(state_dict)
    print(f'Model loaded from {args.load_from}')
    return model

def save_loss_and_performance_plot(args):

    plot_name = ('_').join([args.model_name] + args.loss)

    epochs = range(len(args.train_loss))

    plt.figure(figsize=(25,6))

    plt.subplot(131)
    plt.plot(epochs, args.train_loss, label='Training')
    plt.plot(epochs, args.val_loss, label='Validation')
    plt.xlabel("Epochs", fontsize = 15)
    plt.ylabel("Loss", fontsize = 15)
    plt.legend()
    plt.grid()

    plt.subplot(132)
    plt.plot(epochs, args.train_dice, label='Training')
    plt.plot(epochs, args.val_dice, label='Validation')
    plt.xlabel("Epochs", fontsize = 15)
    plt.ylabel("Dice Score", fontsize = 15)
    plt.legend()
    plt.grid()

    plt.subplot(133)
    plt.plot(epochs, args.learning_rate)
    plt.xlabel("Epochs", fontsize = 15)
    plt.ylabel("Learning Rate", fontsize = 15)
    plt.grid()

    plt.suptitle(plot_name, fontsize=20)

    plt.savefig(plot_name + '.png', bbox_inches='tight')
    plt.close()