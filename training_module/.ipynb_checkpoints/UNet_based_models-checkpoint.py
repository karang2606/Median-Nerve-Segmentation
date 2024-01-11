#!/usr/bin/env python
# coding: utf-8

# In[ ]:
from pathlib import Path
import glob
import time
import numpy as np
from tqdm import tqdm
import torch
from models.models import get_model
from torch import optim
import torch.nn as nn
from utils.dice_metric import dice_coeff, LogCoshDiceLoss, DiceLoss
from dataset import key_func
from utils.utils import save_loss_and_performance_plot

def train_UNet(args, train_loader, val_loader):
    
    device = torch.device(args.device)

    model = get_model(args)
    
    #Print model parameters
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'Model Parameters: {round(params/1e6,3)}M')
    
    model.to(device)

    dir_checkpoint = Path(args.output_dir)
    Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
    
    checkpoint_path = sorted(glob.glob(str(dir_checkpoint/'*.pth')), key = key_func)
    
    start_epoch = 1
    if args.resume and checkpoint_path:
        args_load = checkpoint_path[-1]
        state_dict = torch.load(args_load, map_location=device)
        model.load_state_dict(state_dict)
        print(f'Model loaded from {args_load}')
        
        start_epoch = len(checkpoint_path)*args.save_intervals + 1
        print(f'Starting training from {start_epoch}th epoch')
        
    num_epochs = args.epochs
    # Define loss function
    loss_list = {
        'bce': nn.BCEWithLogitsLoss(),
        'dice': DiceLoss(),
        'logcosh': LogCoshDiceLoss(),
    }

    print('Using')
    if "bce" in args.loss: print("Binary Cross Entropy Loss")
    if "dice" in args.loss: print("Dice Loss")
    if "logcosh" in args.loss: print("LogCosh(Dice Loss)")

    # if "bce" in args.loss: print("Binary Cross Entropy Loss", end=" ")
    # if "dice" in args.loss: print("Dice Loss", end=" ")
    # if "logcosh" in args.loss: print("LogCosh(Dice Loss)")
    # Define optimizer
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5) # Maximize the Dice score
    # exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer,
    #                                          step_size=5, gamma=0.1)
    prev_val_dice = 0

    args.train_loss = []
    args.val_loss = []
    args.train_dice = []
    args.val_dice = []
    args.learning_rate = []

    for epoch in range(start_epoch, num_epochs + 1):

        model.train()  # Set model to training mode
    
        train_loss = 0.0
        total_samples = 0
        train_dice = 0.0

        start = time.time()

        with tqdm(total=args.n_train, desc=f'Epoch [{epoch}/{num_epochs}], Training', 
                  unit='batch', leave=False) as pbar:
            
            for batch in train_loader:
                
                optimizer.zero_grad()

                if len(batch) == 2:
                    images, true_masks = batch
                    images = images.to(device)
                    true_masks = true_masks.to(device)
                    
                    pred_masks = model(images)

                elif len(batch) == 3:
                    images, prev_images, true_masks = batch
                    images = images.to(device)
                    true_masks = true_masks.to(device)

                    prev_images = prev_images.to(device=device)
                    pred_masks = model(images, prev_images)

#                 Calculate loss

                loss = 0

                for loss_name in args.loss:
                    loss += loss_list[loss_name](pred_masks.float(), true_masks.float())
 
                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                running_mean_loss = loss.item()
                train_loss += running_mean_loss * images.size(0)

                running_mean_dice = dice_coeff(pred_masks, true_masks).item()
                train_dice += running_mean_dice * images.size(0)
                
                total_samples += images.size(0)

                pbar.update(images.shape[0])
                pbar.set_postfix(**{'loss (batch)': running_mean_loss,
                                   'DSC' : running_mean_dice})

        pbar.close()
        
        end = time.time()

        epoch_loss = train_loss / total_samples
        epoch_dice = train_dice / total_samples

        args.train_loss.append(epoch_loss)
        args.train_dice.append(epoch_dice)

        print(f"\nEpoch [{epoch}/{num_epochs}]")
        print(f"\t Training Loss: {epoch_loss:.4f}, Dice Score: {epoch_dice:.4f}, Time: {(end-start)/60:.2f} mins")

        # Validation
        model.eval()  # Set model to evaluation mode

        val_loss = 0.0
        val_samples = 0
        val_dice = 0.0

        start = time.time()
        with torch.no_grad():
            with tqdm(total=args.n_val, desc=f'Epoch [{epoch}/{num_epochs}], Validation',
                      unit='img', leave=False) as pbar:

                for val_batch in val_loader:

                    if len(val_batch) == 2:
                        val_images, val_masks = val_batch

                        val_images = val_images.to(device)
                        val_masks = val_masks.to(device)

                        val_outputs = model(val_images)

                    elif len(val_batch) == 3:
                        val_images, val_prev_images, val_masks = val_batch

                        val_images = val_images.to(device)
                        val_masks = val_masks.to(device)
                        val_prev_images = val_prev_images.to(device=device)

                        val_outputs = model(val_images, val_prev_images)

#                         temp_val_loss = criterion(val_outputs, val_masks).item() * val_images.size(0)
                    val_outputs = (val_outputs > 0.5).float()
#                         temp_val_loss += dice_loss(val_outputs, val_masks.float(), multiclass=False)

                    running_mean_val_loss = 0
                    for loss_name in args.loss:
                        running_mean_val_loss += loss_list[loss_name](pred_masks.float(), true_masks.float()).item()

                    val_loss += running_mean_val_loss * val_images.size(0)

                    running_mean_val_dice = dice_coeff(val_outputs, val_masks).item() 
                    val_dice += running_mean_val_dice * val_images.size(0)
                   
                    val_samples += val_images.size(0)
                    
                    pbar.update(val_images.shape[0])
                    pbar.set_postfix(**{'loss (batch)': running_mean_val_loss,
                                        'DSC' : running_mean_val_dice})
                    
            pbar.close()

        end = time.time()

        val_epoch_loss = val_loss / val_samples
        val_epoch_dice = val_dice / val_samples

        args.val_loss.append(val_epoch_loss)
        args.val_dice.append(val_epoch_dice)

        before_lr = optimizer.param_groups[0]["lr"]
        args.learning_rate.append(before_lr)

        scheduler.step(val_epoch_dice)
        # exp_lr_scheduler.step()
        after_lr = optimizer.param_groups[0]["lr"]
        
        print(f"\t Validation Loss: {val_epoch_loss:.4f}, Dice Score: {val_epoch_dice:.4f}, Time: {(end-start)/60:.2f} mins")
        print("\t Learning Rate: %.6f -> %.6f" % (before_lr, after_lr))
        
        #Save Plot
        save_loss_and_performance_plot(args)

        #Save checkpoint
        if val_epoch_dice > prev_val_dice:
            state_dict = model.state_dict()
            torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch.pth'))
            print(f'Checkpoint {epoch} saved!')
            prev_val_dice = val_epoch_dice

    #         if epoch % args.save_intervals == 0:
    #             state_dict = model.state_dict()
    #             torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
    #             torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
    #             print(f'Checkpoint {epoch} saved!')

    print("Training finished.")