#!/usr/bin/env python
# coding: utf-8

# In[ ]:
from pathlib import Path
import numpy as np
from tqdm import tqdm
import torch
from models.models import get_model
from torch import optim
import torch.nn as nn
from utils.dice_metric import dice_loss, dice_coeff

def evaluate(args, model, dataloader, device, amp):
    model.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    epoch_loss = 0
    # iterate over the validation set
    criterion = nn.CrossEntropyLoss() if args.num_classes > 1 else nn.BCEWithLogitsLoss()
    
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            if len(batch) == 2:
                images, true_masks = batch
                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)
                pred_masks = model(images)

            elif len(batch) == 3:
                images, prev_images, true_masks = batch

                images = images.to(device=device, dtype=torch.float32)
                prev_images = prev_images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)
                pred_masks = model(images, prev_images)
            
            loss = criterion(pred_masks, true_masks.float())
            pred_masks = (torch.sigmoid(pred_masks) > 0.5).float()
            loss += dice_loss(pred_masks, true_masks.float(), multiclass=False)
            epoch_loss += loss.item()
            
            assert true_masks.min() >= 0 and true_masks.max() <= 1, 'True mask indices should be in [0, 1]'
            # compute the Dice score
            dice_score += dice_coeff(pred_masks, true_masks, reduce_batch_first=False)

    model.train()
    return epoch_loss, dice_score / max(num_val_batches, 1)


def train_UNet(args, train_loader, val_loader):
    
    device = torch.device(args.device)

    model = get_model(args)
    model.to(device)

    dir_checkpoint = Path(args.output_dir)
    amp = True
    gradient_clipping = 1.0

    optimizer = optim.RMSprop(model.parameters(),
                              lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss() if args.num_classes > 1 else nn.BCEWithLogitsLoss()
    global_step = 0

    train_loss = []
    train_dice_score = []
    val_loss = []
    val_dice_score = []

    for epoch in range(1, args.epochs + 1):

        model.train()
        epoch_loss = 0
        dice_score = 0
        epoch_val_loss = []
        epoch_val_dice_score = []
        with tqdm(total=args.n_train, desc=f'Epoch [{epoch}/{args.epochs}]', unit='img', position=0, leave=True) as pbar:

            for batch in train_loader:

                if len(batch) == 2:
                    images, true_masks = batch
                    images = images.to(device=device, dtype=torch.float32)
                    true_masks = true_masks.to(device=device, dtype=torch.long)
                    pred_masks = model(images)

                elif len(batch) == 3:
                    images, prev_images, true_masks = batch

                    images = images.to(device=device, dtype=torch.float32)
                    prev_images = prev_images.to(device=device, dtype=torch.float32)
                    true_masks = true_masks.to(device=device, dtype=torch.long)
                    pred_masks = model(images, prev_images)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):

                    loss = criterion(pred_masks, true_masks.float())
                    
                    pred_masks = (pred_masks > 0.5).float()

                    loss += dice_loss(pred_masks, true_masks.float(), multiclass=False)

                    temp = dice_coeff(pred_masks, true_masks, reduce_batch_first=False)
                    dice_score +=temp

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()

                pbar.set_postfix(**{'loss (batch)': loss.item(),
                                   'DSC' : temp.item()})

                # Evaluation round
                division_step = (args.n_train // (5 * args.batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:

                        val_step_loss, val_step_score = evaluate(args, model, val_loader, device, amp)
                        scheduler.step(val_step_loss)
    #                     print('Validation Dice score: {}'.format(val_score))
                        epoch_val_loss.append(val_step_loss)
                        epoch_val_dice_score.append(val_step_score.item())
    #                             print('Validation Dice score: {}'.format(val_step_score))

        train_loss.append(epoch_loss)
        train_dice_score.append((dice_score / len(train_loader)).item())

        val_loss.append(np.mean(epoch_val_loss))
        val_dice_score.append(np.mean(epoch_val_dice_score))

        print(f'Train Loss: {round(epoch_loss,4)}, Train DSC: {round(train_dice_score[-1],4)}')
        print(f'Val Loss: {round(np.mean(epoch_val_loss),4)}, Val DSC: {round(val_dice_score[-1],4)}')

        Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
        state_dict = model.state_dict()
        torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
        print(f'Checkpoint {epoch} saved!\n')

