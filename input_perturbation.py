import argparse
import glob
from dataset import *
from PIL import Image
from models.models import get_model
from utils.utils import *
from utils.perturbation import *
import torchvision.transforms as T
from tqdm.notebook import tqdm
from testing_module.eval_models import eval_UNet, eval_LSTM, eval_Siam, eval_VisTR
from pathlib import Path
from test import get_args_parser

def get_predictions(args, print_score = True):

    transform_list = [T.ToTensor(),
                  T.Normalize([0.2316], [0.2038]),
                 ]

    if args.perturb_input:
        transform_list.insert(1, AddPoissonNoise(intensity = args.perturb_input))
        print(f'\nPerturbed input by adding Poisson noise of mean {args.perturb_input}.\n')
    
    transform = T.Compose(transform_list)

    with torch.no_grad():
        
        if args.model_name != 'vistr':
                model = get_model(args)
        else:
            args.masks = True
            model, _, _ = get_model(args)
            
        model = load_model(model, args)

        if args.perturb_model:
            model = perturb_model(args.perturb_model, model)
            print(f'\nModel parameters perturbed by {args.perturb_model}. (i.e., W_new = {1+args.perturb_model}*W)')
            
        model.to(args.device)
        model.eval()
        
        avg_score = []
        args.test_file_dir = sorted(glob.glob(args.data_path+'*'))
        
        for path in args.test_file_dir:
        
            args.sub_name = path.split('/')[-1]
            args.sub_frames = sorted(glob.glob(path+'/images/*.jpg'), key=key_func)
            args.clip_len = 40 if args.wrist else len(args.sub_frames)
            args.pred_masks_seq = []
        
            frames = torch.stack([transform(Image.open(frame)) for frame in args.sub_frames])
        
            if args.model_name == 'siam_unet':
                args.sub_masks = sorted(glob.glob(path+'/masks/*.png'), key=key_func)
                
                eval_Siam(args, model, frames)
        
            elif args.model_name == 'lstm_unet':
                eval_LSTM(args, model, frames)
        
            elif  args.model_name == 'vistr':
                eval_VisTR(args, model, frames)
        
            else:
                eval_UNet(args, model, frames)
        
            if args.no_GT:
                args.sub_masks = sorted(glob.glob(path+'/masks/*.png'), key=key_func)
        
                true_masks = [np.array(Image.open(msk))/255 for msk in args.sub_masks]

                res_wrsit = np.round(get_metric(args.pred_masks_seq[:40], true_masks[:40]), 3)
                res_full = np.round(get_metric(args.pred_masks_seq, true_masks), 3)
                
                if print_score:
                    print(f'\n{args.sub_name}')
                    print(f'\tRecall: {res_wrsit[0]}, Precision: {res_wrsit[1]}, F1_Score: {res_wrsit[2]}, Dice_Score: {res_wrsit[3]}, Hausdorff Distance: {res_wrsit[4]}')
                    print(f'\tRecall: {res_full[0]}, Precision: {res_full[1]}, F1_Score: {res_full[2]}, Dice_Score: {res_full[3]}, Hausdorff Distance: {res_full[4]}')
        
            avg_score.append(np.concatenate((res_wrsit, res_full), axis=0))

        avg_score = np.round(np.mean(avg_score, axis=0),3)
        if print_score:
            print("\nAverage Metric")
            print(f'Recall: {avg_score[0]}, Precision: {avg_score[1]}, F1_Score: {avg_score[2]}, Dice_Score: {avg_score[3]}, Hausdorff Distance: {avg_score[4]}')
            print(f'Recall: {avg_score[5]}, Precision: {avg_score[6]}, F1_Score: {avg_score[7]}, Dice_Score: {avg_score[8]}, Hausdorff Distance: {avg_score[9]}')

        return avg_score


def main():
    
    parser = argparse.ArgumentParser('Model training and evaluation script',
                                 parents=[get_args_parser()])
    args = parser.parse_args()

    args.perturb_model = 0
    intensities = [0.10, 0.20, 0.03]
    # intensities = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    multiple_run = []
    
    for _ in range(10):
        print(_)
        res_on_int = []
        for intensity in intensities:
            
            args.perturb_input = intensity
            
            avg_score = get_predictions(args, print_score=False)
            
            res_on_int.append(avg_score)
    #         print(f'for intensity:{intensity} > {mean_metric}') #chckpnt 7 with first 1 sec
    
        multiple_run.append(res_on_int)
        
    multiple_run = np.array(multiple_run)
    
    t1 = np.round(np.mean(multiple_run, axis=0),3)
    t2 = np.round(np.std(multiple_run, axis=0),4)
    
    for i in range(len(t1)):
        print(f'{intensities[i]} & {t1[i,0]} $\\pm$ {t2[i,0]} & {t1[i,1]} $\\pm$ {t2[i,1]} & {t1[i,3]} $\\pm$ {t2[i,3]}& {t1[i,4]} $\\pm$ {t2[i,4]} \\\\')

    for i in range(len(t1)):
        print(f'{intensities[i]} & {t1[i,5]} $\\pm$ {t2[i,5]} & {t1[i,6]} $\\pm$ {t2[i,6]} & {t1[i,8]} $\\pm$ {t2[i,8]} & {t1[i,9]} $\\pm$ {t2[i,9]} \\\\')

if __name__ == '__main__':
    main()