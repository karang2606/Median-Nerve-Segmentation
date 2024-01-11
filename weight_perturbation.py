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
        
            frames = torch.stack([transform(Image.open(frame)) for frame in args.sub_frames])
        
            args.pred_masks_seq = []
        
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

    intensities = [-0.1,  -0.04, 0.04, 0.1]
    # intensities = [-0.1, -0.08, -0.06, -0.04, -0.02, 0, 0.02, 0.04, 0.06, 0.08, 0.1]
    
    res_on_int = []
    
    for intensity in intensities:
        
        args.perturb_model = intensity
        avg_score = get_predictions(args, print_score=False)
        
        res_on_int.append(avg_score)
        print(f'for intensity:{intensity} > {avg_score}') #chckpnt 7 with first 1 sec

    res_on_int = np.array(res_on_int)
    res_on_int = np.round(res_on_int,3)
    for i in range(len(res_on_int)):
        print(f'{intensities[i]} & {res_on_int[i,0]} & {res_on_int[i,1]} & {res_on_int[i,2]} &  {res_on_int[i,4]} & {res_on_int[i,5]} & {res_on_int[i,6]} & {res_on_int[i,7]} & {res_on_int[i,9]} \\\\')

if __name__ == '__main__':
    main()