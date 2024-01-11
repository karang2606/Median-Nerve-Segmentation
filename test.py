import argparse
import glob
from dataset import *
from PIL import Image
from models.models import get_model
from utils.utils import *
from utils.perturbation import *
import torchvision.transforms as T
from tqdm import tqdm
from testing_module.eval_models import eval_UNet, eval_LSTM, eval_Siam, eval_VisTR
from pathlib import Path

def get_args_parser():
    
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--test_batch_size', default=4, type=int)
    parser.add_argument('--num_classes', default=1, type=int)
    parser.add_argument('--epochs', default=150, type=int)
    parser.add_argument('--perturb_input', default=None, type=float)
    parser.add_argument('--perturb_model', default=None, type=float)
    parser.add_argument('--no_GT', action='store_false',
                        help="Ground Truth not available")

    # Model parameters
    parser.add_argument('--model_name', type=str, default = 'unet',
                        help="Provide Model.")
    parser.add_argument('--load_from', type=str, default=None,
                        help="Path to the model weights.")
    # * Backbone
    parser.add_argument('--backbone', default='resnet101', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=384, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_frames', default=36, type=int,
                        help="Number of frames")
    parser.add_argument('--num_ins', default=1, type=int,
                        help="Number of instances")
    parser.add_argument('--num_queries', default=36, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_false',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    parser.add_argument('--no_labels_loss', dest='labels_loss', action='store_false',
                        help="Enables labels losses")
    parser.add_argument('--no_boxes_loss', dest='boxes_loss', action='store_false',
                        help="Enables bounding box losses")
    parser.add_argument('--no_L1_loss', dest='L1_loss', action='store_false',
                        help="Enables L1 losses for bboxes")
    parser.add_argument('--no_giou_loss', dest='giou_loss', action='store_false',
                        help="Enables Generalized IOU losses for bboxes")
    parser.add_argument('--no_focal_loss', dest='focal_loss', action='store_false',
                        help="Enables Focal losses for mask")
    parser.add_argument('--no_dice_loss', dest='dice_loss', action='store_false',
                        help="Enables dice losses for mask")
    
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--wrist', action='store_true',
                        help="If true, model will train for only first 40 frames. i.e. wrist area")
    parser.add_argument('--save_clip', action='store_true',
                        help="If true, model will save the clip with prediction contours.")
    parser.add_argument('--filter', action='store_true',
                        help="If true, model will filter the clip of prediction contours.")
    parser.add_argument('--data_path', default='data/test/')
    parser.add_argument('--save_path', default='results.json')
    parser.add_argument('--dataset_file', default='ytvos')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='output_ytvos',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_false')
    parser.add_argument('--num_workers', default=0, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser



def main():

    parser = argparse.ArgumentParser('Model training and evaluation script',
                                 parents=[get_args_parser()])
    args = parser.parse_args()

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
    #     model = add_flops_counting_methods(model)

        #  area mm2/ a pixel
        const = (30/448)**2
        
        args.test_file_dir = sorted(glob.glob(args.data_path+'*'))
        avg_score = 0
        avg_cs = 0
        GT_avg_cs = 0
        
        avg_fps = []
        
        for path in args.test_file_dir:

            args.sub_name = path.split('/')[-1]
            args.sub_frames = sorted(glob.glob(path+'/images/*.jpg'), key=key_func)
            args.clip_len = 40 if args.wrist else len(args.sub_frames)
            
            frames = torch.stack([transform(Image.open(frame)) for frame in args.sub_frames])

            args.score = 0
            args.pred_masks_seq = []
            
            if args.model_name == 'siam_unet':
                args.sub_masks = sorted(glob.glob(path+'/masks/*.png'), key=key_func)
                eval_Siam(args, model, frames)
                args.sub_frames = args.sub_frames[1:]
                args.clip_len -=1

            elif args.model_name == 'lstm_unet':
                eval_LSTM(args, model, frames)

            elif  args.model_name == 'vistr':
                eval_VisTR(args, model, frames)

            else:
                eval_UNet(args, model, frames)

            avg_fps.append(args.fps)

            area = np.sum(np.array(args.pred_masks_seq), axis=(1,2))*const
            mean_cs = np.mean(area)
            avg_cs += mean_cs

            if args.filter: args.pred_masks_seq = get_filtered_mask(args.pred_masks_seq)
                
            if args.no_GT:
                args.sub_masks = sorted(glob.glob(path+'/masks/*.png'), key=key_func)

                true_masks = [np.array(Image.open(msk))/255 for msk in args.sub_masks]

                args.score = np.round(get_metric(args.pred_masks_seq, true_masks), 3)
                avg_score += args.score

                GT_area = np.sum(np.array(true_masks), axis=(1,2))*const
                GT_mean_cs = np.mean(GT_area)
                GT_avg_cs += GT_mean_cs
                print(f'\n{args.sub_name}, Recall: {args.score[0]}, Precision: {args.score[1]}, F1_Score: {args.score[2]}, Dice_Score: {args.score[3]}, Hausdorff Distance: {args.score[4]}, GT C/S: {round(GT_mean_cs,3)} mm\u00b2, C/S: {round(mean_cs,3)} mm\u00b2')

            if args.save_clip or not args.no_GT:
                Path('./segmented_clips/').mkdir(parents=True, exist_ok=True)
                create_clip(args)
                print(f"{args.sub_name} Clip Saved!\n")

    print('Frames per second', round(np.mean(avg_fps), 3))
    if args.no_GT:
        avg_score = np.round(avg_score/len(args.test_file_dir), 3)
        print("\nAverage Metric")
        print(f'Recall: {avg_score[0]}, Precision: {avg_score[1]}, F1_Score: {avg_score[2]}, Dice_Score: {avg_score[3]}, Hausdorff Distance: {avg_score[4]}, GT C/S: {round(GT_avg_cs/len(args.test_file_dir), 3)} mm\u00b2, C/S: {round(avg_cs/len(args.test_file_dir), 3)} mm\u00b2')
        print({round(GT_avg_cs/len(args.test_file_dir), 3)})
        print(f'{avg_score[0]} & {avg_score[1]} & {avg_score[3]} & {avg_score[4]}  & {round(avg_cs/len(args.test_file_dir), 3)}')
            
if __name__ == '__main__':
    main()
