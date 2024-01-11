import argparse
import torch
from models.models import get_model
from utils.utils import load_model
from mmcv.cnn.utils.flops_counter import add_flops_counting_methods, flops_to_string, params_to_string

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
    #parser.add_argument('--eval', action='store_true')
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
    
    if args.model_name != 'vistr':
        model = get_model(args)
    else:
        args.masks = True
        model, _, _ = get_model(args)
    
    model.to(args.device)
    model.eval()
    model = add_flops_counting_methods(model)

    with torch.no_grad():
        
        model.start_flops_count()
        if args.model_name == 'vistr':
            a = torch.rand((36,3,448,336))
        
        elif args.model_name == 'siam_unet':
            a = torch.rand((16,1,448,336))
            b = torch.rand((16,1,144,80))
        
        elif args.model_name == 'lstm_unet':
            a = torch.rand((4,10,1,448,336))
        
        else:
            a = torch.rand((16,1,448,336))
        
        if args.model_name == 'siam_unet':
            output = model(a.to(args.device), b.to(args.device))
        else:
            output = model(a.to(args.device))
        
        AVG_flops, params_count = model.compute_average_flops_cost()
        print('Average flops',flops_to_string(AVG_flops))
        print('Parameters',params_to_string(params_count))
        model.stop_flops_count()

if __name__ == '__main__':
    main()