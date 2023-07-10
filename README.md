# Median-Nerve-Segmentation
---
## Installation

Clone the git-repository first.
```
git clone https://github.com/karang2606/Median-Nerve-Segmentation.git
```

Then, install the latest version of Pytorch and Torchvision from [this link](https://pytorch.org/get-started/locally/) or below.
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```
Install pycocotools
```
conda install cython scipy
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
pip install git+https://github.com/youtubevos/cocoapi.git#"egg=pycocotools&subdirectory=PythonAPI"
```
From the pycocotools folder in the repository, copy ytvos.py and ytvoseval.py to the package installation 
location in your system, i.e., home/anaconda3/lib/python3.10/site-packages/pycocotools/

Compile DCN module(requires GCC>=5.3, cuda>=10.0)
```
cd models/dcn
python setup.py build_ext --inplace
```
---
## Preparation
Set directory structure as follows:
```
VisTR
├── data
│   ├── train
│   |   ├──patient_01
│   |   |   ├──images
│   |   |   |   ├──0.jpg, ....
│   |   |   ├──masks
│   |   |       ├──0.png, ....
│   ├── test
...
```

Download the pretrained DETR models [Google Drive](https://drive.google.com/drive/folders/1DlN8uWHT2WaKruarGW2_XChhpZeI9MFG)
on COCO and save it to the pretrained path.
---
## Training
Training the model requires at least 30GB of GPU memory, so we have utilized two NVIDIA RTX A6000 GPU cards with a memory of 48GB each.

Details of some training arguments: <br/>

`--model_name`: Provide the model name from any of these.

| Model name        | Argument |
| :------------- |:-------------:|
| [UNet](https://arxiv.org/abs/1505.04597)[^1]      | unet |
| [UNet++](https://arxiv.org/abs/1807.10165)[^2]    | unetpp |
| [Attention UNet](https://arxiv.org/abs/1804.03999)[^3] | attn_unet |
| [Siamese UNet](https://www.sciencedirect.com/science/article/pii/S1361841519301677)[^4] | siam_unet |
| [LSTM UNet](https://github.com/Michael-MuChienHsu/R_Unet)[^5] | lstm_unet |
| [TransUNet](https://arxiv.org/abs/2102.04306)[^6] | trans_unet |
| [Video Instance Segmentation with Transformers](https://arxiv.org/abs/2011.14503)[^7] | vistr |


`--no_aug`: If true, we augment the data by doing horizontal and vertical flips. <br/>
`--data_path`: Provide a path for training data. The default is "data/train/" <br/>
`--lr`: Learning rate <br/>
`--num_frames`: For LSTM and VisTR, clip length for training the model. The default is 36. <br/>
`--epochs`: Number of epochs for training. <br/>
`--pretrained_weights`: Path to pretrained weights for VisTR backbone. The default is "pretrained/384_coco_r101.pth" <br/>
<br/>
To train the model, run the following:
```
python train_script.py --model_name unet --batch_size 16 --val_batch_size 16 --no_aug
```

---
## Evaluation
Details of some test arguments: <br/>
`--data_path`: Provide a path for testing data. The default is "data/test/" <br/>
`--load_from`: Path to pretrained model weights.
`--save_clip`: If given, the model will save each test video with the segmentation result and cross-sectional area of
the nerve at the bottom.
`--perturb_input`: Perturbe the input frame with Poisson noise.
`--perturb_model`: Perturbe the model with a given fraction.

To test the model, run the following:
```
python evaluate.py --model_name vistr --perturb_input 0.05 --perturb_model 0.1
```

---
## Acknowledgement
We would like to thank the DETR and VisTR open-source projects for their awesome work; part of the code is modified from their project.

---
## References

[^1]: Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. "U-net: Convolutional networks for biomedical image segmentation." Medical Image Computing and Computer-Assisted Intervention–MICCAI 2015: 18th International Conference, Munich, Germany, October 5-9, 2015, Proceedings, Part III 18. Springer International Publishing, 2015.

[^2]: Zhou, Zongwei, et al. "Unet++: A nested u-net architecture for medical image segmentation." Deep Learning in Medical Image Analysis and Multimodal Learning for Clinical Decision Support: 4th International Workshop, DLMIA 2018, and 8th International Workshop, ML-CDS 2018, Held in Conjunction with MICCAI 2018, Granada, Spain, September 20, 2018, Proceedings 4. Springer International Publishing, 2018.

[^3]: Oktay, Ozan, et al. "Attention u-net: Learning where to look for the pancreas." arXiv preprint arXiv:1804.03999 (2018).

[^4]: Dunnhofer, Matteo, et al. "Siam-U-Net: encoder-decoder siamese network for knee cartilage tracking in ultrasound images." Medical Image Analysis 60 (2020): 101631.

[^5]: Hsu, Mu Chien, Jui Chun Shyur, and Hiroshi Watanabe. "Pseudo Ground Truth Segmentation Mask to Improve Video Prediction Quality." 2020 IEEE 9th Global Conference on Consumer Electronics (GCCE). IEEE, 2020.

[^6]: Chen, Jieneng, et al. "Transunet: Transformers make strong encoders for medical image segmentation." arXiv preprint arXiv:2102.04306 (2021).

[^7]: Wang, Yuqing, et al. "End-to-end video instance segmentation with transformers." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2021.

