# Median-Nerve-Segmentation

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

## Training
Training the model requires at least 30GB of GPU memory, so we have utilized two NVIDIA RTX A6000 GPU cards with a memory of 48GB each.

Details of some training arguments: <br />
--model_name: Provide model name from any of these.

| Model name        | Argument |
| :------------- |:-------------:|
| [UNet](https://arxiv.org/abs/1505.04597)      | unet |
| [UNet++](https://arxiv.org/abs/1807.10165)      | unetpp |
| [Attention UNet](https://arxiv.org/abs/1804.03999) | attn_unet |
| [Siamese UNet](https://www.sciencedirect.com/science/article/pii/S1361841519301677) | siam_unet |
|  [LSTM UNet](https://github.com/Michael-MuChienHsu/R_Unet) | lstm_unet |
| [TransUNet](https://arxiv.org/abs/2102.04306) | trans_unet |
| [Video Instance Segmentation with Transformers](https://arxiv.org/abs/2011.14503)| vistr |

> [UNet](https://arxiv.org/abs/1505.04597): unet <br/>
> [UNet++](https://arxiv.org/abs/1807.10165): unetpp <br/>
> [Attention UNet](https://arxiv.org/abs/1804.03999): attn_unet <br/>
> [Siamese UNet](https://www.sciencedirect.com/science/article/pii/S1361841519301677): siam_unet <br/>
> [LSTM UNet](https://github.com/Michael-MuChienHsu/R_Unet): lstm_unet <br/> 
> [TransUNet](https://arxiv.org/abs/2102.04306): trans_unet <br/>
> [Video Instance Segmentation with Transformers](https://arxiv.org/abs/2011.14503): vistr <br/>


[//]: # (this wont be included)
