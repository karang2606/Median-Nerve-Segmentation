# Median-Nerve-Segmentation

## Installation

Clone the git-repository first.
```
git clone https://github.com/karang2606/Median-Nerve-Segmentation.git
```

Then, install the latest version of Pytorch and Torchvision from [this link]{https://pytorch.org/get-started/locally/} or below.
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

Download the pretrained DETR models [Google Drive]{https://drive.google.com/drive/folders/1DlN8uWHT2WaKruarGW2_XChhpZeI9MFG}
on COCO and save it to the pretrained path.

## Training
Training the model requires at least 30GB of GPU memory, so we have utilized two NVIDIA RTX A6000 GPU cards with a memory of 48GB each.

Details of some training arguments:
--model_name: Provide model name from any of these. (unet, unetpp, attn_unet, siam_unet, lstm_unet, trans_unet, vistr)



