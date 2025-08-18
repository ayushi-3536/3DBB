# 3DD Point Cloud Project (3DBB)

This repository contains code for **3D point cloud processing** using [PCDet](https://github.com/open-mmlab/OpenPCDet).  
It supports backbones like **PillarBackBone8x** and **PointNet2Backbone**, along with frozen pretrained encoders for feature extraction and multimodal fusion.  

- **Image Encoder:** Frozen [DINOv2 ViT-B](https://arxiv.org/abs/2304.07193)  
- **Point Cloud Encoder:** Lightweight pretrained model ([download here](https://drive.google.com/file/d/1wMxWTpU1qUoY3DsCH31WJmvJxcjFXKlm/view?usp=sharing))  

---

## 📑 Table of Contents
1. [Environment Setup](#environment-setup) 
2. [Model Checkpoints](#model-checkpoints)  
3. [Data Preparation](#data-preparation)  
6. [Usage](#6️⃣-usage)  

---

## Environment Setup

```bash
# Create Conda environment
conda create -n 3ddet python=3.8 -y
conda activate 3ddet

# Install PyTorch 1.11.0 with CUDA 11.5
pip install torch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu115

# Install spconv (for sparse convolution support)
pip install spconv-cu113

# Install other dependencies
pip install -r requirements.txt

## Model Checkpoint
mkdir -p checkpoint
pip install gdown
gdown https://drive.google.com/uc?id=1wMxWTpU1qUoY3DsCH31WJmvJxcjFXKlm -O checkpoint/pointpillar_gdrive.pth

##Data Preparation
run '../data_preprocessing/create_train_val_split.ipynb'
add 'train.csv' and 'val.csv' in your config file


###Train
./launcher/dist_launch.sh train.py /home/as2114/code/3DBB/config/multimodal.yaml 1


