# 3DD Point Cloud Project (3DBB)

This repository provides code for **3D point cloud processing** using [OpenPCDet](https://github.com/open-mmlab/OpenPCDet).  
It supports flexible backbones like **PillarBackBone8x** and **PointNet2Backbone**, with support for **frozen pretrained encoders** for multimodal fusion.

### 🔍 Modalities & Features
- **Image Encoder:** Frozen [DINOv2 ViT-B](https://arxiv.org/abs/2304.07193)  
- **Point Cloud Encoder:** Lightweight pretrained model ([Download here](https://drive.google.com/file/d/1wMxWTpU1qUoY3DsCH31WJmvJxcjFXKlm/view?usp=sharing))

---

## 📑 Table of Contents
1. [Environment Setup](#environment-setup) 
2. [Model Checkpoints](#model-checkpoints)  
3. [Data Preparation](#data-preparation)  

---

## 🛠️ Environment Setup

```bash
# Create and activate a Conda environment
conda create -n 3ddet python=3.8 -y
conda activate 3ddet

# Install PyTorch 1.11.0 with CUDA 11.5
pip install torch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu115

# Install spconv (for sparse convolution support)
pip install spconv-cu113

# Install additional dependencies
pip install -r requirements.txt

# Install OpenPCDet
Follow the official instructions here:
# https://github.com/open-mmlab/OpenPCDet/blob/master/docs/INSTALL.md
```
## Checkpoint for PointPillar
mkdir -p checkpoint
pip install gdown
gdown https://drive.google.com/uc?id=1wMxWTpU1qUoY3DsCH31WJmvJxcjFXKlm -O checkpoint/pointpillar_gdrive.pth

## Data Preparation

1. Run the following notebook to create training and validation splits:

../data_preprocessing/create_train_val_split.ipynb

## Training

```bash
./launcher/dist_launch.sh train.py /home/as2114/code/3DBB/config/pc.yaml 1

```

## Training using PointPillar with a single ResNet-like detection head

```bash
./launcher/dist_launch.sh train.py /home/as2114/code/3DBB/config/pointpillar.yaml 1

```

## Training using PointPainting with a single ResNet-like detection head

```bash
./launcher/dist_launch.sh train.py /home/as2114/code/3DBB/config/pointcoloring.yaml 1

```


