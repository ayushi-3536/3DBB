import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn
from utils import corners_to_7d, get_logger


class PointCloudDataset(Dataset):
    def __init__(self, dataframe, transform=None, pc_transform=None, size=(504, 504)):
        self.data = dataframe
        self.size = size
    
        # Pre-calculated Normalization Statistics using data_preprocessing/calculate_mean_std_pc.py
        self.pc_mean = torch.tensor([0.000145, -0.002144, 0.940843])
        self.pc_std = torch.tensor([0.206461, 0.156468, 0.261020])
        self.bbox_mean = torch.tensor([0.005002, 0.000544, 1.074962, 0.125731, 0.035442, 0.100822, 0.016726])
        self.bbox_std = torch.tensor([0.110863, 0.093869, 0.120697, 0.048538, 0.033396, 0.060110, 1.835016])

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(self.size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transform
        
        if pc_transform is None:
            self.pc_transform = nn.Identity()
        else:
            self.pc_transform = pc_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # 1. Load and transform Image
        image = Image.open(row['image']).convert('RGB')
        image = self.transform(image)

        # 2. Load, resize, and normalize Point Cloud
        pc = torch.from_numpy(np.load(row['pc']).astype(np.float32))
        pc = pc.unsqueeze(0)
        pc = F.interpolate(pc, size=self.size, mode='nearest')
        pc = pc.squeeze(0)
        pc = self.pc_transform(pc)

        # 3. Early fusion
        fused_input = torch.cat([image, pc], dim=0)
        
        # 4. Load and Normalize 3D Bounding Boxes
        bbox3d = torch.from_numpy(np.load(row['bbox3d']).astype(np.float32))

        temp_bboxes_7 = []
        if bbox3d.shape[0] > 0: # Check if there are any bounding boxes
            for b_corners in bbox3d:
                temp_bboxes_7.append(corners_to_7d(b_corners)[:7])

        if not temp_bboxes_7:
            # Return an empty tensor of the correct shape if no boxes
            bbox3d = torch.empty((0, 7), dtype=torch.float32)
        else:
            # Stack the list of tensors into a single tensor
            bbox3d = torch.stack(temp_bboxes_7)
            
        
        return {
            'fused_input': fused_input,
            'bbox3d': bbox3d
        }