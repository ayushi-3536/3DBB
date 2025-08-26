import torch
import torch.nn as nn
from easydict import EasyDict
from .builder import MODELS
# Import necessary modules from OpenPCDet (pcdet)
# Ensure you have OpenPCDet installed, as these are internal components.
from pcdet.models.backbones_3d.vfe.pillar_vfe import PillarVFE
from pcdet.models.backbones_2d.map_to_bev.pointpillar_scatter import PointPillarScatter
from pcdet.models.backbones_2d.base_bev_backbone import BaseBEVBackbone

def make_voxel_batch_dict(pointclouds, voxel_size=[0.16, 0.16, 4.0], point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1], max_points_per_voxel=32):
    """
    Converts a batch of point clouds to OpenPCDet-style voxel batch dictionary.
    
    Args:
        pointclouds (list of torch.Tensor or torch.Tensor): List of point clouds per batch 
            or a single tensor of shape (B, N, 4), where 4 = (x, y, z, intensity)
        voxel_size (list): [vx, vy, vz] voxel size
        point_cloud_range (list): [x_min, y_min, z_min, x_max, y_max, z_max]
        max_points_per_voxel (int): Max points per voxel
    
    Returns:
        batch_dict (dict): Contains 'voxels', 'voxel_num_points', 'voxel_coords', 'batch_size'
    """
    if isinstance(pointclouds, torch.Tensor):
        # Shape: (B, N, 4)
        B, N, C = pointclouds.shape
    else:
        # List of B tensors
        B = len(pointclouds)
        N = max(pc.shape[0] for pc in pointclouds)
        C = pointclouds[0].shape[1]

    all_voxels = []
    all_voxel_coords = []
    all_voxel_num_points = []

    # Convert to voxel indices
    pc_range = torch.tensor(point_cloud_range, dtype=torch.float32)
    voxel_size = torch.tensor(voxel_size, dtype=torch.float32)
    
    for b_idx in range(B):
        pc = pointclouds[b_idx] if isinstance(pointclouds, list) else pointclouds[b_idx]
        # Compute voxel indices
        voxel_coords = ((pc[:, :3] - pc_range[:3]) / voxel_size).floor().int()
        # Clip to valid voxel grid
        grid_size = ((pc_range[3:] - pc_range[:3]) / voxel_size).floor().int()
        voxel_coords[:, 0] = voxel_coords[:, 0].clamp(0, grid_size[0]-1)
        voxel_coords[:, 1] = voxel_coords[:, 1].clamp(0, grid_size[1]-1)
        voxel_coords[:, 2] = voxel_coords[:, 2].clamp(0, grid_size[2]-1)

        # Combine voxel coords with batch index
        batch_coords = torch.cat([torch.full((voxel_coords.shape[0],1), b_idx, dtype=torch.int32), voxel_coords], dim=1)
        
        # Group points per voxel (simple approach)
        voxel_dict = {}
        for i, vc in enumerate(batch_coords):
            key = tuple(vc.tolist())
            if key not in voxel_dict:
                voxel_dict[key] = []
            voxel_dict[key].append(pc[i])
        
        # Create voxel tensors
        for k, points in voxel_dict.items():
            points = torch.stack(points)
            num_points = min(points.shape[0], max_points_per_voxel)
            # Pad if needed
            padded_points = torch.zeros((max_points_per_voxel, C), dtype=pc.dtype)
            padded_points[:num_points] = points[:num_points]
            
            all_voxels.append(padded_points)
            all_voxel_num_points.append(num_points)
            all_voxel_coords.append(torch.tensor(k, dtype=torch.int32))
    
    batch_dict = {
        'voxels': torch.stack(all_voxels),                  # (num_voxels, max_points, C)
        'voxel_num_points': torch.tensor(all_voxel_num_points),  # (num_voxels,)
        'voxel_coords': torch.stack(all_voxel_coords),      # (num_voxels, 4)
        'batch_size': B
    }

    return batch_dict

@MODELS.register_module()
class PointPillarBEVExtractor(nn.Module):
    def __init__(self):
        super().__init__()

        # --- Model Configuration ---
        # These configs are simplified. In a real OpenPCDet project, they
        # would come from a comprehensive YAML configuration file.
        # They are tailored to mimic a common KITTI PointPillars setup.
        
        # Voxel Feature Encoder (VFE) Config
        vfe_model_cfg = EasyDict()
        vfe_model_cfg.USE_NORM = True
        vfe_model_cfg.NUM_POINT_FEATURES = 4  # Assuming (x, y, z, intensity)
        vfe_model_cfg.NUM_FILTERS = [64]
        vfe_model_cfg.WITH_DISTANCE = False
        vfe_model_cfg.USE_ABSLOTE_XYZ = True

        # Point Cloud Range & Voxel Size (Common for KITTI-like datasets)
        # [X_min, Y_min, Z_min, X_max, Y_max, Z_max]
        point_cloud_range = torch.tensor([0, -39.68, -3, 69.12, 39.68, 1]) 
        # [voxel_x_size, voxel_y_size, voxel_z_size]
        voxel_size = [0.16, 0.16, 4.0] 
        
        # Calculate grid size for BEV map
        grid_size = [(point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0],
                     (point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1],
                     (point_cloud_range[5] - point_cloud_range[2]) / voxel_size[2]]
        grid_size = torch.round(torch.tensor(grid_size)).int().tolist()
        
        # MapToBEV (Scatter) Config
        map_to_bev_cfg = EasyDict()
        map_to_bev_cfg.NUM_BEV_FEATURES = vfe_model_cfg.NUM_FILTERS[-1] # Output channels from VFE (64)

        # 2D Backbone (CNN on BEV map) Config - Similar to SECOND network
        backbone_2d_cfg = EasyDict()
        backbone_2d_cfg.NUM_UPSAMPLE_LAYERS = [128, 128, 128] # Simple backbone, no upsampling for direct BEV feature
        backbone_2d_cfg.LAYER_NUMS = [3, 5, 5]
        backbone_2d_cfg.LAYER_STRIDES = [1, 1, 1]
        backbone_2d_cfg.NUM_FILTERS = [4, 16, 64] # Output channels for each stage
        # Calculate input channels for 2D backbone (must match map_to_bev_cfg.NUM_BEV_FEATURES)
        backbone_2d_cfg.INPUT_FEATURES = map_to_bev_cfg.NUM_BEV_FEATURES 

        # --- Initialize Model Components ---
        self.vfe = PillarVFE(
            model_cfg=vfe_model_cfg,
            num_point_features=vfe_model_cfg.NUM_POINT_FEATURES,
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range
        )

        self.map_to_bev = PointPillarScatter(
            model_cfg=map_to_bev_cfg,
            num_bev_features=map_to_bev_cfg.NUM_BEV_FEATURES,
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            grid_size=grid_size
        )
        
        self.backbone_2d = BaseBEVBackbone(
            model_cfg=backbone_2d_cfg,
            input_channels=backbone_2d_cfg.INPUT_FEATURES
        )

        # Store these for potential use in the main multimodal fusion model
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        self.grid_size = grid_size
        
        # Dummy head - this will be removed when integrating into your full multimodal model
        # The true BEV features will be `batch_dict['spatial_features_2d']`
        self.dummy_detection_head = nn.Linear(backbone_2d_cfg.NUM_FILTERS[-1], 10) 

    def forward(self, pointclouds, only_return_bev=True):
        """
        Processes point cloud data through the PointPillars encoding stages
        to generate BEV features.

        Args:
            batch_dict (dict): Contains voxelized point cloud data.
                Expected keys: 'voxels', 'voxel_num_points', 'voxel_coords'.
                'voxel_coords' should be (N, 4) -> (batch_idx, z_idx, y_idx, x_idx).
        
        Returns:
            torch.Tensor: The dense BEV feature map (spatial_features_2d).
                          Shape will be (BatchSize, Channels, Height, Width).
        """
        batch_dict = make_voxel_batch_dict(pointclouds)
        # 1. Voxel Feature Encoding (PillarVFE)
        # Converts raw points within voxels/pillars into fixed-size features.
        batch_dict = self.vfe(batch_dict)
        # print("Batch dict after VFE:", batch_dict.keys())
        # Expected new keys: 'pillar_features', 'pillar_coords' (or 'voxel_features', 'voxel_coords')
        print("Batch dict after VFE:", batch_dict.keys())
        for k, v in batch_dict.items():
            print(f"{k}: {v.shape if isinstance(v, torch.Tensor) else v}")

        # 2. Map Pillar Features to BEV (PointPillarScatter)
        # Scatters the pillar features onto a 2D BEV grid.
        batch_dict = self.map_to_bev(batch_dict)
        # print("Batch dict after MapToBEV:", batch_dict.keys())
        # Expected new key: 'spatial_features' (the initial BEV map)

        # 3. 2D Backbone on BEV (BaseBEVBackbone)
        # Processes the sparse BEV map with 2D convolutions to create dense features.
        batch_dict = self.backbone_2d(batch_dict)
        # print("Batch dict after BaseBEVBackbone:", batch_dict.keys())
        # Expected new key: 'spatial_features_2d' (the final BEV feature map)
        
        # The final BEV features for fusion are in batch_dict['spatial_features_2d']
        print("Batch dict after BaseBEVBackbone:", batch_dict.keys())
        for k, v in batch_dict.items():
            print(f"{k}: {v.shape if isinstance(v, torch.Tensor) else v}")
        bev_features = batch_dict['spatial_features_2d']
        
        return bev_features # Return the actual BEV features

# if __name__ == "__main__":

#     batch_size = 1 

#     num_dummy_voxels = 1000 # Number of non-empty voxels/pillars
#     max_points_per_dummy_voxel = 10 # Max points to simulate per pillar
#     num_point_features = 4 # (x, y, z, intensity)

#     dummy_voxels = torch.randn(num_dummy_voxels, max_points_per_dummy_voxel, num_point_features)
#     dummy_voxel_num_points = torch.randint(1, max_points_per_dummy_voxel + 1, (num_dummy_voxels,))
    
#     # voxel_coords: (N, 4) -> (batch_idx, z_idx, y_idx, x_idx)
#     # For batch_size=1, batch_idx is always 0.
#     # z_idx can be simplified if using 2D BEV only (e.g., set to 0 or derived from actual z).
#     # y_idx and x_idx depend on the grid_size derived from point_cloud_range and voxel_size.
    
#     # Calculate approximate max x/y indices for dummy coords
#     model_instance = PointPillarBEVExtractor() # Initialize to get grid_size
#     max_x_idx = model_instance.grid_size[0] - 1
#     max_y_idx = model_instance.grid_size[1] - 1

#     dummy_voxel_coords = torch.zeros(num_dummy_voxels, 4, dtype=torch.int32)
#     dummy_voxel_coords[:, 0] = 0 # Batch index
#     dummy_voxel_coords[:, 1] = torch.randint(0, model_instance.grid_size[2], (num_dummy_voxels,)) # Dummy Z index
#     dummy_voxel_coords[:, 2] = torch.randint(0, max_y_idx + 1, (num_dummy_voxels,)) # Y index
#     dummy_voxel_coords[:, 3] = torch.randint(0, max_x_idx + 1, (num_dummy_voxels,)) # X index


#     # Create the batch_dict
#     batch_dict = {
#         'voxels': dummy_voxels,
#         'voxel_num_points': dummy_voxel_num_points,
#         'voxel_coords': dummy_voxel_coords,
#         'batch_size': batch_size # Essential for OpenPCDet's batch processing
#     }

#     model = PointPillarBEVExtractor()
    
#     # Move model and data to GPU if available
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model.to(device)
#     for k, v in batch_dict.items():
#         if isinstance(v, torch.Tensor):
#             batch_dict[k] = v.to(device)
    
#     print(f"Running model on {device}...")
#     with torch.no_grad(): # No need for gradients during feature extraction demo
#         bev_features = model(batch_dict)
    
#     print("\n--- Final Output ---")
#     print(f"Extracted BEV Feature Map Shape (B, C, H, W): {bev_features.shape}")
