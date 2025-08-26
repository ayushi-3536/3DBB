import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict
from scipy.optimize import linear_sum_assignment
import numpy as np # For numpy usage in linear_sum_assignment
from typing import Optional

# --- OpenPCDet Modules (Ensure OpenPCDet is installed and available) ---
from pcdet.models.backbones_3d.vfe.pillar_vfe import PillarVFE
from pcdet.models.backbones_2d.map_to_bev.pointpillar_scatter import PointPillarScatter
from pcdet.models.dense_heads.point_head_simple import PointHeadSimple as PointPillarHead
from pcdet.models.backbones_2d.base_bev_backbone import BaseBEVBackbone
from pcdet.ops.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu

from .builder import MODELS
from utils import get_logger

def corners_to_7d(corners):
    """
    Converts 8-corner 3D bounding box to 7D representation (x,y,z,dx,dy,dz,yaw).
    Assumes corners are ordered consistently for yaw estimation.
    Args:
        corners (torch.Tensor): Shape (8, 3) representing 8 XYZ coordinates of corners.
    Returns:
        torch.Tensor: Shape (7,) containing (x,y,z,dx,dy,dz,yaw).
    """
    
    center = corners.mean(dim=0) # (3,)
    min_coords = corners.min(dim=0).values
    max_coords = corners.max(dim=0).values
    dims = max_coords - min_coords # (3,)
    
    # Example yaw estimation assuming corners[0] and corners[1] define an edge along length
    vec_x = corners[1, :2] - corners[0, :2] 
    yaw = torch.atan2(vec_x[1], vec_x[0]) # atan2(y, x) for angle

    return torch.cat([center, dims, yaw.unsqueeze(0)], dim=0)

# --- PointPillar BEV Feature Extractor ---
class PointPillarBEVExtractor(nn.Module):
    def __init__(self, pc_cfg):
        super().__init__()

        vfe_model_cfg = EasyDict(pc_cfg.VFE)
        map_to_bev_cfg = EasyDict(pc_cfg.MAP_TO_BEV)
        backbone_2d_cfg = EasyDict(pc_cfg.BACKBONE_2D)

        point_cloud_range = torch.tensor(pc_cfg.POINT_CLOUD_RANGE) 
        voxel_size = pc_cfg.VOXEL_SIZE 
        
        # This grid_size represents the *initial* BEV grid dimensions after scattering
        grid_size = [(point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0],
                     (point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1],
                     (point_cloud_range[5] - point_cloud_range[2]) / voxel_size[2]]
        grid_size = torch.round(torch.tensor(grid_size)).int().tolist()
        
        map_to_bev_cfg.NUM_BEV_FEATURES = vfe_model_cfg.NUM_FILTERS[-1] 
        backbone_2d_cfg.INPUT_FEATURES = map_to_bev_cfg.NUM_BEV_FEATURES 

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

        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        self.grid_size = grid_size # Initial BEV grid size
        self.model_cfg = pc_cfg # Store full PC config for easy access to sub-components
        
    def forward(self, batch_dict):
        batch_dict = self.vfe(batch_dict)
        batch_dict = self.map_to_bev(batch_dict)
        batch_dict = self.backbone_2d(batch_dict) # This is where the downsampling happens
        bev_features = batch_dict['spatial_features_2d']
        return bev_features

class BEVResidualBlock(nn.Module):
    """Small residual block for BEV features"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = self.relu(out)
        return out
class BEVDetectionHeadResNet(nn.Module):
    """
    BEV 3D detection head with small ResNet-like conv blocks.
    Input: BEV features [B, C, H, W]
    Output: Predicted boxes [B, num_queries, 7], scores [B, num_queries, num_classes+1]
    """
    def __init__(self, in_channels, num_queries=100, num_blocks=3, hidden_dim=256):
        super().__init__()
        self.num_queries = num_queries

        # Reduce channels to hidden_dim
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )

        # Residual blocks
        self.res_blocks = nn.Sequential(*[BEVResidualBlock(hidden_dim) for _ in range(num_blocks)])

        # Final heads
        self.box_head = nn.Conv2d(hidden_dim, 7 * num_queries, kernel_size=1)
        self.num_classes = 1  # object class
        cls_head_out_channels = self.num_classes + 1  # +1 for "no object"
        self.cls_head = nn.Conv2d(hidden_dim, cls_head_out_channels * num_queries, kernel_size=1)

    def forward(self, bev_features):
        """
        Args:
            bev_features: [B, C, H, W]
        Returns:
            pred_boxes: [B, num_queries, 7]
            pred_logits: [B, num_queries, num_classes+1]
        """
        B, C, H, W = bev_features.shape
        x = self.initial_conv(bev_features)
        x = self.res_blocks(x)

        # Box predictions
        box_pred = self.box_head(x).view(B, self.num_queries, 7, -1).mean(-1)  # [B, num_queries, 7]

        # Classification predictions
        cls_pred = self.cls_head(x)  # [B, cls_out_channels*num_queries, H, W]
        cls_pred = cls_pred.view(B, self.num_queries, self.num_classes + 1, -1).mean(-1)  # [B, num_queries, num_classes+1]

        return box_pred, cls_pred


# --- Multimodal Fusion Detection Network ---
@MODELS.register_module() 
class PCDetectionNet(nn.Module):
    def __init__(self, cfg, point_coloring=False, num_queries=100, num_classes=1): 
        super().__init__()
        #do omegaconf
        
        #self.cfg = omegaconf.OmegaConf.create(cfg) if isinstance(cfg, dict) else cfg
        if not isinstance(cfg, EasyDict):
            cfg = EasyDict(cfg)

        self.cfg = cfg # Store full config
        self.logger = get_logger()
        self.num_queries = num_queries
        self.num_classes = num_classes 
        self.point_coloring = point_coloring
        # --- Point Cloud Branch (PointPillarBEVExtractor) ---
        self.pc_extractor = PointPillarBEVExtractor(self.cfg)
        
        # The BaseBEVBackbone's output is a concatenation of upsampled features.
        pc_bev_output_channels = sum(self.cfg.BACKBONE_2D.NUM_UPSAMPLE_FILTERS) # This correctly sums up to 384 based on your config
        print(f"PC BEV feature channels: {pc_bev_output_channels}")
        # --- Query-Based 3D Detection Head ---
        self.prediction_head = BEVDetectionHeadResNet(in_channels=pc_bev_output_channels)

        # Loss weights and cost weights for Hungarian matching
        self.loss_weights = {
            'reg_l1': 1.0, 'yaw_l1': 1.0, 'cls_pos': 1.0, 'cls_neg': 1.0
        }
        self.cost_cls_weight = 1.0
        self.cost_box_weight =5.0
        self.cost_iou_weight = 5.0
        self.yaw_weight = 2.0

    def _prepare_pc_batch_dict(self, pc_bev_dense, batch_size):
        """
        Converts a dense (B, 3, H, W) point cloud projection into the sparse
        voxel/pillar format required by PointPillarBEVExtractor.
        """
        _, C, H, W = pc_bev_dense.shape 
        
        num_voxels_per_cloud = H * W
        total_voxels = batch_size * num_voxels_per_cloud
        max_points_per_voxel = 1 

        voxels_transformed = pc_bev_dense.permute(0, 2, 3, 1).reshape(total_voxels, max_points_per_voxel, C)
        voxel_num_points = torch.ones(total_voxels, dtype=torch.int32)

        voxel_coords = torch.zeros(total_voxels, 4, dtype=torch.int32, device=pc_bev_dense.device)

        batch_indices = torch.arange(batch_size, device=pc_bev_dense.device).repeat_interleave(H * W)
        
        y_grid, x_grid = torch.meshgrid(
            torch.arange(H, device=pc_bev_dense.device),
            torch.arange(W, device=pc_bev_dense.device),
            indexing='ij'
        )
        y_indices = y_grid.flatten().repeat(batch_size)
        x_indices = x_grid.flatten().repeat(batch_size)
        
        # Ensure indices are strictly within bounds (0 to H-1, 0 to W-1)
        y_indices = torch.clamp(y_indices, 0, H - 1)
        x_indices = torch.clamp(x_indices, 0, W - 1)

        voxel_coords[:, 0] = batch_indices
        voxel_coords[:, 1] = 0 
        voxel_coords[:, 2] = y_indices
        voxel_coords[:, 3] = x_indices

        batch_dict = {
            'voxels': voxels_transformed,
            'voxel_num_points': voxel_num_points,
            'voxel_coords': voxel_coords,
            'batch_size': batch_size
        }
        return batch_dict
    def forward(self, fused_input, bbox3d=None):
        """
        Forward pass for multimodal 3D detection with IoU-based Hungarian matching.

        Args:
            fused_input (torch.Tensor): (B, 6, H, W) input where channels 0-2 are RGB and 3-5 are projected point cloud.
            bbox3d (list of torch.Tensor, optional): GT boxes per sample in (N,8,3) or (N,7) format.
        
        Returns:
            dict: Predicted boxes and, if training, computed losses.
        """
        device = fused_input.device
        B, C_fused, H_orig, W_orig = fused_input.shape

        # Split RGB and point cloud channels
        pc_projected_raw = fused_input[:, 3:, :, :]
        if self.point_coloring:
            rgb_features = fused_input[:, :3, :, :]  # RGB channels
            pc_projected_raw = torch.cat([pc_projected_raw, rgb_features], dim=1)  # [B, 6, H, W]


        # Resize to BEV grid
        pc_bev_H, pc_bev_W = self.pc_extractor.grid_size[1], self.pc_extractor.grid_size[0]
        if pc_projected_raw.shape[2:] != (pc_bev_H, pc_bev_W):
            pc_projected_resized = F.interpolate(pc_projected_raw, size=(pc_bev_H, pc_bev_W),
                                                mode='bilinear', align_corners=False)
        else:
            pc_projected_resized = pc_projected_raw

        # Prepare batch dict for PointPillar
        pc_batch_dict = self._prepare_pc_batch_dict(pc_projected_resized, B)
        pc_bev_features = self.pc_extractor(pc_batch_dict)

        # Prediction head
        pred_boxes_7d, pred_logits = self.prediction_head(pc_bev_features)
        pred_boxes_7d = pred_boxes_7d.cuda()
        pred_logits = pred_logits.cuda()
        print(f"Pred boxes shape: {pred_boxes_7d.shape}, Pred logits shape: {pred_logits.shape}")
        pred_boxes_compat = torch.cat([pred_boxes_7d, pred_logits[:, :, 0:1]], dim=-1)

        # Inference-only mode
        if bbox3d is None:
            return {'pred_boxes': pred_boxes_compat}
        

        # Convert GT boxes to 7D format
        gt_7d_boxes_list = [gt.to(device) for gt in bbox3d]

        # Initialize losses
        losses = {k: torch.tensor(0.0, device='cuda:0') for k in
                ['total_loss', 'reg_l1', 'yaw_l1', 'cls_pos', 'cls_neg', '3diouloss']}
        matched_info = [{} for _ in range(B)]

        for b in range(B):
            pred_boxes_sample = pred_boxes_7d[b]  # (Q,7)
            pred_logits_sample = pred_logits[b]   # (Q, num_classes)
            gt_boxes_sample = gt_7d_boxes_list[b] # (G,7)
            num_gt, num_preds = gt_boxes_sample.shape[0], self.num_queries

            # Zero GT case: all predictions are background
            if num_gt == 0:
                target_noobj = torch.full((num_preds,), 1, dtype=torch.long, device=device)  # background index = 1
                cls_loss = F.cross_entropy(pred_logits_sample, target_noobj)
                losses['cls_neg'] += self.loss_weights['cls_neg'] * cls_loss
                losses['total_loss'] += self.loss_weights['cls_neg'] * cls_loss
                continue

            # GT exists: Hungarian assignment
            gt_labels_sample = torch.zeros(num_gt, dtype=torch.long, device=device)  # foreground index = 0

            with torch.no_grad():
                iou_matrix = boxes_iou3d_gpu(pred_boxes_sample.cuda(), gt_boxes_sample.cuda())  # (Q, G)
                reg_cost = (
                    F.l1_loss(pred_boxes_sample[:, None, 0:3].cuda(), gt_boxes_sample[None, :, 0:3].cuda(), reduction='none').sum(-1) +
                    F.l1_loss(pred_boxes_sample[:, None, 3:6].cuda(), gt_boxes_sample[None, :, 3:6].cuda(), reduction='none').sum(-1)
                )
                cls_prob = pred_logits_sample.softmax(-1)
                cls_cost = -cls_prob[:, 0:1].expand(-1, num_gt)  # only foreground class
                iou_cost = 1 - iou_matrix
                pred_vec = torch.stack([torch.cos(pred_boxes_sample[:, 6]).cuda(), torch.sin(pred_boxes_sample[:, 6]).cuda()], dim=-1)
                yaw_cost = 1 - torch.abs(torch.matmul(pred_vec.cuda(),
                                                    torch.stack([torch.cos(gt_boxes_sample[:, 6].cuda()), torch.sin(gt_boxes_sample[:, 6].cuda())], dim=-1).t()))
                cost_matrix = (
                    self.cost_cls_weight * cls_cost +
                    self.cost_box_weight * reg_cost +
                    self.cost_iou_weight * iou_cost +
                    self.yaw_weight * yaw_cost
                ).cpu().numpy()

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            row_ind = torch.as_tensor(row_ind, device=device, dtype=torch.long)
            col_ind = torch.as_tensor(col_ind, device=device, dtype=torch.long)

            # Classification losses
            pred_logits_sample = pred_logits_sample.cuda()
            gt_labels_sample = gt_labels_sample.cuda()
            cls_loss_matched = F.cross_entropy(pred_logits_sample[row_ind], gt_labels_sample[col_ind])

            # Background for unmatched predictions
            unmatched_mask = torch.ones(num_preds, dtype=torch.bool, device=device)
            unmatched_mask[row_ind] = False
            if unmatched_mask.any():
                target_noobj = torch.full((unmatched_mask.sum(),), 1, dtype=torch.long, device=device)  # background index = 1
                cls_loss_unmatched = F.cross_entropy(pred_logits_sample[unmatched_mask].cuda(), target_noobj.cuda())
            else:
                cls_loss_unmatched = torch.tensor(0.0, device=device)

            # Regression losses for matched predictions
            matched_pred = pred_boxes_sample[row_ind]
            matched_gt = gt_boxes_sample[col_ind]
            
            matched_info[b] = {
                'pred_boxes': matched_pred,
                'gt_boxes': matched_gt,
                'scores': pred_logits_sample[row_ind].softmax(-1)[:,0]
            }


            reg_centers = F.l1_loss(matched_pred[:, 0:3].cuda(), matched_gt[:, 0:3].cuda(), reduction='mean')
            reg_sizes = F.l1_loss(matched_pred[:, 3:6].cuda(), matched_gt[:, 3:6].cuda(), reduction='mean')
            reg_loss_matched = reg_centers + reg_sizes

            # Yaw loss
            pred_vec = torch.stack([torch.cos(matched_pred[:, 6]).cuda(), torch.sin(matched_pred[:, 6]).cuda()], dim=-1)
            gt_vec = torch.stack([torch.cos(matched_gt[:, 6]).cuda(), torch.sin(matched_gt[:, 6]).cuda()], dim=-1)
            yaw_loss_matched = F.l1_loss(pred_vec, gt_vec, reduction='mean')

            # IoU loss
            iou_loss = 1 - boxes_iou3d_gpu(matched_pred.cuda(), matched_gt.cuda()).diagonal().mean()

            # Aggregate losses
            losses['reg_l1'] += reg_loss_matched.cuda()
            losses['yaw_l1'] += yaw_loss_matched.cuda()
            losses['cls_pos'] += cls_loss_matched.cuda()
            losses['cls_neg'] += cls_loss_unmatched.cuda()
            losses['3diouloss'] += iou_loss.cuda()
            losses['total_loss'] += (
                self.loss_weights['reg_l1'] * reg_loss_matched +
                self.loss_weights['yaw_l1'] * yaw_loss_matched +
                self.loss_weights['cls_pos'] * cls_loss_matched +
                self.loss_weights['cls_neg'] * cls_loss_unmatched +
                self.loss_weights.get('iou', 1.0) * iou_loss
            )

        # Average over batch
        for k in losses:
            losses[k] = losses[k] / B

        return {'loss': losses['total_loss'], 'pred_boxes': pred_boxes_compat, 'losses': losses, 'matched_info': matched_info}
