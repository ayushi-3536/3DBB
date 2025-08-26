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
from pcdet.models.backbones_2d.base_bev_backbone import BaseBEVBackbone
from pcdet.ops.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu
from pcdet.ops.iou3d_nms import iou3d_nms_utils
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

# --- Custom Dinov2 Class ---
class Dinov2(nn.Module):
    def __init__(self):
        super(Dinov2, self).__init__()
        try:
            self.dinomodel = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
            print("DINOv2 model loaded successfully.")
        except Exception as e:
            print(f"Error loading DINOv2 model: {e}")
        
        for param in self.dinomodel.parameters():
                param.requires_grad = False
        print(f"DINOv2 model parameters frozen.")

        self.output_feat_size = self.dinomodel.embed_dim # 768 for vitb14
        dino_patch_size = self.dinomodel.patch_size # 14 for vitb14
        
        # Images must be resized to this resolution before input to DINOv2
        self.dino_input_H = 504 
        self.dino_input_W = 504
        
        dino_patch_grid_H = self.dino_input_H // dino_patch_size
        dino_patch_grid_W = self.dino_input_W // dino_patch_size
        self.patch_grid_shape = (dino_patch_grid_H, dino_patch_grid_W)

    def forward_feature(self, image):
        patch_feat = self.dinomodel.forward_features(image)['x_norm_patchtokens']
        dino_features_2d = patch_feat.permute(0, 2, 1).view(
            patch_feat.size(0), 
            self.output_feat_size, 
            self.patch_grid_shape[0], 
            self.patch_grid_shape[1]  
        )
        return dino_features_2d 
    
    @torch.no_grad()
    def forward_test(self, image):
        patch_feat = self.dinomodel.forward_features(image)['x_norm_patchtokens']
        dino_features_2d = patch_feat.permute(0, 2, 1).view(
            patch_feat.size(0), 
            self.output_feat_size, 
            self.patch_grid_shape[0], 
            self.patch_grid_shape[1]  
        )
        return dino_features_2d
    
    def forward(self, image, train=True):
        if not train:
            return self.forward_test(image)
        else:   
            return self.forward_feature(image)


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

# --- Image to BEV Projection Module (Learned Alignment) ---
class ImageToBEVProjector(nn.Module):
    def __init__(self, image_input_channels, bev_output_channels, target_H_bev, target_W_bev, image_patch_grid_size=(36, 36)): 
        super().__init__()
        self.target_H_bev = target_H_bev
        self.target_W_bev = target_W_bev
        self.image_patch_grid_size = image_patch_grid_size 

        self.projection_convs = nn.Sequential(
            nn.Conv2d(image_input_channels, bev_output_channels // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(bev_output_channels // 2),
            nn.ReLU(True),
            nn.Conv2d(bev_output_channels // 2, bev_output_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(bev_output_channels),
            nn.ReLU(True),
        )
        self.output_channels = bev_output_channels

    def forward(self, image_features_2d):
        x = self.projection_convs(image_features_2d)
        
        # Add a debug print to confirm the target size before interpolation
        # self.logger.debug(f"DEBUG_Projector: Interpolating Image features to size=({self.target_H_bev}, {self.target_W_bev})")

        x = nn.functional.interpolate(
            x, 
            size=(self.target_H_bev, self.target_W_bev), 
            mode='bilinear', 
            align_corners=False
        )
        return x
    

class DetectionHead(nn.Module):
    def __init__(self, fused_channels, num_queries, num_classes):
        super().__init__()
        self.num_queries = num_queries
        self.num_classes = num_classes

        # Shared linear layer
        self.shared_fc = nn.Sequential(
            nn.Linear(fused_channels, 256),
            nn.ReLU(True)
        )

        # Separate heads
        self.box_xyz_yaw = nn.Linear(256, num_queries * 4)      # x, y, z, yaw
        self.box_dims = nn.Linear(256, num_queries * 3)         # dx, dy, dz
        self.cls_head = nn.Linear(256, num_queries * (num_classes + 1)) # classes + confidence

    def forward(self, fused_features):
        batch_size = fused_features.size(0)
        shared = self.shared_fc(fused_features)

        # Predict unconstrained xyz+yaw
        box_xyz_yaw = self.box_xyz_yaw(shared).view(batch_size, self.num_queries, 4)

        # Predict dimensions and enforce positivity
        box_dims = F.softplus(self.box_dims(shared)).view(batch_size, self.num_queries, 3)

        # Class predictions
        cls_pred = self.cls_head(shared).view(batch_size, self.num_queries, self.num_classes + 1)

        # Concatenate box outputs: [x, y, z, dx, dy, dz, yaw]
        box_pred = torch.cat([box_xyz_yaw[:, :, :3], box_dims, box_xyz_yaw[:, :, 3:4]], dim=-1)


        return box_pred, cls_pred

# --- Multimodal Fusion Detection Network ---
@MODELS.register_module() 
class MultimodalDetectionNet_v2(nn.Module):
    def __init__(self, cfg, num_queries=100, num_classes=1): 
        super().__init__()
        #do omegaconf
        
        #self.cfg = omegaconf.OmegaConf.create(cfg) if isinstance(cfg, dict) else cfg
        if not isinstance(cfg, EasyDict):
            cfg = EasyDict(cfg)

        self.cfg = cfg # Store full config
        self.logger = get_logger()
        self.num_queries = num_queries
        self.num_classes = num_classes 
        # --- Point Cloud Branch (PointPillarBEVExtractor) ---
        self.pc_extractor = PointPillarBEVExtractor(self.cfg.POINT_PILLARS)
        
        # The BaseBEVBackbone's output is a concatenation of upsampled features.
        num_upsample_filters = self.cfg.POINT_PILLARS.BACKBONE_2D.NUM_UPSAMPLE_FILTERS
        pc_bev_output_channels = sum(num_upsample_filters) # This correctly sums up to 384 based on your config

        final_fused_H_bev = 248 
        final_fused_W_bev = 216

        # --- Image Feature Extractor (DINOv2) ---
        self.dino_model = Dinov2() 
        self.image_to_bev_projector = ImageToBEVProjector(
            image_input_channels=self.dino_model.output_feat_size, # 768 for vitb14
            bev_output_channels=cfg.FUSION.IMAGE_BEV_CHANNELS,
            target_H_bev=final_fused_H_bev, # Align image features to PC extractor's *final* output size
            target_W_bev=final_fused_W_bev, # Align image features to PC extractor's *final* output size
            image_patch_grid_size=self.dino_model.patch_grid_shape
        )
        image_bev_output_channels = cfg.FUSION.IMAGE_BEV_CHANNELS

        # --- Multimodal Fusion Module ---
        fused_input_channels_bev = pc_bev_output_channels + image_bev_output_channels
        
        self.logger.info(f"DEBUG_INIT: fused_input_channels_bev (sum of PC and Image BEV) = {fused_input_channels_bev}")
        self.logger.info(f"DEBUG_INIT: cfg.FUSION.FUSED_CHANNELS (output channels of first conv) = {cfg.FUSION.FUSED_CHANNELS}")

        self.fusion_convs = nn.Sequential(
            nn.BatchNorm2d(fused_input_channels_bev),
            nn.Conv2d(fused_input_channels_bev, cfg.FUSION.FUSED_CHANNELS, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(cfg.FUSION.FUSED_CHANNELS),
            nn.Conv2d(cfg.FUSION.FUSED_CHANNELS, cfg.FUSION.FUSED_CHANNELS, kernel_size=3, padding=1),
            nn.ReLU(True),
        )
        
        # --- Query-Based 3D Detection Head ---
        self.prediction_head = DetectionHead(cfg.FUSION.FUSED_CHANNELS, num_queries, num_classes)

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
        fused_input = fused_input.cuda()
        device = fused_input.device
        B, C_fused, H_orig, W_orig = fused_input.shape

        # Split RGB and point cloud channels
        rgb_image = fused_input[:, :3, :, :]
        pc_projected_raw = fused_input[:, 3:, :, :]
        
        

        # 1) Image features (DINOv2)
        dino_features_2d = self.dino_model(rgb_image, train=self.training)

        # 2) Point cloud BEV features
        pc_bev_H, pc_bev_W = self.pc_extractor.grid_size[1], self.pc_extractor.grid_size[0]
        if pc_projected_raw.shape[2:] != (pc_bev_H, pc_bev_W):
            pc_projected_resized = F.interpolate(pc_projected_raw, size=(pc_bev_H, pc_bev_W),
                                                mode='bilinear', align_corners=False)
        else:
            pc_projected_resized = pc_projected_raw

        pc_batch_dict = self._prepare_pc_batch_dict(pc_projected_resized, B)
        pc_bev_features = self.pc_extractor(pc_batch_dict)

        # 3) Project image features to BEV
        image_bev_features = self.image_to_bev_projector(dino_features_2d)

        # 4) Fusion
        fused_features_bev = torch.cat([pc_bev_features, image_bev_features], dim=1)
        fused_features_bev = self.fusion_convs(fused_features_bev)

        # 5) Global pooling + prediction head
        global_fused_features = F.adaptive_avg_pool2d(fused_features_bev, (1, 1)).view(B, -1)
        pred_boxes_7d, pred_logits = self.prediction_head(global_fused_features)
        pred_boxes_compat = torch.cat([pred_boxes_7d, pred_logits[:, :, 0:1]], dim=-1)

        # # Reshape predictions: [B, Q, 7 + num_classes + 1]
        # pred_raw = pred_raw.view(B, self.num_queries, 7 + self.num_classes + 1)
        
        # pred_boxes_7d = pred_raw[:, :, :7]
        # pred_logits = pred_raw[:, :, 7:]
        # pred_boxes_compat = torch.cat([pred_boxes_7d, pred_logits[:, :, 0:1]], dim=-1)

        # Inference-only
        if bbox3d is None:
            return {'pred_boxes': pred_boxes_compat}

        # ---- Convert GT to 7D format ----
        gt_7d_boxes_list = []
        for gt_data_per_sample in bbox3d:
            gt_data_per_sample = gt_data_per_sample.to(device)
            gt_7d_boxes_list.append(gt_data_per_sample)
            #assert that dimensions are positive
            if (gt_7d_boxes_list[-1][:, 3:6] <0).any():
                print(f"Error: GT boxes have negative dimensions: {gt_7d_boxes_list[-1]}")
                raise ValueError("GT boxes have non-positive dimensions.")

        # ---- Initialize losses ----
        losses = {k: torch.tensor(0.0, device=device) for k in
                ['total_loss', 'reg_l1', 'yaw_l1', 'cls_pos', 'cls_neg', '3diouloss']}
        matched_info = [{} for _ in range(B)] # To store matched boxes and scores per sample

        # ---- Per-sample Hungarian matching & loss ----
        for b in range(B):
            pred_boxes_sample = pred_boxes_7d[b]  # (Q,7)
            pred_logits_sample = pred_logits[b]   # (Q,C+1)
            gt_boxes_sample = gt_7d_boxes_list[b] # (G,7)

            num_gt, num_preds = gt_boxes_sample.shape[0], self.num_queries
            gt_labels_sample = torch.zeros(num_gt, dtype=torch.long, device=device)

            # No GT case
            if num_gt == 0:
                target_noobj = torch.full((num_preds,), self.num_classes, dtype=torch.long, device=device)
                cls_loss = F.cross_entropy(pred_logits_sample, target_noobj)
                losses['cls_neg'] += self.loss_weights['cls_neg'] * cls_loss
                losses['total_loss'] += self.loss_weights['cls_neg'] * cls_loss
                continue

            # --- Compute cost components ---
            with torch.no_grad():
                iou_matrix = boxes_iou3d_gpu(pred_boxes_sample, gt_boxes_sample)  # (Q, G)

                # Regression L1 cost (center + size)
                reg_cost = (
                    F.l1_loss(pred_boxes_sample[:, None, 0:3], gt_boxes_sample[None, :, 0:3], reduction='none').sum(-1) +
                    F.l1_loss(pred_boxes_sample[:, None, 3:6], gt_boxes_sample[None, :, 3:6], reduction='none').sum(-1)
                )  # (Q, G)

                # Classification cost
                cls_prob = pred_logits_sample.softmax(-1)  # (Q, C+1)
                cls_cost = -cls_prob[:, 0:1].expand(-1, num_gt)  # 1-class case, class 0 is object
                
                

                # IoU cost
                iou_cost = 1 - iou_matrix  # higher IoU → lower cost
                
                #Yaw cost, take into account angle periodicity
                pred_vec = torch.stack([torch.cos(pred_boxes_sample[:, 6]), torch.sin(pred_boxes_sample[:, 6])], dim=-1)  # (Q, 2)
                yaw_cost = 1 - torch.abs(torch.matmul(pred_vec, torch.stack([torch.cos(gt_boxes_sample[:, 6]), torch.sin(gt_boxes_sample[:, 6])], dim=-1).t()))  # (Q, G)

                # Total cost
                cost_matrix = (
                    self.cost_cls_weight * cls_cost +
                    self.cost_box_weight * reg_cost +
                    self.cost_iou_weight * iou_cost +
                    self.yaw_weight * yaw_cost
                    
                ).cpu().numpy()

            # Hungarian assignment
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            row_ind = torch.as_tensor(row_ind, device=device, dtype=torch.long)  # predictions
            col_ind = torch.as_tensor(col_ind, device=device, dtype=torch.long)  # GTs

            # --- Matched classification loss ---
            cls_loss_matched = F.cross_entropy(pred_logits_sample[row_ind], gt_labels_sample[col_ind])

            # --- Unmatched predictions classification loss ---
            unmatched_mask = torch.ones(num_preds, dtype=torch.bool, device=device)
            unmatched_mask[row_ind] = False
            if unmatched_mask.any():
                cls_loss_unmatched = F.cross_entropy(
                    pred_logits_sample[unmatched_mask],
                    torch.full((unmatched_mask.sum(),), self.num_classes, device=device, dtype=torch.long)
                )
            else:
                cls_loss_unmatched = torch.tensor(0.0, device=device)

            # --- Regression losses ---
            matched_pred = pred_boxes_sample[row_ind]
            matched_gt = gt_boxes_sample[col_ind]
            
            matched_info[b] = {
                'pred_boxes': matched_pred,
                'gt_boxes': matched_gt,
                'scores': pred_logits_sample[row_ind].softmax(-1)[:,0]
            }


            reg_centers = F.l1_loss(matched_pred[:, 0:3], matched_gt[:, 0:3], reduction='mean')
            reg_sizes = F.l1_loss(matched_pred[:, 3:6], matched_gt[:, 3:6], reduction='mean')
            reg_loss_matched = reg_centers + reg_sizes

            # --- Yaw loss ---
            pred_vec = torch.stack([torch.cos(matched_pred[:, 6]), torch.sin(matched_pred[:, 6])], dim=-1)
            gt_vec = torch.stack([torch.cos(matched_gt[:, 6]), torch.sin(matched_gt[:, 6])], dim=-1)
            yaw_loss_matched = F.l1_loss(pred_vec, gt_vec, reduction='mean')

            # --- IoU loss ---
            iou_loss = 1 - boxes_iou3d_gpu(matched_pred, matched_gt).diagonal().mean()

            # --- Aggregate ---
            losses['reg_l1'] += reg_loss_matched
            losses['yaw_l1'] += yaw_loss_matched
            losses['cls_pos'] += cls_loss_matched
            losses['cls_neg'] += cls_loss_unmatched
            losses['3diouloss'] += iou_loss
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

    # @torch.no_grad()
    # def predict(self, fused_inputs, conf_thresh=0.5, iou_thresh=0.5):
    #     self.eval() 

    #     outputs = self.forward(fused_inputs, bbox3d=None) 
        
    #     pred_boxes_compat = outputs['pred_boxes'] 

    #     results = {'preds': []}

    #     for b in range(pred_boxes_compat.shape[0]):
    #         boxes_8d_sample = pred_boxes_compat[b]  
            
    #         centers = boxes_8d_sample[:, :3]
    #         dims    = boxes_8d_sample[:, 3:6]
    #         yaw     = boxes_8d_sample[:, 6]
    #         scores  = torch.sigmoid(boxes_8d_sample[:, 7])  

    #         keep = scores > conf_thresh
    #         if keep.sum() == 0:
    #             results['preds'].append(torch.empty((0, 8), device=boxes_8d_sample.device))
    #             continue
            
    #         xy_boxes = torch.cat([
    #             centers[keep, :2] - dims[keep, :2] / 2,
    #             centers[keep, :2] + dims[keep, :2] / 2
    #         ], dim=-1)  

    #         try:
    #             keep_idx = torch.ops.torchvision.nms(xy_boxes, scores[keep], iou_thresh)
    #         except AttributeError:
    #             print("Warning: torch.ops.torchvision.nms not found. Using simple top-K sort instead of NMS.")
    #             sorted_scores, sort_indices = torch.sort(scores[keep], descending=True)
    #             keep_idx = sort_indices[:min(10, len(sort_indices))] 

    #         final_boxes = boxes_8d_sample[keep][keep_idx]

    #         results['preds'].append(final_boxes)

    #     return results
