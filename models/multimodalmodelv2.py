import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict
from scipy.optimize import linear_sum_assignment
import numpy as np # For numpy usage in linear_sum_assignment
from typing import Optional

# --- OpenPCDet Modules (Ensure OpenPCDet is installed and available) ---
# Assuming these imports work after correctly installing OpenPCDet
from pcdet.models.backbones_3d.vfe.pillar_vfe import PillarVFE
from pcdet.models.backbones_2d.map_to_bev.pointpillar_scatter import PointPillarScatter
from pcdet.models.backbones_2d.base_bev_backbone import BaseBEVBackbone
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
            print("Please ensure you have an internet connection. Falling back to dummy DINOv2 model.")
            self.dinomodel = nn.Sequential(
                nn.Conv2d(3, 768, kernel_size=14, stride=14),
                nn.Flatten(2),
                nn.Permute(0, 2, 1)
            )
            class MockDinoConfig:
                patch_size = 14
                embed_dim = 768
            self.dinomodel.config = MockDinoConfig()

        for param in self.dinomodel.parameters():
                param.requires_grad = False
        print(f"DINOv2 model parameters frozen.")

        self.output_feat_size = self.dinomodel.embed_dim 
        dino_patch_size = self.dinomodel.patch_size 
        
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

# --- Multimodal Fusion Detection Network ---
@MODELS.register_module() 
class MultimodalDetectionNet_v2(nn.Module):
    def __init__(self, cfg, num_queries=50, num_classes=1): 
        super().__init__()
        #do omegaconf
        
        #self.cfg = omegaconf.OmegaConf.create(cfg) if isinstance(cfg, dict) else cfg
        if not isinstance(cfg, EasyDict):
            cfg = EasyDict(cfg)

        self.cfg = cfg # Store full config
        self.logger = get_logger()
        self.num_queries = num_queries
        self.num_classes = num_classes 

        # --- RGB Image Branch (ResNet50) ---
        # Note: ResNet50 is not used here as DINOv2 provides image features.
        # This 'rgb_backbone' might be a remnant from a previous design.
        # If you intend to use it, you'd need to adapt how its features are fused.
        # For current design, DINOv2 is the primary image feature extractor.
        self.rgb_backbone = nn.Identity() # Placeholder if not used, or remove if truly not needed.
        # Original: self.rgb_backbone = resnet50(pretrained=True); self.rgb_backbone.avgpool = nn.Identity(); self.rgb_backbone.fc = nn.Identity()

        # --- Point Cloud Branch (PointPillarBEVExtractor) ---
        self.pc_extractor = PointPillarBEVExtractor(self.cfg.POINT_PILLARS)
        # Calculate actual output channels for pc_extractor's BaseBEVBackbone
        # The BaseBEVBackbone's output is a concatenation of upsampled features.
        num_upsample_filters = self.cfg.POINT_PILLARS.BACKBONE_2D.NUM_UPSAMPLE_FILTERS
        pc_bev_output_channels = sum(num_upsample_filters) # This correctly sums up to 384 based on your config

        # --- Determine the final fusion resolution ---
        # Based on your debug output: PC BEV Features Shape: [4, 384, 248, 216]
        # This means the target H and W for fusion are 248 (Height) and 216 (Width).
        final_fused_H_bev = 248 
        final_fused_W_bev = 216

        # --- Image Feature Extractor (DINOv2) ---
        self.dino_model = Dinov2() 
        self.image_to_bev_projector = ImageToBEVProjector(
            image_input_channels=self.dino_model.output_feat_size, 
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
            nn.Conv2d(fused_input_channels_bev, cfg.FUSION.FUSED_CHANNELS, kernel_size=3, padding=1),
            nn.BatchNorm2d(cfg.FUSION.FUSED_CHANNELS),
            nn.ReLU(True),
            nn.Conv2d(cfg.FUSION.FUSED_CHANNELS, cfg.FUSION.FUSED_CHANNELS, kernel_size=3, padding=1),
            nn.BatchNorm2d(cfg.FUSION.FUSED_CHANNELS),
            nn.ReLU(True),
        )
        # Explicit print after fusion_convs is defined
        self.logger.info(f"DEBUG_INIT: self.fusion_convs[0].in_channels = {self.fusion_convs[0].in_channels}")
        
        # --- Query-Based 3D Detection Head ---
        self.prediction_head = nn.Sequential(
            nn.Linear(cfg.FUSION.FUSED_CHANNELS, 256), 
            nn.ReLU(True),
            nn.Linear(256, self.num_queries * (7 + self.num_classes + 1)) 
        )

        # Loss weights and cost weights for Hungarian matching
        self.loss_weights = {
            'reg_l1': 10.0, 'yaw_l1': 1.0, 'cls_pos': 1.0, 'cls_neg': 1.0
        }
        self.cost_cls_weight = 1.0
        self.cost_box_weight = 1.0
        self.cost_yaw_weight = 0.1 

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
        Performs a forward pass through the multimodal 3D detection network.

        Args:
            fused_input (torch.Tensor): Combined input (B, 6, H, W) where channels 0-2 are RGB
                                        and channels 3-5 are projected point cloud (x,y,z).
            bbox3d (list of torch.Tensor, optional): Ground truth 3D bounding boxes.
                                  Each tensor in the list is (num_gt_objects, 8, 3) for one batch item.
                                  If None, performs inference only.
        
        Returns:
            dict: Dictionary containing predicted boxes and, during training, the computed losses.
        """
        B, C_fused, H_orig, W_orig = fused_input.shape
        device = fused_input.device

        rgb_image = fused_input[:, :3, :, :]
        pc_projected_raw = fused_input[:, 3:, :, :]  

        # 1. Image Feature Extraction (DINOv2)
        if rgb_image.shape[2] != self.dino_model.dino_input_H or rgb_image.shape[3] != self.dino_model.dino_input_W:
            rgb_image_resized = F.interpolate(rgb_image, 
                                              size=(self.dino_model.dino_input_H, self.dino_model.dino_input_W), 
                                              mode='bilinear', align_corners=False)
        else:
            rgb_image_resized = rgb_image
            
        dino_features_2d = self.dino_model(rgb_image_resized.cuda(), train=self.training)
        self.logger.debug(f"DINOv2 Features Shape: {dino_features_2d.shape}")

        # 2. Point Cloud Feature Extraction (PointPillarBEVExtractor)
        pc_bev_H_initial = self.pc_extractor.grid_size[1] # H of initial BEV grid (e.g., 496)
        pc_bev_W_initial = self.pc_extractor.grid_size[0] # W of initial BEV grid (e.g., 432)

        if pc_projected_raw.shape[2] != pc_bev_H_initial or pc_projected_raw.shape[3] != pc_bev_W_initial:
            pc_projected_resized = F.interpolate(pc_projected_raw, 
                                                size=(pc_bev_H_initial, pc_bev_W_initial), 
                                                mode='bilinear', align_corners=False)
        else:
            pc_projected_resized = pc_projected_raw

        pc_batch_dict_for_extractor = self._prepare_pc_batch_dict(pc_projected_resized, B)
        pc_bev_features = self.pc_extractor(pc_batch_dict_for_extractor)
        self.logger.debug(f"PC BEV Features Shape: {pc_bev_features.shape}") 

        # 3. Project Image Features to BEV Space (Learned Alignment)
        image_bev_features = self.image_to_bev_projector(dino_features_2d)
        self.logger.debug(f"Image BEV Features Shape: {image_bev_features.shape}") 

        # 4. Multimodal Fusion - Now shapes should match
        fused_features_bev = torch.cat([pc_bev_features, image_bev_features], dim=1)
        self.logger.debug(f"Fused Features Shape: {fused_features_bev.shape}")
        fused_features_bev = self.fusion_convs(fused_features_bev)

        # 5. Global Pooling and Prediction Head
        global_fused_features = F.adaptive_avg_pool2d(fused_features_bev, (1, 1)).view(B, -1)
        
        pred_raw = self.prediction_head(global_fused_features)
        
        pred_raw = pred_raw.view(B, self.num_queries, (7 + self.num_classes + 1))
        
        pred_boxes_7d = pred_raw[:, :, :7] 
        pred_logits   = pred_raw[:, :, 7:] 

        pred_boxes_compat = torch.cat([pred_boxes_7d, pred_logits[:, :, 0:1]], dim=-1) 

        # --- Loss Calculation (ONLY if bbox3d is provided) ---
        if bbox3d is None: 
            return {'pred_boxes': pred_boxes_compat} 

        # Convert GT corners to 7D format for all batch items (if not already done)
        gt_7d_boxes_list = []
        for gt_data_per_sample in bbox3d: # bbox3d is a list of (N_i, 8, 3) or (N_i, 7) tensors
            gt_data_per_sample = gt_data_per_sample.to(device) # Ensure GT is on device
            
            if gt_data_per_sample.shape[0] == 0:
                gt_7d_boxes_list.append(torch.empty((0, 7), dtype=gt_data_per_sample.dtype, device=device))
            # Check if it's already in (N, 7) format
            elif gt_data_per_sample.ndim == 2 and gt_data_per_sample.shape[1] == 7:
                self.logger.debug(f"GT already in 7D format. Shape: {gt_data_per_sample.shape}")
                gt_7d_boxes_list.append(gt_data_per_sample)
            # Assume it's in (N, 8, 3) corners format, then convert
            elif gt_data_per_sample.ndim == 3 and gt_data_per_sample.shape[1] == 8 and gt_data_per_sample.shape[2] == 3:
                self.logger.debug(f"Converting {gt_data_per_sample.shape} GT corners to 7D format.")
                # Apply _convert_corners_to_7d to each object's 8 corners and stack
                converted_7d_boxes = torch.stack([self.corners_to_7d(obj_corners) for obj_corners in gt_data_per_sample])
                gt_7d_boxes_list.append(converted_7d_boxes)
            else:
                raise ValueError(f"Unexpected GT bounding box format. Expected (N, 8, 3) or (N, 7), got {gt_data_per_sample.shape}")

        losses = {
            'total_loss': torch.tensor(0.0, device=device), 'reg_l1': torch.tensor(0.0, device=device),
            'yaw_l1': torch.tensor(0.0, device=device), 'cls_pos': torch.tensor(0.0, device=device),
            'cls_neg': torch.tensor(0.0, device=device),
        }
        
        for b in range(B):
            pred_boxes_sample = pred_boxes_7d[b] 
            pred_logits_sample = pred_logits[b] 
            
            self.logger.debug(f"DEBUG: Sample {b} - pred_boxes_sample shape: {pred_boxes_sample.shape}, pred_logits_sample shape: {pred_logits_sample.shape}")
            
            # Get the ALREADY converted 7D GT boxes for the current sample
            gt_boxes_sample = gt_7d_boxes_list[b] # This is now (N_gt_objects_sample, 7)
            self.logger.debug(f"DEBUG: Sample {b} - gt_boxes_sample shape: {gt_boxes_sample.shape}")
            
            # Define gt_labels_sample for the current batch item
            gt_labels_sample = torch.zeros(gt_boxes_sample.shape[0], dtype=torch.long, device=pred_logits_sample.device)

            num_gt = gt_boxes_sample.shape[0]
            num_preds = self.num_queries

            if num_gt == 0:
                target_cls_for_loss = torch.full((num_preds,), self.num_classes, dtype=torch.long, device=pred_logits_sample.device)
                cls_loss = F.cross_entropy(pred_logits_sample, target_cls_for_loss)
                losses['cls_neg'] += self.loss_weights['cls_neg'] * cls_loss
                losses['total_loss'] += self.loss_weights['cls_neg'] * cls_loss
                continue
            
            cost_cls = F.cross_entropy(pred_logits_sample.repeat(num_gt, 1, 1).flatten(0,1), 
                                    gt_labels_sample.repeat_interleave(num_preds), reduction='none').view(num_gt, num_preds)
            
            # This line will now work correctly as both inputs have 6 columns
            # This line will now work correctly as both inputs have 6 columns
            print(f"DEBUG: Sample {b} - pred_boxes_sample shape: {pred_boxes_sample.shape}, gt_boxes_sample shape: {gt_boxes_sample.shape}")
            gt_boxes_sample = gt_boxes_sample.to(pred_boxes_sample.device) # Ensure GT boxes are on the same device
            cost_box = torch.cdist(pred_boxes_sample[:, :6], gt_boxes_sample[:, :6], p=1)
            # Transpose cost_box to match (num_gt, num_preds) format for element-wise addition
            cost_box = cost_box.transpose(0, 1) # Now (num_gt, num_preds)

            pred_yaw = pred_boxes_sample[:, 6]
            gt_yaw = gt_boxes_sample[:, 6]
            
            pred_yaw_expanded = pred_yaw.unsqueeze(0).repeat(num_gt, 1) 
            gt_yaw_expanded = gt_yaw.unsqueeze(1).repeat(1, num_preds)   
            
            diff_yaw = torch.abs(pred_yaw_expanded - gt_yaw_expanded)
            cost_yaw = torch.min(diff_yaw, 2 * np.pi - diff_yaw) 

            cost_matrix = self.cost_cls_weight * cost_cls + \
                          self.cost_box_weight * cost_box + \
                          self.cost_yaw_weight * cost_yaw
            
            row_ind, col_ind = linear_sum_assignment(cost_matrix.detach().cpu().numpy())
            
            row_ind = torch.from_numpy(row_ind).long().to(pred_logits_sample.device)
            col_ind = torch.from_numpy(col_ind).long().to(pred_logits_sample.device)

            cls_loss_matched = F.cross_entropy(pred_logits_sample[col_ind], gt_labels_sample[row_ind])
            # Separate regression loss for box parameters (x,y,z,dx,dy,dz) and yaw
            reg_loss_matched = F.smooth_l1_loss(
                pred_boxes_sample[col_ind][:, :6], # Only x,y,z,dx,dy,dz
                gt_boxes_sample[row_ind][:, :6],
                reduction='mean'
            )
            yaw_loss_matched = F.smooth_l1_loss(
                pred_boxes_sample[col_ind][:, 6], # Only yaw
                gt_boxes_sample[row_ind][:, 6],
                reduction='mean'
            )
            
            unmatched_preds_mask = torch.ones(num_preds, dtype=torch.bool, device=pred_logits_sample.device)
            unmatched_preds_mask[col_ind] = False 

            if unmatched_preds_mask.sum() > 0:
                cls_loss_unmatched = F.cross_entropy(
                    pred_logits_sample[unmatched_preds_mask], 
                    torch.full((unmatched_preds_mask.sum(),), self.num_classes, dtype=torch.long, device=pred_logits_sample.device)
                )
            else:
                cls_loss_unmatched = torch.tensor(0.0, device=pred_logits_sample.device)
            
            cls_loss_sample = cls_loss_matched + cls_loss_unmatched

            losses['reg_l1'] += reg_loss_matched.to(device)
            losses['yaw_l1'] += yaw_loss_matched.to(device) # Ensure yaw loss is accumulated
            losses['cls_pos'] += cls_loss_matched.to(device)
            losses['cls_neg'] += cls_loss_unmatched.to(device)
            
            sample_total_loss = (
                self.loss_weights['reg_l1'] * reg_loss_matched +
                self.loss_weights['yaw_l1'] * yaw_loss_matched + # Use the actual yaw loss here
                self.loss_weights['cls_pos'] * cls_loss_matched +
                self.loss_weights['cls_neg'] * cls_loss_unmatched
            )
            losses['total_loss'] += sample_total_loss.to(device)


        for k in losses:
            losses[k] /= B

        return {'loss': losses['total_loss'], 'pred_boxes': pred_boxes_compat, 'losses': losses}

    @torch.no_grad()
    def predict(self, fused_inputs, conf_thresh=0.5, iou_thresh=0.5):
        self.eval() 

        outputs = self.forward(fused_inputs, bbox3d=None) 
        
        pred_boxes_compat = outputs['pred_boxes'] 

        results = {'preds': []}

        for b in range(pred_boxes_compat.shape[0]):
            boxes_8d_sample = pred_boxes_compat[b]  
            
            centers = boxes_8d_sample[:, :3]
            dims    = boxes_8d_sample[:, 3:6]
            yaw     = boxes_8d_sample[:, 6]
            scores  = torch.sigmoid(boxes_8d_sample[:, 7])  

            keep = scores > conf_thresh
            if keep.sum() == 0:
                results['preds'].append(torch.empty((0, 8), device=boxes_8d_sample.device))
                continue
            
            xy_boxes = torch.cat([
                centers[keep, :2] - dims[keep, :2] / 2,
                centers[keep, :2] + dims[keep, :2] / 2
            ], dim=-1)  

            try:
                keep_idx = torch.ops.torchvision.nms(xy_boxes, scores[keep], iou_thresh)
            except AttributeError:
                print("Warning: torch.ops.torchvision.nms not found. Using simple top-K sort instead of NMS.")
                sorted_scores, sort_indices = torch.sort(scores[keep], descending=True)
                keep_idx = sort_indices[:min(10, len(sort_indices))] 

            final_boxes = boxes_8d_sample[keep][keep_idx]

            results['preds'].append(final_boxes)

        return results

# if __name__ == "__main__":
#     cfg_multimodal = EasyDict({
#         'POINT_PILLARS': {
#             'VFE': {
#                 'USE_NORM': True, 'NUM_POINT_FEATURES': 3, 
#                 'NUM_FILTERS': [64],
#                 'WITH_DISTANCE': False, 'USE_ABSLOTE_XYZ': True
#             },
#             'MAP_TO_BEV': {
#                 'NUM_BEV_FEATURES': 64 
#             },
#             'BACKBONE_2D': {
#                 'USE_CONV_FOR_FIRST_MODULE': True, 
#                 'LAYER_NUMS': [3, 5, 5], 
#                 'LAYER_STRIDES': [2, 2, 2], 
#                 'NUM_FILTERS': [64, 128, 256], 
#                 'UPSAMPLE_STRIDES': [1, 2, 4], 
#                 'NUM_UPSAMPLE_FILTERS': [128, 128, 128], 
#                 'INPUT_FEATURES': 64 
#             },
#             'POINT_CLOUD_RANGE': [0, -39.68, -3, 69.12, 39.68, 1], 
#             'VOXEL_SIZE': [0.16, 0.16, 4.0] 
#         },
#         'DINO_FEATURES': {
#             'MODEL_NAME': "facebook/dinov2-base", 
#             'FREEZE': True, 
#             'IMAGE_INPUT_SIZE': (504, 504) 
#         },
#         'FUSION': {
#             'IMAGE_BEV_CHANNELS': 256, 
#             'FUSED_CHANNELS': 512 
#         },
#         'MODEL': { 
#             'NUM_CLASSES': 1 
#         }
#     })

#     model = MultimodalDetectionNet(cfg_multimodal, num_queries=50, num_classes=cfg_multimodal.MODEL.NUM_CLASSES)
    
#     batch_size = 4 
#     H_img_pc = 504 
#     W_img_pc = 504 

#     dummy_fused_input = torch.randn(batch_size, 6, H_img_pc, W_img_pc)
    
#     dummy_bbox3d_gt_corners = [
#         torch.randn(2, 8, 3), 
#         torch.randn(1, 8, 3), 
#         torch.randn(0, 8, 3), 
#         torch.randn(4, 8, 3)  
#     ]
    
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model.to(device)
#     dummy_fused_input = dummy_fused_input.to(device)

#     for i in range(len(dummy_bbox3d_gt_corners)):
#         dummy_bbox3d_gt_corners[i] = dummy_bbox3d_gt_corners[i].to(device)
    
#     print(f"Running MultimodalDetectionNet on {device}...")
    
#     model.train() 
#     outputs_train = model(dummy_fused_input, bbox3d=dummy_bbox3d_gt_corners)
    
#     print("\n--- MultimodalDetectionNet Training Output ---")
#     print(f"Total Loss: {outputs_train['loss'].item()}")
#     print(f"Predicted Boxes (8D Compat) Shape: {outputs_train['pred_boxes'].shape}")
#     print(f"Detailed Losses: {outputs_train['losses']}")

#     model.eval() 
#     with torch.no_grad():
#         predictions_final = model.predict(dummy_fused_input) 
    
#     print("\n--- MultimodalDetectionNet Inference Output (first batch item) ---")
#     if predictions_final['preds']:
#         print(f"Detected Boxes 8D Shape: {predictions_final['preds'][0].shape}")
#         print(f"Example Detection (first box): {predictions_final['preds'][0][0] if predictions_final['preds'][0].shape[0] > 0 else 'N/A'}")
#     else:
#         print("No detections in first batch item after thresholding and mock NMS.")
