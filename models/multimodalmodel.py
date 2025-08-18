import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet50
from scipy.optimize import linear_sum_assignment
from .builder import MODELS  
from utils import corners_to_7d , get_logger

# --- Multimodal Fusion Detection Network ---
@MODELS.register_module()
class MultimodalDetectionNet(nn.Module):
    def __init__(self, num_queries=50):
        super().__init__()
        self.logger = get_logger()
        self.num_queries = num_queries
        #For RGB
        self.rgb_backbone = resnet50(pretrained=True)
        self.rgb_backbone.fc = nn.Identity()
        
        #For Point Cloud
        self.pc_backbone = resnet50(pretrained=False) # Start from scratch
        self.pc_backbone.fc = nn.Identity()
        
        self.fusion_head = nn.Sequential(
            nn.Linear(2048 + 2048, 2048),  # Sum of channels from both backbones
            nn.ReLU(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_queries * 8)
        )

    def forward(self, fused_input, bbox3d=None):
        
        fused_inputs = fused_input # Shape: (B, 6, H, W)
        gt_boxes_batched = bbox3d # Shape: (B, N_i, 7) - where N_i can vary or be 0 for each sample

        B = fused_inputs.shape[0]
        
        
        device = self.fusion_head[0].weight.device
        fused_inputs = fused_inputs.to(device)
        gt_boxes_batched = gt_boxes_batched.to(device) # Move GT to device
        
        # Split fused inputs into RGB and Point Cloud channels
        img = fused_inputs[:, :3, :, :] # RGB channels
        pc = fused_inputs[:, 3:, :, :]  # Point Cloud channels
        
        rgb_feat = self.rgb_backbone(img)
        pc_feat = self.pc_backbone(pc)

        # Concatenate features
        fused_feat = torch.cat([rgb_feat, pc_feat], dim=1)
        
        # Pass fused features to the final head
        pred_boxes = self.fusion_head(fused_feat).view(B, self.num_queries, 8) # (B, num_queries, 8)

        # If bbox3d is None, 
        #check if bbox3d is None, then return predictions only
        if bbox3d is not None:
            # Initialize loss trackers
            losses = {
                'total_loss': torch.tensor(0.0, device=device),
                'reg_l1': torch.tensor(0.0, device=device),
                'yaw_l1': torch.tensor(0.0, device=device),
                'cls_pos': torch.tensor(0.0, device=device),
                'cls_neg': torch.tensor(0.0, device=device),
            }
            
            #loss weights
            loss_weights = {
                'reg_l1': 10.0,
                'yaw_l1': 1.0,
                'cls_pos': 1.0, # Increased weight for positive class due to imbalance
                'cls_neg': 1.0
            }

            for b in range(B): # Loop over batch dimension
                pred = pred_boxes[b]        # (num_queries, 8) for current sample
                gt_current_sample = gt_boxes_batched[b] # (N_gt_boxes_for_this_sample, 7)

                # If there are no ground truth boxes for this sample
                if gt_current_sample.shape[0] == 0:
                    self.logger.warning(f'No ground truth boxes for sample {b}. Using unmatched predictions only.')
                    unmatched_pred_scores = pred[:, 7] # All predictions are unmatched
                    score_loss_neg = F.binary_cross_entropy_with_logits(
                        unmatched_pred_scores, torch.zeros_like(unmatched_pred_scores)
                    )
                    losses['cls_neg'] += loss_weights['cls_neg'] * score_loss_neg
                    losses['total_loss'] += loss_weights['cls_neg'] * score_loss_neg
                    continue # Skip to the next sample in the batch

                # Find the optimal assignment using the Hungarian algorithm
                # Use pred[:, :6] (center+dims) for cost, and detach for graph
                cost_matrix = torch.cdist(pred[:, :6], gt_current_sample[:, :6], p=0.2).detach().cpu().numpy()
                row_ind, col_ind = linear_sum_assignment(cost_matrix)

                # Get matched predictions and ground truths
                matched_pred = pred[row_ind]        # (num_matched_gt, 8)
                matched_gt   = gt_current_sample[col_ind] # (num_matched_gt, 7)
                
                self.logger.info(f'Sample {b}: Matched {len(row_ind)} predictions with ground truths.')
                
                # Find unmatched prediction indices
                matched_indices = torch.tensor(row_ind, device=device)
                
                # Create a boolean mask for unmatched predictions
                unmatched_mask = torch.ones(self.num_queries, dtype=torch.bool, device=device)
                if matched_indices.numel() > 0: # Ensure matched_indices is not empty before indexing
                    unmatched_mask[matched_indices] = False
                unmatched_pred_scores = pred[unmatched_mask, 7]

                # 1. Regression Loss on matched predictions
                reg_loss = F.l1_loss(matched_pred[:, :6], matched_gt[:, :6])
                yaw_loss = F.l1_loss(matched_pred[:, 6], matched_gt[:, 6])
                
                # 2. Classification Loss on matched predictions (target=1)
                score_loss_pos = F.binary_cross_entropy_with_logits(
                    matched_pred[:, 7], torch.ones_like(matched_pred[:, 7])
                )

                # 3. Classification Loss on UNMATCHED predictions (target=0)
                score_loss_neg = F.binary_cross_entropy_with_logits(
                    unmatched_pred_scores, torch.zeros_like(unmatched_pred_scores)
                )
                
                # Sum all weighted losses for the current sample
                sample_loss = (
                    loss_weights['reg_l1'] * reg_loss +
                    loss_weights['yaw_l1'] * yaw_loss +
                    loss_weights['cls_pos'] * score_loss_pos +
                    loss_weights['cls_neg'] * score_loss_neg
                )
                
                losses['reg_l1'] += reg_loss
                losses['yaw_l1'] += yaw_loss
                losses['cls_pos'] += score_loss_pos
                losses['cls_neg'] += score_loss_neg
                losses['total_loss'] += sample_loss

            # Average losses over the batch
            for k in losses:
                losses[k] /= B

            return {'loss': losses['total_loss'], 'pred_boxes': pred_boxes, 'losses': losses}
        else:
            # Inference mode
            return {'pred_boxes': pred_boxes, 'losses': losses if 'losses' in locals() else None}

    @torch.no_grad()
    def predict(self, fused_inputs, conf_thresh=0.5, iou_thresh=0.5):
        out = self.forward(fused_inputs, bbox3d=None)
        pred_boxes = out['pred_boxes']
        B = pred_boxes.shape[0]

        results = {'preds': []}

        for b in range(B):
            boxes8d = pred_boxes[b]  # [num_queries, 8]
            centers = boxes8d[:, :3]
            dims    = boxes8d[:, 3:6]
            yaw     = boxes8d[:, 6]
            scores  = torch.sigmoid(boxes8d[:, 7])  # objectness score

            keep = scores > conf_thresh
            if keep.sum() == 0:
                results['preds'].append(torch.empty((0, 8), device=boxes8d.device))
                continue

            # Convert to 2D boxes for NMS
            xy_boxes = torch.cat([
                centers[keep, :2] - dims[keep, :2] / 2,
                centers[keep, :2] + dims[keep, :2] / 2
            ], dim=-1)  # [N,4]

            keep_idx = torch.ops.torchvision.nms(xy_boxes, scores[keep], iou_thresh)
            final_boxes = boxes8d[keep][keep_idx]

            results['preds'].append(final_boxes)

        return results
