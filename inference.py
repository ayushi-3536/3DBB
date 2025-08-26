import argparse
import os
import numpy as np
import torch
import time
import joblib
from utils import (get_config, get_logger, load_checkpoint, average_precision_3d)
from datasets import build_inference_loader
from models import build_model
from torch.backends import cudnn
from timm.utils import AverageMeter
import pandas as pd

logger = get_logger()

def parse_args():
    parser = argparse.ArgumentParser('Evaluation Pipeline')
    parser.add_argument('--cfg', type=str, required=True, help='path to config file')
    parser.add_argument('--opts', help="Modify config options by adding 'KEY VALUE' pairs.", default=None, nargs='+')
    
    parser.add_argument('--output_dir', type=str, help='Directory containing all checkpoints')
    #add skip existing
    parser.add_argument('--skip_existing', type=str, default='False', help='Skip existing output directories')
    parser.add_argument('--tag', type=str, help='tag of experiment')
    #add resume
    
    parser.add_argument('--seed', type=int, default=0, help='random seed for initialization')
    parser.add_argument('--resume', type=str, default='', help='path to checkpoint to resume from')
    parser.add_argument('--path_imgref_test',type=str, help='path to test image reference file if overwriting the config file')

    #add trait
    parser.add_argument('--trait', type=str, default=None, help='Trait to evaluate')
    args = parser.parse_args()
    return args

def inference_for_all(cfg, output_path, checkpoint):
    logger = get_logger()

    dataset, data_loader = build_inference_loader(cfg.data)
    logger.info(f'Evaluating dataset: {dataset}')
    logger.info(f'Creating model:{cfg.model.type}/{cfg.model_name}')
    model = build_model(cfg.model)
    model.cuda()
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'number of params: {n_parameters}')
    print(f'checkpoints to evaluate: {checkpoint}')

    cfg = get_config(args)
    logger.info(f'Inference for checkpoint: {checkpoint}')
    max_metrics = load_checkpoint(cfg, model, None, None)
    logger.info(f'Max metrics: {max_metrics}')
    validate(config=cfg, data_loader=data_loader, model=model)
@torch.no_grad()
def validate(config, data_loader, model):
    logger = get_logger()
    model.eval()
    
    loss_meter = AverageMeter()
    reg_l1_meter = AverageMeter()
    yaw_l1_meter = AverageMeter()
    cls_pos_meter = AverageMeter()
    cls_neg_meter = AverageMeter()
    iou3d_meter = AverageMeter()
    
    all_loss_meters = {
        'total_loss': loss_meter,
        '3diouloss': iou3d_meter,
        'reg_l1': reg_l1_meter,
        'yaw_l1': yaw_l1_meter,
        'cls_pos': cls_pos_meter,
        'cls_neg': cls_neg_meter,
    }

    all_preds, all_gts = [], []
    all_aps = []
    device = next(model.parameters()).device
    print(f'length of data loader: {len(data_loader)}')
    with torch.no_grad():
        for idx, samples in enumerate(data_loader):
            print(f'Processing sample {idx}')
            # Move all inputs to the correct device
            for key, value in samples.items():
                samples[key] = value.to(device)
            
            outputs = model(**samples)

            # --- Process each sample in the batch ---
            batch_preds = outputs['pred_boxes']  # [B, num_queries, 8]
            batch_gt_boxes = samples['bbox3d']   # list of GT tensors
            batch_matched = outputs.get('matched_info', None)  # Optional matched indices
            
            for b in range(batch_preds.shape[0]):
                if batch_matched is not None:
                    pred_current = batch_matched[b]['pred_boxes']
                    scores = batch_matched[b]['scores']
                    #print min max scores
                    logger.info(f'Sample {b} - Prediction scores range: min {scores.min().item():.4f}, max {scores.max().item():.4f}')
                    scores = torch.sigmoid(scores)
                    #after sigmoid
                    logger.info(f'Sample {b} - Prediction scores after sigmoid range: min {scores.min().item():.4f}, max {scores.max().item():.4f}')
                else:
                    pred_current = batch_preds[b]  # (num_queries, 8)
                    # Prediction scores and thresholding
                    scores = torch.sigmoid(pred_current[:, 7])
                    

                #SIZE OF MATCHED PRED
                logger.info(f'Sample {b} - Number of matched predictions: {pred_current.shape[0]}')
                
                gt_current = batch_gt_boxes[b]  # (num_gt, 7)

                # Filter out zero-padded GTs
                non_padded_indices = (gt_current[:, :7].abs().sum(dim=1) != 0)
                gt_boxes_sample = gt_current[non_padded_indices] if non_padded_indices.any() else torch.empty((0,7), device=device)

                
                keep = scores > 0.0  # keep all for AP ranking
                pred_boxes_sample = pred_current[keep, :7]
                pred_scores_sample = scores[keep]
                
                #log range of scores
                logger.info(f'Sample {b} scores range: min {pred_scores_sample.min().item():.4f}, max {pred_scores_sample.max().item():.4f}')
                logger.info(f'Sample {b} - Number of predictions after thresholding: {pred_boxes_sample.shape[0]}')
                logger.info(f'Sample {b} - Number of GT boxes: {gt_boxes_sample.shape[0]}')
                

                # Compute per-sample binary 3D AP
                if len(pred_boxes_sample) == 0 or len(gt_boxes_sample) == 0:
                    ap_sample = 0.0
                else:
                    ap_sample = average_precision_3d(pred_boxes_sample, pred_scores_sample, gt_boxes_sample, iou_threshold=0.1)
                all_aps.append(ap_sample)

                # Store predictions and GTs for logging or further metrics
                all_preds.append(pred_current[keep].cpu().numpy())
                all_gts.append(gt_boxes_sample.cpu().numpy())
                
            # --- Update loss meters ---
            losses_dict = outputs['losses']
            for key, value_tensor in losses_dict.items():
                if key in all_loss_meters:
                    all_loss_meters[key].update(value_tensor.item())
            
            if idx % config.print_freq == 0:
                logger.info(f'Val: [{idx}/{len(data_loader)}] '
                            f'loss {all_loss_meters["total_loss"].val:.4f} ({all_loss_meters["total_loss"].avg:.4f})')

    # Compute mean AP over all samples
    mean_ap = sum(all_aps) / len(all_aps) if all_aps else 0.0
    logger.info(f"=> Loss {all_loss_meters['total_loss'].avg:.4f}, Binary 3D AP {mean_ap:.4f}")

    return all_loss_meters['total_loss'].avg, mean_ap, all_loss_meters



if __name__ == '__main__':
    args = parse_args()
    cudnn.benchmark = True
    # Set random seed
    config = get_config(args)
    checkpoints = args.resume
    inference_for_all(config, args.output_dir, checkpoints)

