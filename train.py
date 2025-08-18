import argparse
import os
import numpy as np
import torch
import time
from torch.backends import cudnn
from mmcv.runner import init_dist
import torch.distributed as dist
import torch.multiprocessing as mp
from utils import (get_config, get_dist_info, get_logger, build_optimizer, build_scheduler,
                save_checkpoint, set_random_seed, compute_ap3d_r11, box7_to_corners, corners_to_7d)
from datasets import build_loader
from models import build_model
import datetime
from mmcv.parallel import MMDistributedDataParallel
import os.path as osp
from omegaconf import OmegaConf
from timm.utils import AverageMeter
from sklearn.metrics import r2_score
import math
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning)
import numpy as np
import math

def parse_args():
    parser = argparse.ArgumentParser('3DBB training and evaluation script')
    parser.add_argument('--cfg', type=str, required=True, help='path to config file')
    parser.add_argument('--opts', help="Modify config options by adding 'KEY=VALUE' list. ", default=None, nargs='+')
    # To overwrite config file
    parser.add_argument('--batch-size', type=int, help='batch size for single GPU')
    #parser.add_argument('--resume', default='/mnt/gsdata/projects/panops/outputs/2newdata_moretreedata/checkpoint.pth', help='resume from checkpoint')
    parser.add_argument('--resume',type=str, help='resume from checkpoint')
    parser.add_argument('--seed', type=int, default=0, help='random seed for initialization')

    parser.add_argument(
        '--output', default="/mnt/gsdata/projects/panops/plant_trait_net/outputs_v5/3dbb7_test", type=str, help='root of output folder, '
        'the full path is <output>/<model_name>/<tag>')
    parser.add_argument('--tag', type=str, help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--wandb', action='store_true', help='Use W&B to log experiments')
    parser.add_argument('--keep', type=int, help='Maximum checkpoint to keep')
    parser.add_argument('--debug', type=bool, default=False, help='tag of experiment')
    args = parser.parse_args()

    return args

def train_one_epoch(config, model, data_loader, optimizer, epoch, lr_scheduler):
    """
    Trains the model for one epoch.

    Args:
        config (dict): Configuration parameters.
        model (nn.Module): The model to be trained.
        data_loader (DataLoader): The data loader for training data.
        optimizer (Optimizer): The optimizer for updating model parameters.
        epoch (int): The current epoch number.
        lr_scheduler (LRScheduler): The learning rate scheduler.
        target_transformer (callable): A function to transform the target data.

    Returns:
        dict: A dictionary containing the average loss and other metrics for the epoch.
    """
    logger = get_logger()
    dist.barrier()
    model.train()
    optimizer.zero_grad()
    if config.wandb and dist.get_rank() == 0:
        import wandb
    else:
        wandb = None

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    
    # Create AverageMeters for each individual loss component
    reg_l1_meter = AverageMeter()
    yaw_l1_meter = AverageMeter()
    cls_pos_meter = AverageMeter()
    cls_neg_meter = AverageMeter()
    
    all_loss_meters = {
        'total_loss': loss_meter,
        'reg_l1': reg_l1_meter,
        'yaw_l1': yaw_l1_meter,
        'cls_pos': cls_pos_meter,
        'cls_neg': cls_neg_meter,
    }


    start = time.time()
    end = time.time()

    for idx, samples in enumerate(data_loader):
        logger.info(f'Iteration:{idx}')
        torch.cuda.empty_cache()
        optimizer.zero_grad()
        result = model(**samples)
        loss = result['loss']
        losses_dict = result['losses']
        # Update each individual loss meter
        for key, value_tensor in losses_dict.items():
            if key in all_loss_meters: # Only update meters that we defined
                all_loss_meters[key].update(value_tensor.item())
            logger.info(f'Loss {key}: {value_tensor.item():.4f}')
            
        loss.backward()
        
        if config.train.clip_grad:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.clip_grad)

        optimizer.step()
        lr_scheduler.step_update(epoch*num_steps + idx)

        torch.cuda.synchronize()

        
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.print_freq == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0 * 1024.0) #Memory in GB
            etas = batch_time.avg * (num_steps - idx)
            logger.info(f'Train: [{epoch}/{config.train.epochs}][{idx}/{num_steps}]\t'
                        f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                        f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                        f'total_loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                        f'mem {memory_used:.0f}MB')
            if wandb is not None:
                log_stat = {} 
                log_stat['iter/learning_rate'] = lr
                for k, meter in all_loss_meters.items():
                    log_stat[f'iter/{k}'] = meter.avg
                wandb.log(log_stat)

    epoch_time = time.time() - start
    logger.info(f'EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}')
    result_dict = dict(total_loss=loss_meter.avg)
    dist.barrier()
    return result_dict

def train(cfg):
    if cfg.wandb and dist.get_rank() == 0:
        import wandb
        wandb.init(
            project='3DBB',
            name=osp.join(cfg.model_name, cfg.tag),
            dir=cfg.output,
            config=OmegaConf.to_container(cfg, resolve=True),
            resume=cfg.checkpoint.auto_resume)
    else:
        wandb = None
    
    dist.barrier()

    # Remove the redefinition of 'logger'
    logger = get_logger()
    
    dataset_train, data_loader_train, \
        data_loader_test = build_loader(cfg.data)
    
    model = build_model(cfg.model).cuda()

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'number of params before setting gradient false: {n_parameters}')

    optimizer = build_optimizer(cfg.train, model)
    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'number of params after setting gradient false: {n_parameters}')

    model = MMDistributedDataParallel(model, broadcast_buffers=False)
    model_without_ddp = model.module

    if cfg.wandb and dist.get_rank() == 0:
        wandb.watch(model, log="all")

    lr_scheduler = build_scheduler(cfg.train, optimizer, len(data_loader_train)) 
    
    min_test_loss = math.inf
    max_ap3dr11 = -math.inf
    max_metrics = {'min_test_loss': min_test_loss, 'max_ap3dr11': max_ap3dr11}


    logger.info(f'Start training from epoch {cfg.train.start_epoch}')
    start_time = time.time()
    
    for epoch in range(cfg.train.start_epoch, cfg.train.epochs):
        torch.cuda.empty_cache()
        logger.info(f'Epoch {epoch} starts')
        loss_train_dict = train_one_epoch(cfg, model, data_loader_train, optimizer, epoch, lr_scheduler)        
        loss_train = loss_train_dict['total_loss']
        logger.info(f'Avg loss of the network on the {len(dataset_train)} train images: {loss_train:.2f}')

        if dist.get_rank() == 0 and (epoch % cfg.checkpoint.save_freq == 0 or epoch == (cfg.train.epochs - 1)):
            metrics = {'total_loss': loss_train}
            save_checkpoint(cfg, epoch, model_without_ddp,metrics,
                            optimizer, lr_scheduler, suffix=f'train_epoch_{epoch}')
        dist.barrier()
        #evaluate
        if (epoch % cfg.evaluate.eval_freq == 0 or epoch == (cfg.train.epochs - 1)):
            test_loss, ap3d_r11, all_loss_meters = validate(cfg, data_loader_test,  model)
            logger.info(f'Test loss of the network on the {len(data_loader_test.dataset)} val images: {test_loss:.2f}%')


            if 'min_test_loss' not in max_metrics:
                max_metrics['min_test_loss'] = math.inf
                max_metrics['max_ap3dr11'] = -math.inf
                
            if cfg.evaluate.save_best and dist.get_rank() == 0 and ( test_loss < max_metrics['min_test_loss']):
                if test_loss < max_metrics['min_test_loss']:
                    logger.info(f'Min test loss: {test_loss:.2f}%')
                    suffix = 'min_loss'
                max_metrics['min_test_loss'] = min(max_metrics['min_test_loss'], test_loss)
                max_metrics['max_ap3dr11'] = max(max_metrics['max_ap3dr11'], ap3d_r11)
                
                save_checkpoint(cfg, epoch, model_without_ddp, max_metrics, optimizer, lr_scheduler, suffix=suffix)
                    

            if wandb is not None:
                log_stat = {}
                log_stat.update({
                            'epoch/epoch': epoch,
                            'epoch/n_parameters': n_parameters,
                            'epoch/ap3d_r11': ap3d_r11
                })
                # Log all individual losses
                for key, meter in all_loss_meters.items():
                    log_stat[f'epoch/{key}'] = meter.avg
                wandb.log(log_stat)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))
    dist.barrier()
    
def validate(config, data_loader, model):
    logger = get_logger()
    model.eval()
    
    loss_meter = AverageMeter()
    reg_l1_meter = AverageMeter()
    yaw_l1_meter = AverageMeter()
    cls_pos_meter = AverageMeter()
    cls_neg_meter = AverageMeter()
    
    all_loss_meters = {
        'total_loss': loss_meter,
        'reg_l1': reg_l1_meter,
        'yaw_l1': yaw_l1_meter,
        'cls_pos': cls_pos_meter,
        'cls_neg': cls_neg_meter,
    }

    all_preds, all_gts = [], []

    # Get the normalization stats from the dataset
    dataset_instance = data_loader.dataset
    
    device = next(model.parameters()).device
    bbox_mean = dataset_instance.bbox_mean.to(device)
    bbox_std = dataset_instance.bbox_std.to(device)

    with torch.no_grad():
        for idx, samples in enumerate(data_loader):
            # Move all inputs to the correct device
            for key, value in samples.items():
                samples[key] = value.to(device)
            
            outputs = model(**samples)
            
            losses_dict = outputs['losses']
            for key, value_tensor in losses_dict.items():
                if key in all_loss_meters:
                    all_loss_meters[key].update(value_tensor.item())
            
            # --- Un-normalize the predictions ---
            # batch_preds has shape [B, num_queries, 8]
            batch_preds = outputs['pred_boxes'] 
            batch_preds_unnorm = batch_preds.clone()
            batch_preds_unnorm[:, :, :7] = batch_preds_unnorm[:, :, :7] * bbox_std + bbox_mean

            # --- Un-normalize the ground truths ---
            # batch_gts is a single tensor from the DataLoader, shape [B, N_max, 7]
            batch_gts_unnorm = samples['bbox3d'].clone()
            batch_gts_unnorm[:, :, :7] = batch_gts_unnorm[:, :, :7] * bbox_std + bbox_mean

            # --- Process each sample in the batch ---
            for b in range(batch_preds_unnorm.shape[0]):
                # Process predictions for the current sample
                pred_current_unnorm = batch_preds_unnorm[b]
                scores = torch.sigmoid(pred_current_unnorm[:, 7])
                
                keep = scores > 0.5
                if keep.sum() == 0:
                    final_preds_np = np.empty((0, 8), dtype=np.float32)
                else:
                    centers = pred_current_unnorm[keep, :3]
                    dims = pred_current_unnorm[keep, 3:6]
                    xy_boxes = torch.cat([centers[:, :2] - dims[:, :2] / 2,
                                        centers[:, :2] + dims[:, :2] / 2], dim=-1)
                    keep_idx = torch.ops.torchvision.nms(xy_boxes, scores[keep], 0.7)
                    final_preds_np = pred_current_unnorm[keep][keep_idx].cpu().numpy()

                all_preds.append(final_preds_np)

                # Process ground truths for the current sample
                gt_current_unnorm = batch_gts_unnorm[b]

                if gt_current_unnorm.shape[0] > 0:
                    # Filter out padded rows (all zeros)
                    non_padded_indices = (gt_current_unnorm[:, :7].abs().sum(dim=1) != 0)
                    gt_current_unpadded = gt_current_unnorm[non_padded_indices]
                    
                    if gt_current_unpadded.numel() > 0:
                        # Convert un-normalized 7-params to 8-corners (Numpy)
                        gt_corners_list = [box7_to_corners(box.cpu().numpy()) for box in gt_current_unpadded]
                        gt_corners_np = np.array(gt_corners_list)
                    else:
                        gt_corners_np = np.empty((0, 8, 3), dtype=np.float32)
                else:
                    gt_corners_np = np.empty((0, 8, 3), dtype=np.float32)

                all_gts.append(gt_corners_np)
            
            if idx % config.print_freq == 0:
                logger.info(f'Val: [{idx}/{len(data_loader)}] '
                            f'loss {all_loss_meters["total_loss"].val:.4f} ({all_loss_meters["total_loss"].avg:.4f})')

    # Compute AP3D|R11
    ap3d_r11 = compute_ap3d_r11(all_preds, all_gts, iou_thresh=0.7)
    
    logger.info(f"=> Loss {all_loss_meters['total_loss'].avg:.4f}, AP3D|R11 {ap3d_r11:.4f}")

    return all_loss_meters['total_loss'].avg, ap3d_r11, all_loss_meters

if __name__ == '__main__':
    # Parse arguments
    args = parse_args()
    is_debug = args.debug

    # Load config file
    config = get_config(args)
    logger = get_logger(config)

    #To facilitate debugging via vscode debugger by bypassing the \\
    # distributed training launcher file in shellscript
    if is_debug:
        os.environ['RANK'] = '1'
        os.environ['WORLD_SIZE'] = '2'
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29500'
    
    mp.set_start_method('fork', force=True)
    init_dist('pytorch')

    world_size, rank = get_dist_info()
    logger.info(f'RANK and WORLD_SIZE in environ: {rank}/{world_size}')

    dist.barrier()

    # Set random seed
    set_random_seed(config.seed)
    cudnn.benchmark = True
    
    
    os.makedirs(config.output, exist_ok=True)
    train(config)
    dist.barrier()