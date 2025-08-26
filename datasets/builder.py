# dataloader_builder.py
# This file builds dataloaders for the dataset

import warnings
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from utils import parse_dataset, get_logger
from .dataset import PointCloudDataset


def warn_and_continue(exn):
    """Ignore exceptions, issue a warning, and continue."""
    warnings.warn(repr(exn))
    return True

def collate_fn(batch):
    """
    Custom collate function to handle variable number of bounding boxes per sample.
    It will correctly stack all tensors, including empty ones.
    """
    # Initialize lists to hold the tensors for each key
    fused_inputs = []
    bbox3d_list = []

    # Separate the data from the list of dictionaries
    for item in batch:
        fused_inputs.append(item['fused_input'])
        bbox3d_list.append(item['bbox3d'])

    # Stack the items that have a consistent shape
    fused_inputs_batched = torch.stack(fused_inputs, dim=0)

    # --- Manually handle stacking of bbox3d ---
    # This is the crucial part that fixes the error
    
    # Pad or stack the bounding boxes to a single tensor
    # Find the maximum number of bounding boxes in this batch
    max_num_boxes = max(b.shape[0] for b in bbox3d_list)

    if max_num_boxes == 0:
        # If the whole batch has no boxes, return an empty tensor
        bboxes_batched = torch.empty((len(batch), 0, bbox3d_list[0].shape[1]), dtype=torch.float32)
    else:
        # Pad each tensor to the max size with zeros
        padded_bboxes = []
        for b in bbox3d_list:
            if b.shape[0] < max_num_boxes:
                padding = torch.zeros(max_num_boxes - b.shape[0], b.shape[1], device=b.device)
                padded_bboxes.append(torch.cat([b, padding], dim=0))
            else:
                padded_bboxes.append(b)
        
        # Stack the padded tensors
        bboxes_batched = torch.stack(padded_bboxes, dim=0)

    # Return the batched dictionary
    return {
        'fused_input': fused_inputs_batched,
        'bbox3d': bboxes_batched
    }


def build_dataset(dataframe):
    return PointCloudDataset(dataframe=dataframe)


def build_loader(config):
    """
    Build train/val datasets and dataloaders.
    """
    logger = get_logger()

    # Parse train/val splits
    train_df, val_df = parse_dataset(config.meta)

    dataset_train = build_dataset(train_df)
    logger.info(f"Successfully built train dataset: {len(dataset_train)} samples")

    dataset_val = build_dataset(val_df)
    logger.info(f"Successfully built val dataset: {len(dataset_val)} samples")

    # Simple samplers (not distributed)
    sampler_train = RandomSampler(dataset_train, replacement=False)
    sampler_val = SequentialSampler(dataset_val)

    # Train dataloader
    data_loader_train = DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=config.batch_size,
        num_workers=config.num_workers if hasattr(config, "num_workers") else 8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        collate_fn=collate_fn
    )

    # Val dataloader
    data_loader_val = DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=1,
        num_workers=config.num_workers_val if hasattr(config, "num_workers_val") else 2,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=collate_fn
    )

    logger.info(f"len dataloader train: {len(data_loader_train)}")
    logger.info(f"len dataloader val:   {len(data_loader_val)}")

    return dataset_train, data_loader_train, data_loader_val
