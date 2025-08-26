import numpy as np
from shapely.geometry import Polygon
import torch



def get_polygon(box):
    """
    Convert a 7-parameter box to a Shapely Polygon.
    box: [x, y, z, dx, dy, dz, yaw]
    """
    x, y, dx, dy, yaw = box[0], box[1], box[3], box[4], box[6]
    
    # Corner offsets from center in local frame
    corners = np.array([
        [ dx/2,  dy/2],
        [ dx/2, -dy/2],
        [-dx/2, -dy/2],
        [-dx/2,  dy/2],
    ])
    
    # Rotation matrix
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    
    # Rotate and translate
    corners = corners @ R.T + np.array([x, y])
    return Polygon(corners)

def box3d_iou(box1, box2):
    """
    Computes a true 3D IoU using Shapely for robust BEV intersection.
    box1, box2: [x, y, z, dx, dy, dz, yaw, score]
    """
    # 1. Get BEV polygons
    poly1 = get_polygon(box1)
    poly2 = get_polygon(box2)
    
    # 2. Check for BEV intersection and calculate intersection area
    if not poly1.intersects(poly2):
        return 0.0
    
    inter_area = poly1.intersection(poly2).area

    # 3. Calculate BEV IoU
    union_area = poly1.area + poly2.area - inter_area
    if union_area == 0:
        return 0.0
    bev_iou = inter_area / union_area

    # 4. Compute Z-axis intersection
    z_min1, z_max1 = box1[2] - box1[5]/2, box1[2] + box1[5]/2
    z_min2, z_max2 = box2[2] - box2[5]/2, box2[2] + box2[5]/2
    intersection_z = max(0, min(z_max1, z_max2) - max(z_min1, z_min2))
    
    # 5. Compute 3D volumes
    volume1 = box1[3] * box1[4] * box1[5]
    volume2 = box2[3] * box2[4] * box2[5]
    
    # 6. Calculate the final 3D IoU
    intersection_3d = inter_area * intersection_z
    union_3d = volume1 + volume2 - intersection_3d
    
    if union_3d == 0:
        return 0.0
        
    return intersection_3d / union_3d
import numpy as np
import torch
from pcdet.ops.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu

def compute_ap3d_r11(all_preds, all_gts, iou_thresh=0.3, device='cuda'):
    """
    Compute 3D Average Precision at R11 (AP3D|R11) using N×7 boxes and boxes_iou3d_gpu.
    
    Args:
        all_preds: list of np.array, each of shape (N_i, 8) -> [x, y, z, dx, dy, dz, yaw, score]
        all_gts:   list of np.array, each of shape (M_i, 7) -> [x, y, z, dx, dy, dz, yaw]
        iou_thresh: IoU threshold for matching
        device: 'cuda' or 'cpu'
    Returns:
        ap3d_r11: float
    """
    all_scores = []
    all_matches = []

    for pred_boxes, gt_boxes in zip(all_preds, all_gts):
        if len(pred_boxes) == 0:
            continue

        pred_boxes_tensor = torch.tensor(pred_boxes[:, :7], dtype=torch.float32, device=device)
        gt_boxes_tensor = torch.tensor(gt_boxes, dtype=torch.float32, device=device)

        matched_gt = torch.zeros(len(gt_boxes), dtype=torch.bool, device=device)

        for i, pb in enumerate(pred_boxes_tensor):
            if len(gt_boxes_tensor) == 0:
                all_matches.append(0)
                all_scores.append(pred_boxes[i, 7])
                continue

            ious = boxes_iou3d_gpu(pb.unsqueeze(0), gt_boxes_tensor).squeeze(0)
            ious[matched_gt] = -1  # ignore already matched GTs

            best_iou, best_idx = ious.max(0)
            if best_iou >= iou_thresh:
                matched_gt[best_idx] = True
                all_matches.append(1)
            else:
                all_matches.append(0)

            all_scores.append(pred_boxes[i, 7])

    # Sort by score descending
    all_scores = np.array(all_scores)
    all_matches = np.array(all_matches)
    sorted_idx = np.argsort(-all_scores)
    all_matches = all_matches[sorted_idx]

    # Compute precision-recall
    num_gts = sum(len(gt_array) for gt_array in all_gts)
    tp_cum = np.cumsum(all_matches)
    fp_cum = np.cumsum(1 - all_matches)
    recalls = tp_cum / max(num_gts, 1)
    precisions = tp_cum / (tp_cum + fp_cum + 1e-6)

    # R11 recall points
    r11_points = np.linspace(0, 1, 11)
    precisions_interp = []
    for r in r11_points:
        prec = precisions[recalls >= r]
        precisions_interp.append(np.max(prec) if len(prec) > 0 else 0.0)

    ap3d_r11 = np.mean(precisions_interp)
    return ap3d_r11

import torch

def average_precision_3d(pred_boxes, pred_scores, gt_boxes, iou_threshold=0.5):
    """
    pred_boxes: (N_pred, 7)
    pred_scores: (N_pred,) confidence for object class
    gt_boxes: (N_gt, 7)
    iou_threshold: IoU threshold to consider a detection as TP
    """
    
    if len(pred_boxes) == 0:
        return 0.0  # no predictions, AP = 0

    # Sort predictions by score descending
    sorted_idx = torch.argsort(pred_scores, descending=True)
    pred_boxes = pred_boxes[sorted_idx]
    pred_scores = pred_scores[sorted_idx]

    num_gt = gt_boxes.shape[0]
    matched_gt = torch.zeros(num_gt, dtype=torch.bool, device=gt_boxes.device)
    tp, fp = [], []

    for box in pred_boxes:
        if num_gt == 0:
            tp.append(0)
            fp.append(1)
            continue

        ious = boxes_iou3d_gpu(box[None, :], gt_boxes)  # (1, G)
        max_iou, max_idx = ious.max(dim=1)

        if max_iou.item() >= iou_threshold and not matched_gt[max_idx]:
            tp.append(1)
            fp.append(0)
            matched_gt[max_idx] = True
        else:
            tp.append(0)
            fp.append(1)

    tp = torch.cumsum(torch.tensor(tp, dtype=torch.float32), dim=0)
    fp = torch.cumsum(torch.tensor(fp, dtype=torch.float32), dim=0)

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (num_gt + 1e-8)

    # AP: area under precision-recall curve
    ap = torch.trapz(precision, recall)
    return ap.item()



def compute_ap3d_r11_8d(all_preds, all_gts, iou_thresh=0.3):
    """
    Compute 3D Average Precision at R11 (AP3D|R11)
    Args:
        all_preds: list of np.array, each of shape (N_i, 8)
        all_gts:   list of np.array, each of shape (M_i, 8, 3) (GT is in corners)
        iou_thresh: IoU threshold for matching
    Returns:
        ap3d_r11: float
    """
    print("Computing AP3D|R11 with IoU threshold:", iou_thresh)
    print("Number of samples:", len(all_preds))
    #SHAPE
    print("Example pred shape:", all_preds[0].shape if len(all_preds) > 0 else "N/A")
    print("Example GT shape:", all_gts[0].shape if len(all_gts) > 0 else "N/A")
    all_scores = []
    all_matches = []
    
    # loop over all samples
    for pb_array, gb_array in zip(all_preds, all_gts):
        # Convert GT corners to 7-parameter boxes
        # This is the CRITICAL FIX!
        gt_boxes7 = np.array([corners_to_box7(gb) for gb in gb_array])
        
        # store which GTs are matched
        matched_gt = np.zeros(len(gt_boxes7), dtype=bool)

        # loop over predicted boxes
        for pb in pb_array:  # pb.shape = (8,)
            best_iou = 0.0
            best_idx = -1
            # loop over GT boxes
            for idx, gb_box in enumerate(gt_boxes7):  # gb_box is now in (8,) format
                if matched_gt[idx]:
                    continue
                iou = box3d_iou(pb, gb_box) # Pass the correct format
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
                    
            
            if best_iou >= iou_thresh and best_idx >= 0:
                matched_gt[best_idx] = True
                all_matches.append(1)  # True positive
            else:
                all_matches.append(0)  # False positive

            # use predicted score (last dimension)
            all_scores.append(pb[7])  # raw score, not sigmoid

    # sort by score descending
    all_scores = np.array(all_scores)
    all_matches = np.array(all_matches)
    sorted_idx = np.argsort(-all_scores)
    all_matches = all_matches[sorted_idx]

    # compute precision-recall
    num_gts = sum(len(gt_array) for gt_array in all_gts)
    tp_cum = np.cumsum(all_matches)
    fp_cum = np.cumsum(1 - all_matches)
    recalls = tp_cum / max(num_gts, 1)
    precisions = tp_cum / (tp_cum + fp_cum + 1e-6)

    # R11 recall points
    r11_points = np.linspace(0, 1, 11)
    precisions_interp = []
    for r in r11_points:
        prec = precisions[recalls >= r]
        if len(prec) == 0:
            precisions_interp.append(0.0)
        else:
            precisions_interp.append(np.max(prec))

    ap3d_r11 = np.mean(precisions_interp)
    return ap3d_r11

def corners_to_box7(corners):
    """
    Convert 8 corners (8,3) to 7D box: (cx,cy,cz,dx,dy,dz,dz,yaw)
    Returns: A numpy array of shape (7,).
    """
    # CRITICAL FIX: Add a check for empty input or a non-array input
    if not isinstance(corners, np.ndarray) or corners.size == 0:
        return np.zeros(7, dtype=np.float32)

    # Compute center
    cx, cy, cz = corners.mean(axis=0)

    # Compute dimensions
    dists = np.array([
        np.linalg.norm(corners[0,:2] - corners[1,:2]),
        np.linalg.norm(corners[0,:2] - corners[3,:2])
    ])
    dx = np.max(dists)
    dy = np.min(dists)
    dz = np.linalg.norm(corners[0] - corners[4])

    # Approximate yaw
    if dists[0] > dists[1]:
        vec = corners[1,:2] - corners[0,:2]
    else:
        vec = corners[3,:2] - corners[0,:2]
    yaw = np.arctan2(vec[1], vec[0])

    return np.array([cx, cy, cz, dx, dy, dz, yaw])

def box7_to_corners(box7):
    x, y, z, dx, dy, dz, yaw = box7[:7]
    corners = np.array([
        [ dx/2,  dy/2,  dz/2], [ dx/2, -dy/2,  dz/2],
        [-dx/2, -dy/2,  dz/2], [-dx/2,  dy/2,  dz/2],
        [ dx/2,  dy/2, -dz/2], [ dx/2, -dy/2, -dz/2],
        [-dx/2, -dy/2, -dz/2], [-dx/2,  dy/2, -dz/2],
    ])
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c,-s,0],[s,c,0],[0,0,1]])
    corners = corners @ R.T
    corners += np.array([x,y,z])
    return corners

def sanity_check_corners(corners):
    if corners.shape != (8,3):
        print("Warning: Unexpected corner shape:", corners.shape)
        return False

    # Check if any coordinate is NaN or Inf
    if torch.isnan(corners).any() or torch.isinf(corners).any():
        print("Error: Corners contain NaN or Inf")
        return False

    # Compute dimension ranges
    dx = corners[:,0].max() - corners[:,0].min()
    dy = corners[:,1].max() - corners[:,1].min()
    dz = corners[:,2].max() - corners[:,2].min()

    print(f"dx: {dx:.3f}, dy: {dy:.3f}, dz: {dz:.3f}")
    
    if dx <= 0 or dy <= 0 or dz <= 0:
        print("Error: Non-positive dimensions!")
        return False

    return True
import torch

def corners_to_7d(corners):
    """
    Convert 8 corners (8,3) to 7D box: (cx, cy, cz, dx, dy, dz, yaw)
    Ensures positive dimensions and checks corner validity.
    """
    # --- Check empty or invalid corners ---
    if corners.shape != (8, 3):
        print("Warning: Invalid corner shape:", corners.shape)
        return torch.zeros(7, dtype=torch.float32, device=corners.device)
    
    if torch.isnan(corners).any() or torch.isinf(corners).any():
        print("Warning: Corners contain NaN or Inf")
        return torch.zeros(7, dtype=torch.float32, device=corners.device)
    
    # --- Compute center ---
    cx, cy, cz = corners.mean(dim=0)

    # --- Compute dimensions ---
    # X dimension: distance between corners along x-axis
    dx = torch.linalg.norm(corners[0,:2] - corners[1,:2])
    dy = torch.linalg.norm(corners[0,:2] - corners[3,:2])
    dz = torch.linalg.norm(corners[0] - corners[4])

    # Ensure positive dimensions
    dx = torch.abs(dx).clamp(min=1e-3)
    dy = torch.abs(dy).clamp(min=1e-3)
    dz = torch.abs(dz).clamp(min=1e-3)

    # --- Approximate yaw ---
    vec = corners[1,:2] - corners[0,:2] if dx >= dy else corners[3,:2] - corners[0,:2]
    yaw = torch.atan2(vec[1], vec[0])
    
    dim_7d=torch.tensor([cx, cy, cz, dx, dy, dz, yaw], dtype=torch.float32, device=corners.device)


    return dim_7d
