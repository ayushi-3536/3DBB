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

def box3d_iou_correct(box1, box2):
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

def compute_ap3d_r11(all_preds, all_gts, iou_thresh=0.7):
    """
    Compute 3D Average Precision at R11 (AP3D|R11)
    Args:
        all_preds: list of np.array, each of shape (N_i, 8)
        all_gts:   list of np.array, each of shape (M_i, 8, 3) (GT is in corners)
        iou_thresh: IoU threshold for matching
    Returns:
        ap3d_r11: float
    """
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
                iou = box3d_iou_correct(pb, gb_box) # Pass the correct format
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

def corners_to_7d(corners):
    """
    Convert 8 corners (8,3) to 7D box: (cx,cy,cz,dx,dy,dz,yaw)
    Returns: A PyTorch tensor of shape (7,).
    """
    # --- CRITICAL FIX: The input tensor 'corners' can be empty.
    if corners.shape[0] == 0:
        return torch.zeros(7, device=corners.device)
    
    # Compute center
    cx, cy, cz = corners.mean(axis=0)

    # Compute dimensions
    dists = torch.tensor([
        torch.linalg.norm(corners[0,:2] - corners[1,:2]),
        torch.linalg.norm(corners[0,:2] - corners[3,:2])
    ])
    dx = torch.max(dists)
    dy = torch.min(dists)
    dz = torch.linalg.norm(corners[0] - corners[4])

    # Approximate yaw
    if dists[0] > dists[1]:
        vec = corners[1,:2] - corners[0,:2]
    else:
        vec = corners[3,:2] - corners[0,:2]
    yaw = torch.atan2(vec[1], vec[0])

    return torch.tensor([cx, cy, cz, dx, dy, dz, yaw])

