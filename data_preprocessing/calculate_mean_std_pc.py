import pandas as pd
import numpy as np
import torch
import os

# --- Define necessary helper functions used for bbox conversion ---
# Copy these directly from your train.py file to ensure consistency
def corners_to_box7(corners):
    """
    Convert 8 corners (8,3) to box7: (cx,cy,cz,dx,dy,dz,dz,yaw)
    Returns: A NumPy array of shape (7,) (excluding dummy score for stats calculation).
    """
    if corners.shape[0] == 0: # Handle empty corner arrays
        return np.zeros(7) # Return a dummy zero array

    # Compute center
    cx, cy, cz = corners.mean(axis=0)

    # Compute dimensions (more robust estimation from actual distances)
    # Find the longest edge on the XY plane to determine dx and dy
    # This is more robust than assuming corner order
    dists = np.array([
        np.linalg.norm(corners[0,:2] - corners[1,:2]),
        np.linalg.norm(corners[0,:2] - corners[3,:2])
    ])
    dx = np.max(dists)
    dy = np.min(dists)
    dz = np.linalg.norm(corners[0] - corners[4])

    # Approximate yaw from the longest edge on the XY plane
    if dists[0] > dists[1]:
        vec = corners[1,:2] - corners[0,:2]
    else:
        vec = corners[3,:2] - corners[0,:2]
    yaw = np.arctan2(vec[1], vec[0])

    return np.array([cx, cy, cz, dx, dy, dz, yaw]) # Return 7 parameters

# --- Configuration ---
TRAIN_CSV_PATH = '/home/as2114/code/3DBB/data/dl_challenge/train.csv' # Path to your train.csv file
# Assuming your .npy paths in the CSV are relative to the CSV's directory
# Or adjust this to your data root
DATA_ROOT = os.path.dirname(TRAIN_CSV_PATH) if os.path.exists(TRAIN_CSV_PATH) else './' 

# --- Data Collection ---
all_pc_coords = [] # To store all x, y, z coordinates from all point clouds
all_bbox_params = [] # To store all 7 bbox parameters from all ground truths

# Load the dataframe
df = pd.read_csv(TRAIN_CSV_PATH)

print(f"Loading data from {len(df)} samples...")

for index, row in df.iterrows():
    pc_path = os.path.join(DATA_ROOT, row['pc'])
    bbox_path = os.path.join(DATA_ROOT, row['bbox3d'])

    # --- Process Point Cloud ---
    if os.path.exists(pc_path):
        pc_data = np.load(pc_path).astype(np.float32) # (3, H, W)
        # Reshape to (3, N) where N is H*W, then transpose to (N, 3) for easier mean/std
        pc_coords = pc_data.reshape(3, -1).T 
        all_pc_coords.append(pc_coords)
    else:
        print(f"Warning: Point cloud file not found: {pc_path}")

    # --- Process Bounding Boxes ---
    if os.path.exists(bbox_path):
        bbox_data = np.load(bbox_path).astype(np.float32) # Expected (N_boxes, 8, 3) (corners)
        
        # Ensure bbox_data is 3D (N_boxes, 8, 3)
        if bbox_data.ndim == 3 and bbox_data.shape[1] == 8 and bbox_data.shape[2] == 3:
            for bbox_corners in bbox_data: # Iterate over each box (8,3)
                bbox_params = corners_to_box7(bbox_corners) # Convert to 7 params
                all_bbox_params.append(bbox_params)
        else:
            print(f"Warning: Bbox file {bbox_path} has unexpected shape: {bbox_data.shape}. Skipping.")
    else:
        print(f"Warning: Bbox file not found: {bbox_path}")

# --- Calculate Statistics ---

# Concatenate all collected data
if all_pc_coords:
    all_pc_coords_combined = np.concatenate(all_pc_coords, axis=0)
    pc_mean = np.mean(all_pc_coords_combined, axis=0)
    pc_std = np.std(all_pc_coords_combined, axis=0)
    # Handle zero std to prevent division by zero in normalization
    pc_std[pc_std == 0] = 1.0 # Or a very small epsilon
else:
    pc_mean = np.array([0.0, 0.0, 0.0])
    pc_std = np.array([1.0, 1.0, 1.0]) # Default to unit if no data

if all_bbox_params:
    all_bbox_params_combined = np.array(all_bbox_params) # Convert list of arrays to single array
    bbox_mean = np.mean(all_bbox_params_combined, axis=0)
    bbox_std = np.std(all_bbox_params_combined, axis=0)
    # Handle zero std
    bbox_std[bbox_std == 0] = 1.0 # Or a very small epsilon
else:
    bbox_mean = np.zeros(7)
    bbox_std = np.ones(7) # Default to unit if no data


print("\n--- Calculated Statistics ---")
print("Point Cloud Mean (x, y, z):", pc_mean)
print("Point Cloud Std (x, y, z):", pc_std)
print("\nBounding Box Mean (cx, cy, cz, dx, dy, dz, yaw):", bbox_mean)
print("Bounding Box Std (cx, cy, cz, dx, dy, dz, yaw):", bbox_std)

print("\n--- Copy these values into your PointCloudDataset class ---")
print(f"self.pc_mean = torch.tensor([{pc_mean[0]:.6f}, {pc_mean[1]:.6f}, {pc_mean[2]:.6f}])")
print(f"self.pc_std = torch.tensor([{pc_std[0]:.6f}, {pc_std[1]:.6f}, {pc_std[2]:.6f}])")
print(f"self.bbox_mean = torch.tensor([{bbox_mean[0]:.6f}, {bbox_mean[1]:.6f}, {bbox_mean[2]:.6f}, "
      f"{bbox_mean[3]:.6f}, {bbox_mean[4]:.6f}, {bbox_mean[5]:.6f}, {bbox_mean[6]:.6f}])")
print(f"self.bbox_std = torch.tensor([{bbox_std[0]:.6f}, {bbox_std[1]:.6f}, {bbox_std[2]:.6f}, "
      f"{bbox_std[3]:.6f}, {bbox_std[4]:.6f}, {bbox_std[5]:.6f}, {bbox_std[6]:.6f}])")