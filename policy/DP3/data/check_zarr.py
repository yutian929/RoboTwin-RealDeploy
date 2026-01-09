import zarr
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# 默认路径，根据您的上下文推断
default_path = "20260109dp3-cut4096pts-20.zarr"

def visualize_zarr(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    print(f"Checking Zarr file: {file_path}")
    print("="*60)
    
    root = zarr.open(file_path, mode='r')
    
    # 1. Print Structure
    print("\n[File Structure]")
    print(root.tree())
    
    # Get Datasets
    point_cloud = root['data/point_cloud']
    state = root['data/state']
    action = root['data/action']
    episode_ends = root['meta/episode_ends']
    
    print("\n[Dataset Shapes]")
    print(f"  point_cloud: {point_cloud.shape}")
    print(f"  state:       {state.shape}")
    print(f"  action:      {action.shape}")
    print(f"  episode_ends:{episode_ends.shape}")
    print(f"  Total Transitions: {state.shape[0]}")
    
    # 2. Visualize First Frame Point Cloud
    print("\n" + "="*60)
    print("VISUALIZATION (Index 0)")
    print("="*60)
    
    pc_data = point_cloud[0] # (N, 6)
    state_data = state[0]    # (14,)
    action_data = action[0]  # (14,)
    
    print(f"\n[State Vector]  : {np.array2string(state_data, precision=4, suppress_small=True)}")
    print(f"[Action Vector] : {np.array2string(action_data, precision=4, suppress_small=True)}")
    
    # 3D Plot
    xyz = pc_data[:, :3]
    rgb = pc_data[:, 3:]
    
    # Filter valid
    mask = ~np.all(xyz == 0, axis=1)
    xyz_valid = xyz[mask]
    rgb_valid = rgb[mask]
    
    print(f"[Point Cloud]: {len(xyz_valid)} valid points")
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(xyz_valid[:, 0], xyz_valid[:, 1], xyz_valid[:, 2], 
               c=np.clip(rgb_valid, 0, 1), s=2, marker='.')
    
    ax.set_title(f"Zarr Point Cloud [Index 0]")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    
    # Aspect Ratio Hack
    max_range = np.array([
        xyz_valid[:, 0].max()-xyz_valid[:, 0].min(), 
        xyz_valid[:, 1].max()-xyz_valid[:, 1].min(), 
        xyz_valid[:, 2].max()-xyz_valid[:, 2].min()
    ]).max() / 2.0
    mid_x = (xyz_valid[:, 0].max()+xyz_valid[:, 0].min()) * 0.5
    mid_y = (xyz_valid[:, 1].max()+xyz_valid[:, 1].min()) * 0.5
    mid_z = (xyz_valid[:, 2].max()+xyz_valid[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.show()
    
    # 3. Trajectory Inspection
    print("\n" + "="*60)
    print("TRAJECTORY INSPECTION (First Episode)")
    print("="*60)
    
    # Get first episode range
    start_idx = 0
    end_idx = episode_ends[0]
    
    ep_state = state[start_idx:end_idx]   # (T, 14)
    ep_action = action[start_idx:end_idx] # (T, 14)
    
    print(f"Episode 0 Length: {end_idx - start_idx} steps")
    
    # Plot State vs Action for ALL joints
    # Ideally State[t+1] should resemble Action[t] closely in this pipeline
    plt.figure(figsize=(20, 16))
    
    num_joints = ep_state.shape[1]
    cols = 4
    rows = (num_joints + cols - 1) // cols
    
    for j_idx in range(num_joints):
        plt.subplot(rows, cols, j_idx+1)
        plt.plot(ep_state[:, j_idx], label='State (t)', marker='o', markersize=2, linestyle='-', linewidth=1)
        plt.plot(ep_action[:, j_idx], label='Action (Target)', alpha=0.7, linestyle='--', linewidth=1)
        
        # Determine title based on index
        if j_idx < 6:
            title = f"Left Arm Joint {j_idx}"
        elif j_idx == 6:
            title = "Left Gripper"
        elif j_idx < 13:
            title = f"Right Arm Joint {j_idx - 7}"
        elif j_idx == 13:
            title = "Right Gripper"
        else:
            title = f"Index {j_idx}"
            
        plt.title(title)
        if j_idx == 0: # Only legend on first plot to avoid clutter
            plt.legend()
        plt.grid(True, alpha=0.3)
        
    plt.suptitle("Check Zarr- All Joints")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize_zarr(default_path)
