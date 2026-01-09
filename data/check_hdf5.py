import h5py
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

file_path = "/home/yutian/projs/RoboTwin-RealDeploy/data/FuckingDP3/2048pts/data/episode0.hdf5"

def visualize_hdf5(file_path):
    print(f"Checking HDF5 file: {file_path}")
    print("="*60)
    
    with h5py.File(file_path, 'r') as f:
        # 1. Print Structure
        print("\n[File Structure]")
        def print_attrs(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  {name}: {obj.shape} ({obj.dtype})")
            else:
                print(f"  {name}/")
        f.visititems(print_attrs)

        # 2. Visualize First Frame Observation
        print("\n" + "="*60)
        print("VISUALIZATION (First Frame)")
        print("="*60)
        
        # --- Image ---
        if 'observation/head_camera/rgb' in f:
            # HDF5 often stores compressed images (as bytes). We need to decompress.
            rgb_data = f['observation/head_camera/rgb'][0] 
            
            # Check if it's raw bytes (compressed) or raw array
            # The current convert script saves it as 'S{max_len}' which usually means bytes string in numpy
            # Let's handle byte decoding if needed
            if len(rgb_data.shape) == 0 or (len(rgb_data.shape) == 1 and isinstance(rgb_data[0], (bytes, np.bytes_))):
               # Likely encoded bytes
               nparr = np.frombuffer(rgb_data, np.uint8)
               img_rgb = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # Decodes to BGR
               # Convert BGR to RGB for matplotlib
               img_rgb_vis = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
            else:
               # Raw array
               img_rgb = rgb_data
               img_rgb_vis = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB) if img_rgb.shape[-1] == 3 else img_rgb

            plt.figure(figsize=(6, 4))
            plt.imshow(img_rgb_vis)
            plt.title(f"RGB Frame 0\nShape: {img_rgb_vis.shape}")
            plt.axis('off')
            plt.show()
        else:
            print("[Info] No RGB image data found.")

        # --- Point Cloud ---
        if 'pointcloud' in f:
            pc_data = f['pointcloud'][0] # Take first frame: (N, 6)
            
            print(f"\n[Point Cloud Frame 0]")
            print(f"Shape: {pc_data.shape}")
            
            xyz = pc_data[:, :3]
            rgb = pc_data[:, 3:]

            # Filter valid points (remove padding)
            mask = ~np.all(xyz == 0, axis=1)
            xyz_valid = xyz[mask]
            rgb_valid = rgb[mask]
            
            print(f"Valid points: {len(xyz_valid)}")

            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(xyz_valid[:, 0], xyz_valid[:, 1], xyz_valid[:, 2], 
                      c=np.clip(rgb_valid, 0, 1), s=2, marker='.')
            
            ax.set_title(f"Point Cloud Frame 0 ({len(xyz_valid)} pts)")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            
            # Equal aspect ratio hack
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
        else:
            print("[Info] No Point Cloud data found.")

        # --- Joint Actions ---
        if 'joint_action/vector' in f:
            action_vec = f['joint_action/vector'][0]
            print(f"\n[Action Vector Frame 0] (Shape: {action_vec.shape})")
            with np.printoptions(precision=4, suppress=True):
                print(action_vec)
                
            # Plot joint trajectories for entire episode
            all_actions = f['joint_action/vector'][:] # (T, 14)
            plt.figure(figsize=(10, 6))
            for i in range(min(14, all_actions.shape[1])):
                label = f"Joint {i}"
                if i == 6: label = "Left Gripper"
                elif i == 13: label = "Right Gripper"
                elif i < 6: label = f"L-Joint {i}"
                else: label = f"R-Joint {i-7}"
                
                plt.plot(all_actions[:, i], label=label, alpha=0.7)
            
            plt.title("Joint Action Trajectories (Whole Episode)")
            plt.xlabel("Time Step")
            plt.ylabel("Value (Rad / Normalized)")
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.show()

if __name__ == "__main__":
    visualize_hdf5(file_path)
