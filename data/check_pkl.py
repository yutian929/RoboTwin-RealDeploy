import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2

# 替换为你的文件名
file_path = 'real_data_robotwin_collected/episode_000/50.pkl'

def visualize_data(data):
    # 1. Image Visualization (RGB & Depth)
    head_camera = data['observation']['head_camera']
    
    # RGB Image
    rgb_img = head_camera['rgb']
    # Depth Image
    depth_img = head_camera['depth']
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    # Convert BGR to RGB for matplotlib if needed (assuming origin is BGR from cv2)
    # Although your printed array looks like it might be RGB or BGR, usually cv2 saves BGR
    # Let's assume it's valid for visualization. 
    # If colors look weird, you might need: rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
    plt.imshow(rgb_img[:, :, ::-1]) # Assuming stored as BGR
    plt.title(f"RGB Image {rgb_img.shape}")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(depth_img, cmap='plasma')
    plt.title(f"Depth Image {depth_img.shape}\n(Min: {depth_img.min():.1f}, Max: {depth_img.max():.1f})")
    plt.colorbar(label='Depth (mm)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

    # 2. Point Cloud Visualization
    pointcloud = data['pointcloud'] # (N, 6) xyz+rgb
    
    if pointcloud is not None and len(pointcloud) > 0:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        xyz = pointcloud[:, :3]
        rgb = pointcloud[:, 3:] # Assuming normalized [0, 1] based on standard pyrealsense/open3d logic or your previous outputs
        
        # Filter out zero-padding if any (often [0,0,0] points)
        non_zero_mask = ~np.all(xyz == 0, axis=1)
        xyz = xyz[non_zero_mask]
        rgb = rgb[non_zero_mask]
        
        # Clip RGB to valid range just in case
        rgb = np.clip(rgb, 0.0, 1.0)
        
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=rgb, s=2, marker='.')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f"Point Cloud ({len(xyz)} valid points)")
        
        # Set equal aspect ratio for better viewing
        # Create a bounding box to force equal aspect ratio
        max_range = np.array([xyz[:, 0].max()-xyz[:, 0].min(), xyz[:, 1].max()-xyz[:, 1].min(), xyz[:, 2].max()-xyz[:, 2].min()]).max() / 2.0
        mid_x = (xyz[:, 0].max()+xyz[:, 0].min()) * 0.5
        mid_y = (xyz[:, 1].max()+xyz[:, 1].min()) * 0.5
        mid_z = (xyz[:, 2].max()+xyz[:, 2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        plt.show()
    
    # 3. Robot State / Action Visualization
    print("\n" + "="*50)
    print("ROBOT STATE / ACTION DETAILS")
    print("="*50)
    
    joint_action = data['joint_action']
    endpose = data.get('endpose', None)
    
    # Print Joint Actions
    print("\n[Joint Action]")
    print(f"Left Arm ({len(joint_action['left_arm'])}): {np.array2string(joint_action['left_arm'], precision=4, suppress_small=True)}")
    print(f"Left Gripper: {joint_action['left_gripper']:.4f}")
    print(f"Right Arm ({len(joint_action['right_arm'])}): {np.array2string(joint_action['right_arm'], precision=4, suppress_small=True)}")
    print(f"Right Gripper: {joint_action['right_gripper']:.4f}")
    
    if 'vector' in joint_action:
         print(f"Full Vector ({len(joint_action['vector'])}): {np.array2string(joint_action['vector'], precision=4, suppress_small=True)}")

    # Print Endpose if available
    if endpose:
        print("\n[End Effector Pose]")
        if 'left_endpose' in endpose:
            le = endpose['left_endpose']
            print(f"Left Pose [x,y,z]: {np.array2string(le[:3], precision=4)}")
            print(f"Left Quat [w,x,y,z]: {np.array2string(le[3:], precision=4)}")
            
        if 'right_endpose' in endpose:
            re = endpose['right_endpose']
            print(f"Right Pose [x,y,z]: {np.array2string(re[:3], precision=4)}")
            print(f"Right Quat [w,x,y,z]: {np.array2string(re[3:], precision=4)}")

    # 4. Camera Intrinsics/Extrinsics
    print("\n" + "="*50)
    print("CAMERA PARAMETERS")
    print("="*50)
    obs = data['observation']['head_camera']
    print("\n[Intrinsics]")
    print(obs.get('intrinsic_cv', 'N/A'))
    
    if 'extrinsic_cv' in obs:
        print("\n[Extrinsics]")
        print(obs['extrinsic_cv'])

with open(file_path, 'rb') as f:
    data = pickle.load(f)

print(f"数据类型: {type(data)}")
print(f"Keys: {data.keys()}")

visualize_data(data)