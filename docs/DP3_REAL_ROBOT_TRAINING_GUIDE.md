# DP3 Real Robot Training and Deployment Guide

This guide explains how to use data collected with `real_deploy_collect_data.py` to train a DP3 (3D Diffusion Policy) model and deploy it on your real robot.

## Table of Contents
1. [Data Collection](#1-data-collection)
2. [Data Conversion](#2-data-conversion)
3. [Training](#3-training)
4. [Deployment](#4-deployment)

---

## 1. Data Collection

### 1.1 Run the Data Collector

```bash
python real_deploy_collect_data.py \
    --save_dir ./data/real_data_robotwin/my_task \
    --record_freq 30 \
    --gripper_max_width 0.08 \
    --pcd_num_points 1024
```

### 1.2 Controls
- **[Space]**: Start/Stop Recording
- **[H]**: Re-Home & Gravity Compensation
- **[Q]**: Quit

### 1.3 Data Structure
Each episode is saved as a folder with numbered `.pkl` files:
```
data/real_data_robotwin/my_task/
├── episode_000/
│   ├── 0.pkl
│   ├── 1.pkl
│   └── ...
├── episode_001/
│   └── ...
```

### 1.4 PKL Data Format
```python
{
    "observation": {
        "head_camera": {
            "rgb": (H, W, 3) uint8,
            "depth": (H, W) float64 in mm,
            "intrinsic_cv": (3, 3),
            "extrinsic_cv": (4, 4),
            "cam2world_gl": (4, 4)
        }
    },
    "pointcloud": (1024, 6) float32,  # xyz + rgb
    "joint_action": {
        "left_arm": (6,),
        "left_gripper": float [0,1],
        "right_arm": (6,),
        "right_gripper": float [0,1],
        "vector": (14,) float32
    },
    "endpose": {
        "left_endpose": (7,),   # [x,y,z,qw,qx,qy,qz]
        "right_endpose": (7,),
        "left_gripper": float,
        "right_gripper": float
    }
}
```

---

## 2. Data Conversion

### 2.1 Convert PKL to HDF5

Use the `real_deploy_convert_data.py` script to convert collected PKL files to HDF5 format:

```bash
# Convert PKL to HDF5 with task_name and task_config
python real_deploy_convert_data.py \
    --task_name my_real_task \
    --task_config default \
    --data_root ./data/real_data_robotwin_collected
```

**Parameters:**
- `--task_name`: Task name (required, used in output path and DP3 training)
- `--task_config`: Task config name (default: `default`)
- `--data_root`: Root directory containing episode folders (default: `./data/real_data_robotwin_collected`)

**Output Structure:**
```
./data/{task_name}/{task_config}/
├── data/
│   ├── episode0.hdf5
│   ├── episode1.hdf5
│   └── ...
└── video/
    ├── episode0.mp4
    ├── episode1.mp4
    └── ...
```

### 2.2 Convert HDF5 to Zarr (for DP3)

Navigate to the DP3 policy directory and run the data processing script:

```bash
cd policy/DP3

# Process data: task_name, task_config, num_episodes
bash process_data.sh my_real_task default 50
```

This creates a zarr dataset at:
```
policy/DP3/data/my_real_task-default-50.zarr
```

### 2.3 Complete Workflow Example

```bash
# Step 1: Collect data (multiple episodes)
python real_deploy_collect_data.py --save_dir ./data/real_data_robotwin_collected

# Step 2: Convert PKL to HDF5
python real_deploy_convert_data.py --task_name my_real_task --task_config default

# Step 3: Convert HDF5 to Zarr
cd policy/DP3
bash process_data.sh my_real_task default 50

# Step 4: Train DP3
bash train.sh my_real_task default 50 0 0
```

---

## 3. Training

### 3.1 Training Command

```bash
cd policy/DP3

# Arguments: task_name, task_config, num_episodes, seed, gpu_id
bash train.sh my_real_task default 50 0 0
```

### 3.2 Training Configuration

The training uses `robot_dp3.yaml` configuration. Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `horizon` | 8 | Prediction horizon |
| `n_obs_steps` | 3 | Number of observation steps |
| `n_action_steps` | 6 | Number of action steps to execute |
| `batch_size` | 256 | Training batch size |
| `num_epochs` | 3000 | Total training epochs |
| `checkpoint_every` | 3000 | Save checkpoint frequency |

### 3.3 Monitor Training

Training logs are saved to:
- **WandB**: Online logging (configure in `robot_dp3.yaml`)
- **Local**: `policy/DP3/3D-Diffusion-Policy/data/outputs/`

### 3.4 Checkpoints

Checkpoints are saved at:
```
policy/DP3/checkpoints/my_real_task-default-50_0/3000.ckpt
```

---

## 4. Deployment

### 4.1 Simulation Evaluation (Optional)

To evaluate in RoboTwin simulation:

```bash
cd policy/DP3

# Arguments: task_name, task_config, ckpt_setting, num_episodes, seed, gpu_id
bash eval.sh my_real_task default default 50 0 0
```

### 4.2 Real Robot Deployment

Create a deployment script for your real robot:

```python
# real_deploy_dp3.py
import sys
import os
import numpy as np
import time

sys.path.append('./')
sys.path.append('./policy/DP3')
sys.path.append('./policy/DP3/3D-Diffusion-Policy')

from policy.DP3.deploy_policy import get_model, encode_obs, reset_model

# Your robot hardware interface
from fucking_arx_mujoco.real.real_single_arm import RealSingleArm
import pyrealsense2 as rs

class RealDP3Deployer:
    def __init__(self, usr_args):
        # Load DP3 model
        self.model = get_model(usr_args)
        
        # Initialize robot arms
        self.left_arm = RealSingleArm(can_port='can3', arm_type=0)
        self.right_arm = RealSingleArm(can_port='can1', arm_type=0)
        
        # Initialize camera
        self._init_camera()
        
        # Gripper normalization
        self.gripper_max_width = 0.08
        
    def _init_camera(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        
        profile = self.pipeline.start(config)
        color_stream = profile.get_stream(rs.stream.color)
        intr = color_stream.as_video_stream_profile().get_intrinsics()
        self.fx, self.fy = intr.fx, intr.fy
        self.cx, self.cy = intr.ppx, intr.ppy
        
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        self.align = rs.align(rs.stream.color)
        
    def get_observation(self):
        # Get camera frames
        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        
        color_img = np.asanyarray(color_frame.get_data())
        depth_img = np.asanyarray(depth_frame.get_data())
        
        # Generate point cloud
        pointcloud = self._depth_to_pointcloud(depth_img, color_img)
        
        # Get robot state
        left_qpos = np.array(self.left_arm.get_joint_positions())
        right_qpos = np.array(self.right_arm.get_joint_positions())
        left_gripper = self.left_arm.get_gripper_width() / self.gripper_max_width
        right_gripper = self.right_arm.get_gripper_width() / self.gripper_max_width
        
        # Construct observation
        obs = {
            'agent_pos': np.concatenate([
                left_qpos, [left_gripper],
                right_qpos, [right_gripper]
            ]).astype(np.float32),
            'point_cloud': pointcloud
        }
        return obs
    
    def _depth_to_pointcloud(self, depth_img, color_img, num_points=1024):
        depth_m = depth_img.astype(np.float32) * self.depth_scale
        v, u = np.meshgrid(np.arange(480), np.arange(640), indexing='ij')
        valid_mask = (depth_m > 0.1) & (depth_m < 2.0)
        
        z = depth_m[valid_mask]
        x = (u[valid_mask] - self.cx) * z / self.fx
        y = (v[valid_mask] - self.cy) * z / self.fy
        colors = color_img[valid_mask][:, ::-1].astype(np.float32) / 255.0
        
        points = np.stack([x, y, z], axis=-1)
        pointcloud = np.hstack([points, colors])
        
        if len(pointcloud) > num_points:
            indices = np.random.choice(len(pointcloud), num_points, replace=False)
            pointcloud = pointcloud[indices]
        elif len(pointcloud) < num_points:
            padding = np.zeros((num_points - len(pointcloud), 6))
            pointcloud = np.vstack([pointcloud, padding])
        
        return pointcloud.astype(np.float32)
    
    def execute_action(self, action):
        """Execute a single action step."""
        # Parse action vector (14-dim)
        left_qpos = action[:6]
        left_gripper = action[6]
        right_qpos = action[7:13]
        right_gripper = action[13]
        
        # Send commands to robot
        self.left_arm.set_joint_positions(left_qpos)
        self.left_arm.set_gripper_width(left_gripper * self.gripper_max_width)
        self.right_arm.set_joint_positions(right_qpos)
        self.right_arm.set_gripper_width(right_gripper * self.gripper_max_width)
        
    def run(self, max_steps=300):
        """Main deployment loop."""
        reset_model(self.model)
        
        for step in range(max_steps):
            # Get observation
            obs = self.get_observation()
            
            # Update model observation
            if len(self.model.env_runner.obs) == 0:
                self.model.update_obs(obs)
            
            # Get actions from policy
            actions = self.model.get_action()
            
            # Execute each action step
            for action in actions:
                self.execute_action(action)
                time.sleep(0.033)  # ~30Hz
                
                # Update observation after each action
                obs = self.get_observation()
                self.model.update_obs(obs)
            
            print(f"Step {step}/{max_steps}", end='\r')
        
        print("\nDeployment complete.")
        
    def cleanup(self):
        self.pipeline.stop()


if __name__ == "__main__":
    # Configuration
    usr_args = {
        'task_name': 'my_real_task',
        'config_name': 'robot_dp3',
        'checkpoint_num': 3000,
        'expert_data_num': 50,
        'ckpt_setting': 'default',
        'seed': 0,
        'use_rgb': False,  # Set True if using RGB in point cloud
    }
    
    deployer = RealDP3Deployer(usr_args)
    
    try:
        deployer.run(max_steps=300)
    finally:
        deployer.cleanup()
```

### 4.3 Run Deployment

```bash
python real_deploy_dp3.py
```

---

## Troubleshooting

### Common Issues

1. **ZeroDivisionError during data processing**
   - Ensure `pointcloud` data is present in your PKL files
   - Check that point cloud has shape `(1024, 6)`

2. **Checkpoint not found**
   - Verify checkpoint path: `policy/DP3/checkpoints/{task_name}-{setting}-{num}_{seed}/{checkpoint_num}.ckpt`

3. **Action dimension mismatch**
   - Ensure `joint_action.vector` has 14 dimensions: `[left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)]`

4. **Point cloud quality issues**
   - Adjust depth range in `_depth_to_pointcloud()` (default: 0.1m to 2.0m)
   - Ensure camera is properly calibrated

### Tips for Better Performance

1. **Collect diverse demonstrations**: 50-100 episodes recommended
2. **Consistent lighting**: Avoid dramatic lighting changes
3. **Camera calibration**: Calibrate extrinsic matrix for accurate point clouds
4. **Data augmentation**: Consider adding noise during training
5. **Checkpoint selection**: Evaluate multiple checkpoints to find the best one

---

## File Structure Reference

```
RoboTwin-RealDeploy/
├── real_deploy_collect_data.py    # Data collection script
├── real_deploy_convert_data.py    # PKL to HDF5 conversion
├── data/
│   ├── real_data_robotwin_collected/  # Raw collected PKL data
│   │   ├── episode_000/
│   │   │   ├── 0.pkl
│   │   │   ├── 1.pkl
│   │   │   └── ...
│   │   └── episode_001/
│   │       └── ...
│   └── {task_name}/                   # Converted HDF5 data
│       └── {task_config}/
│           ├── data/
│           │   ├── episode0.hdf5
│           │   └── ...
│           └── video/
│               ├── episode0.mp4
│               └── ...
├── docs/
│   └── DP3_REAL_ROBOT_TRAINING_GUIDE.md
└── policy/
    └── DP3/
        ├── train.sh
        ├── eval.sh
        ├── process_data.sh
        ├── data/                      # Zarr datasets
        │   └── {task_name}-{config}-{num}.zarr
        ├── checkpoints/               # Trained models
        │   └── {task_name}-{config}-{num}_{seed}/
        │       └── 3000.ckpt
        └── 3D-Diffusion-Policy/
```

---

## Quick Reference Commands

```bash
# 1. Collect data
python real_deploy_collect_data.py --save_dir ./data/real_data_robotwin_collected

# 2. Convert PKL to HDF5
python real_deploy_convert_data.py --task_name my_task --task_config default

# 3. Process HDF5 to Zarr (from policy/DP3 directory)
cd policy/DP3
bash process_data.sh my_task default NUM_EPISODES

# 4. Train DP3
bash train.sh my_task default NUM_EPISODES SEED GPU_ID

# 5. Evaluate
bash eval.sh my_task default default NUM_EPISODES SEED GPU_ID
```
