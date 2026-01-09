#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RoboTwin Real Robot Data Collection Script (Adapted from ALOHA style)
Hardware: 2x ARX R5 Arms (as Followers/Puppets) + Masters (optional/implied) + RealSense RGBD
Target Format: RoboTwin .pkl sequence format for DP3 training

Usage:
    python real_deploy_collect_data.py --save_dir ./data/real_robot_task_v1 --record_freq 30

Data Format (RoboTwin compatible):
    - observation/head_camera: rgb, depth, intrinsic_cv, extrinsic_cv, cam2world_gl
    - pointcloud: (N, 6) array with xyz + rgb
    - joint_action: left_arm, left_gripper, right_arm, right_gripper, vector (14-dim)
    - endpose: left_endpose (7-dim), right_endpose (7-dim), grippers
"""

import os
import time
import signal
import sys
import threading
import numpy as np
import cv2
import argparse
import pickle
import pyrealsense2 as rs
from copy import deepcopy
import transforms3d as t3d
import matplotlib.pyplot as plt

# Hardware Interface
# Note: Ensure this module is in your python path
try:
    from fucking_arx_mujoco.real.real_single_arm import RealSingleArm
except ImportError:
    print("[Error] Could not import 'fucking_arx_mujoco.real.real_single_arm'. Please check your environment.")
    sys.exit(1)

def save_pkl(path, data):
    with open(path, "wb") as f:
        pickle.dump(data, f)

def get_next_episode_dir(root_dir):
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    idx = 0
    while True:
        # RoboTwin convention usually doesn't strictly force naming, but let's use episode_N
        dir_name = f"episode_{idx:03d}" 
        full_path = os.path.join(root_dir, dir_name)
        if not os.path.exists(full_path):
            return full_path, idx
        idx += 1

class RoboTwinRealCollector:
    def __init__(self, args):
        self.args = args
        self.running = True
        self.is_recording = False
        
        # Directories
        self.save_root = args.save_dir
        self.current_episode_dir = None
        self.current_episode_idx = 0
        self.frame_idx = 0

        # Intervals
        self.record_interval = 1.0 / self.args.record_freq
        self.last_record_time = 0.0

        # Robot State Buffer (Thread-safe)
        self.state_lock = threading.Lock()
        self.current_state = {
            "qpos_L": np.zeros(6), "width_L": 0.0,
            "qpos_R": np.zeros(6), "width_R": 0.0,
            "pose_L": np.zeros(7),  # [x, y, z, qw, qx, qy, qz]
            "pose_R": np.zeros(7),  # [x, y, z, qw, qx, qy, qz]
        }
        
        # Gripper normalization parameters (adjust based on your hardware)
        self.gripper_max_width = args.gripper_max_width  # Max gripper width in meters

        # Camera Intrinsics
        self.intrinsics_matrix = np.eye(3)
        self.depth_scale = 0.001
        
        # Initialize Hardware
        self._init_arms()
        self._init_camera()
        
        # 3D Visualization setup
        self.vis_pcd_fig = None
        self.vis_pcd_ax = None
        if self.args.vis_pcd:
            plt.ion()
            self.vis_pcd_fig = plt.figure(figsize=(8, 6))
            self.vis_pcd_ax = self.vis_pcd_fig.add_subplot(111, projection='3d')
            print(">>> 3D Visualization Enabled.")
        
        signal.signal(signal.SIGINT, self._signal_handler)

    def _init_arms(self):
        print(">>> Initializing ARX Arms...")
        try:
            # Assuming standard ALOHA 4-arm setup based on user reference
            # If you only use 2 arms, you might need to adjust this logic to just init followers
            # and use gravity compensation for teaching.
            self.master_l = RealSingleArm(can_port=self.args.master_l, arm_type=0, max_velocity=300, max_acceleration=800)
            self.master_r = RealSingleArm(can_port=self.args.master_r, arm_type=0, max_velocity=300, max_acceleration=800)
            self.follower_l = RealSingleArm(can_port=self.args.follower_l, arm_type=0, max_velocity=300, max_acceleration=800)
            self.follower_r = RealSingleArm(can_port=self.args.follower_r, arm_type=0, max_velocity=300, max_acceleration=800)
            print(">>> Arms Initialized.")
        except Exception as e:
            print(f"[Error] Arm initialization failed: {e}")
            sys.exit(1)

    def _init_camera(self):
        print(">>> Initializing RealSense...")
        self.pipeline = rs.pipeline()
        config = rs.config()
        
        self.img_width, self.img_height = 640, 480
        config.enable_stream(rs.stream.color, self.img_width, self.img_height, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, self.img_width, self.img_height, rs.format.z16, 30)
        
        try:
            profile = self.pipeline.start(config)
            
            # Intrinsics
            color_stream = profile.get_stream(rs.stream.color)
            intr = color_stream.as_video_stream_profile().get_intrinsics()
            self.intrinsics_matrix = np.array([
                [intr.fx, 0, intr.ppx],
                [0, intr.fy, intr.ppy],
                [0, 0, 1]
            ])
            self.fx, self.fy = intr.fx, intr.fy
            self.cx, self.cy = intr.ppx, intr.ppy
            
            # Depth Scale
            depth_sensor = profile.get_device().first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()
            
            # Extrinsic matrix (camera to world transform)
            # Default: identity if camera is not calibrated to world frame
            # You should calibrate and set this based on your setup
            self.extrinsic_matrix = np.eye(4)
            self.cam2world_gl = np.eye(4)
            
            self.align = rs.align(rs.stream.color)
            print(">>> RealSense Initialized.")
            print(f"    Intrinsics: fx={self.fx:.2f}, fy={self.fy:.2f}, cx={self.cx:.2f}, cy={self.cy:.2f}")
            print(f"    Depth scale: {self.depth_scale}")
        except Exception as e:
            print(f"[Error] Camera init failed: {e}")
            sys.exit(1)

    def _signal_handler(self, signum, frame):
        print("\n[Signal] Exiting...")
        self.running = False

    def _depth_to_pointcloud(self, depth_img, color_img, num_points=1024,
                             xyz_range=None):
        """
        Convert depth image to point cloud with colors and distance-based filtering.
        
        Args:
            depth_img: Raw depth image (uint16, in depth units)
            color_img: BGR color image (uint8)
            num_points: Number of points to sample (default 1024 for DP3)
            xyz_range: Dict with keys 'x', 'y', 'z' containing [min, max] ranges
            
        Returns:
            pointcloud: (N, 6) array with [x, y, z, r, g, b] (rgb normalized to [0,1])
            sampled_pixel_coords: (N, 2) array with [u, v] pixel coordinates for visualization
        """
        # Convert depth to meters
        depth_m = depth_img.astype(np.float32) * self.depth_scale
        
        # Create pixel coordinate grid
        v, u = np.meshgrid(
            np.arange(self.img_height),
            np.arange(self.img_width),
            indexing='ij'
        )
        
        # Basic valid depth mask (filter out invalid depth values)
        if xyz_range is None:
            valid_mask = (depth_m > 0.1) & (depth_m < 2.0)  # Default 0.1m to 2m range
        else:
            z_min, z_max = xyz_range.get('z', [0.1, 2.0])
            valid_mask = (depth_m > z_min) & (depth_m < z_max)
        
        # Back-project to 3D
        z = depth_m[valid_mask]
        x = (u[valid_mask] - self.cx) * z / self.fx
        y = (v[valid_mask] - self.cy) * z / self.fy
        
        # Get pixel coordinates
        pixel_u = u[valid_mask]
        pixel_v = v[valid_mask]
        
        # Apply XYZ range filtering if specified
        if xyz_range is not None:
            x_min, x_max = xyz_range.get('x', [-np.inf, np.inf])
            y_min, y_max = xyz_range.get('y', [-np.inf, np.inf])
            
            range_mask = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)
            x = x[range_mask]
            y = y[range_mask]
            z = z[range_mask]
            pixel_u = pixel_u[range_mask]
            pixel_v = pixel_v[range_mask]
        
        # Get colors (convert BGR to RGB and normalize to [0,1])
        colors = color_img[valid_mask][:, ::-1].astype(np.float32) / 255.0
        
        if len(x) > 0 and xyz_range is not None:
            colors = colors[range_mask]
        
        # Stack points
        points = np.stack([x, y, z], axis=-1)
        # filtered_points_3d = points.copy() # No longer needed
        
        colors = colors.astype(np.float32)
        pointcloud = np.hstack([points, colors])
        pixel_coords = np.stack([pixel_u, pixel_v], axis=-1)
        
        # Downsample to fixed number of points
        sampled_pixel_coords = None
        if len(pointcloud) > num_points:
            indices = np.random.choice(len(pointcloud), num_points, replace=False)
            pointcloud = pointcloud[indices]
            sampled_pixel_coords = pixel_coords[indices]
        elif len(pointcloud) < num_points:
            # Pad with zeros if not enough points
            padding = np.zeros((num_points - len(pointcloud), 6))
            pointcloud = np.vstack([pointcloud, padding])
            if len(pixel_coords) > 0:
                sampled_pixel_coords = pixel_coords
        else:
            sampled_pixel_coords = pixel_coords
        
        return pointcloud.astype(np.float32), sampled_pixel_coords
    
    def _update_3d_pointcloud_vis(self, pointcloud):
        """Update the 3D matplotlib visualization with sampled pointcloud."""
        if self.vis_pcd_ax is None:
            return

        xyz = pointcloud[:, :3]
        rgb = pointcloud[:, 3:]
        
        # Filter out padding zeros
        valid = ~np.all(xyz == 0, axis=1)
        xyz = xyz[valid]
        rgb = rgb[valid]
        
        if len(xyz) == 0:
            return

        self.vis_pcd_ax.clear()
        
        self.vis_pcd_ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=rgb, s=20, marker='.')
        
        self.vis_pcd_ax.set_xlabel('X')
        self.vis_pcd_ax.set_ylabel('Y')
        self.vis_pcd_ax.set_zlabel('Z')
        
        # Use ranges from args to fix view
        self.vis_pcd_ax.set_xlim(self.args.x_range)
        self.vis_pcd_ax.set_ylim(self.args.y_range)
        self.vis_pcd_ax.set_zlim(self.args.z_range)
        
        plt.pause(0.001)

    def _matrix_to_pose7(self, T):
        """Convert 4x4 transformation matrix to 7-element pose [x,y,z,qw,qx,qy,qz]."""
        if T is None:
            return np.zeros(7)
        pos = T[:3, 3]
        quat = t3d.quaternions.mat2quat(T[:3, :3])  # Returns [qw, qx, qy, qz]
        return np.concatenate([pos, quat])
    
    def _normalize_gripper(self, width):
        """Normalize gripper width to [0, 1] range. 0=closed, 1=open."""
        if width is None:
            return 0.0
        return np.clip(width / self.gripper_max_width, 0.0, 1.0)

    def control_loop(self):
        """Sync Master to Follower and read states."""
        dt = 1.0 / self.args.freq
        while self.running:
            t_start = time.time()
            try:
                # 1. Teleoperation Sync
                m1_j = self.master_l.get_joint_positions()
                m2_j = self.master_r.get_joint_positions()
                m1_g = self.master_l.get_gripper_width(teacher=True)
                m2_g = self.master_r.get_gripper_width(teacher=True)

                if m1_j is not None:
                    self.follower_l.set_joint_positions(m1_j)
                    self.follower_l.set_gripper_width(m1_g)
                if m2_j is not None:
                    self.follower_r.set_joint_positions(m2_j)
                    self.follower_r.set_gripper_width(m2_g)

                # 2. Read Follower States for Data Collection
                f1_j = self.follower_l.get_joint_positions()
                f2_j = self.follower_r.get_joint_positions()
                f1_g = self.follower_l.get_gripper_width()
                f2_g = self.follower_r.get_gripper_width()
                f1_pose = self.follower_l.get_gripper_pose()  # 4x4 matrix
                f2_pose = self.follower_r.get_gripper_pose()

                with self.state_lock:
                    if f1_j is not None:
                        self.current_state["qpos_L"] = np.array(f1_j)
                    if f2_j is not None:
                        self.current_state["qpos_R"] = np.array(f2_j)
                    # Normalize gripper width to [0, 1]
                    self.current_state["width_L"] = self._normalize_gripper(f1_g)
                    self.current_state["width_R"] = self._normalize_gripper(f2_g)
                    # Convert 4x4 matrix to 7-element pose
                    if f1_pose is not None:
                        self.current_state["pose_L"] = self._matrix_to_pose7(np.array(f1_pose))
                    if f2_pose is not None:
                        self.current_state["pose_R"] = self._matrix_to_pose7(np.array(f2_pose))

            except Exception as e:
                # print(f"Control loop warning: {e}")
                pass
            
            elapsed = time.time() - t_start
            if elapsed < dt:
                time.sleep(dt - elapsed)

    def start_recording(self):
        self.current_episode_dir, self.current_episode_idx = get_next_episode_dir(self.save_root)
        os.makedirs(self.current_episode_dir, exist_ok=True)
        self.frame_idx = 0
        self.last_record_time = 0.0
        self.is_recording = True
        print(f"\n>>> [REC] Started Episode {self.current_episode_idx} in {self.current_episode_dir}")

    def stop_and_save(self):
        if self.is_recording:
            print(f"\n>>> [STOP] Episode {self.current_episode_idx} saved with {self.frame_idx} frames.")
            self.is_recording = False

    def go_home(self):
        print(">>> Homing arms...")
        self.master_l.go_home()
        self.master_r.go_home()
        self.follower_l.go_home()
        self.follower_r.go_home()
        time.sleep(1.0)  # Wait for homing to complete
        print(">>> Gravity Compensation Enabled.")
        self.master_l.arm.gravity_compensation()
        self.master_r.arm.gravity_compensation()

    def run(self):
        self.go_home()
        
        # Start Control Thread
        t = threading.Thread(target=self.control_loop, daemon=True)
        t.start()

        print("\n" + "="*50)
        print("  Controls:")
        print("  [Space] Start/Stop Recording")
        print("  [H]     Re-Home & Gravity Comp")
        print("  [Q]     Quit")
        print("="*50 + "\n")

        try:
            while self.running:
                # 1. Capture Camera
                frames = self.pipeline.wait_for_frames()
                aligned = self.align.process(frames)
                color_frame = aligned.get_color_frame()
                depth_frame = aligned.get_depth_frame()
                
                if not color_frame or not depth_frame:
                    continue
                
                color_img = np.asanyarray(color_frame.get_data()) # BGR for cv2
                depth_img = np.asanyarray(depth_frame.get_data()) # mm

        # Get current robot state for visualization
                with self.state_lock:
                    state = deepcopy(self.current_state)
                
                # Define xyz range for filtering (adjust these values based on your workspace)
                xyz_range = {
                    'x': self.args.x_range,
                    'y': self.args.y_range,
                    'z': self.args.z_range
                }
                
                # Generate point cloud with distance-based filtering
                pointcloud, sampled_pixel_coords = self._depth_to_pointcloud(
                    depth_img, color_img, num_points=self.args.pcd_num_points, 
                    xyz_range=xyz_range
                )
                
                # Update 3D visualization if enabled
                if self.args.vis_pcd:
                    self._update_3d_pointcloud_vis(pointcloud)
                
                # 2. Create visualization panels
                # Main camera view with overlays
                display_img = color_img.copy()
                
                # Add status overlays to main display
                y_offset = 30
                line_height = 25
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                
                # Episode and recording status
                if self.is_recording:
                    cv2.circle(display_img, (20, y_offset), 8, (0, 0, 255), -1)
                    cv2.putText(display_img, f"REC Episode {self.current_episode_idx} | Frame {self.frame_idx}",
                                (35, y_offset + 5), font, font_scale, (0, 0, 255), 2)
                else:
                    cv2.putText(display_img, f"IDLE | Next Episode: {self.current_episode_idx}",
                                (20, y_offset + 5), font, font_scale, (0, 255, 0), 2)
                
                # Left arm Joint Angles (6-dim)
                y_offset += line_height + 10
                qpos_L = state["qpos_L"]
                
                # Display first 3 joints
                cv2.putText(display_img, f"L J[0:6]: [{qpos_L[0]:.2f}, {qpos_L[1]:.2f}, {qpos_L[2]:.2f}, {qpos_L[3]:.2f}, {qpos_L[4]:.2f}, {qpos_L[5]:.2f}]",
                            (20, y_offset), font, font_scale, (255, 200, 0), 1)
                y_offset += line_height
                cv2.putText(display_img, f"L Grip: {state['width_L']:.3f}",
                            (20, y_offset), font, font_scale, (255, 200, 0), 1)
                
                # Right arm Joint Angles
                y_offset += line_height + 5
                qpos_R = state["qpos_R"]
                cv2.putText(display_img, f"R J[0:6]: [{qpos_R[0]:.2f}, {qpos_R[1]:.2f}, {qpos_R[2]:.2f}, {qpos_R[3]:.2f}, {qpos_R[4]:.2f}, {qpos_R[5]:.2f}]",
                            (20, y_offset), font, font_scale, (0, 200, 255), 1)
                y_offset += line_height
                cv2.putText(display_img, f"R Grip: {state['width_R']:.3f}",
                            (20, y_offset), font, font_scale, (0, 200, 255), 1)
                
                # Show both windows
                cv2.imshow("RealSense Feed", display_img)

                # 3. Recording Logic
                if self.is_recording:
                    now = time.time()
                    if now - self.last_record_time >= self.record_interval:
                        # Convert depth to mm and float64 (RoboTwin format)
                        depth_mm = (depth_img.astype(np.float64) * self.depth_scale * 1000.0)
                        
                        # Construct RoboTwin pkl dictionary
                        pkl_dic = {
                            "observation": {
                                "head_camera": {
                                    "rgb": color_img,  # BGR uint8 (H, W, 3)
                                    "depth": depth_mm,  # Depth in mm, float64
                                    "intrinsic_cv": self.intrinsics_matrix,
                                    "extrinsic_cv": self.extrinsic_matrix,
                                    "cam2world_gl": self.cam2world_gl,
                                }
                            },
                            "pointcloud": pointcloud,  # (N, 6) xyz + rgb
                            "joint_action": {
                                "left_arm": state["qpos_L"],  # 6 DOF
                                "left_gripper": state["width_L"],  # Normalized [0,1]
                                "right_arm": state["qpos_R"],  # 6 DOF
                                "right_gripper": state["width_R"],  # Normalized [0,1]
                                # RoboTwin format: [left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)]
                                "vector": np.concatenate([
                                    state["qpos_L"],
                                    [state["width_L"]],
                                    state["qpos_R"],
                                    [state["width_R"]]
                                ]).astype(np.float32)
                            },
                            "endpose": {
                                "left_endpose": state["pose_L"],  # 7-dim [x,y,z,qw,qx,qy,qz]
                                "left_gripper": state["width_L"],
                                "right_endpose": state["pose_R"],  # 7-dim
                                "right_gripper": state["width_R"]
                            }
                        }

                        # Save PKL
                        save_path = os.path.join(self.current_episode_dir, f"{self.frame_idx}.pkl")
                        save_pkl(save_path, pkl_dic)
                        
                        self.frame_idx += 1
                        self.last_record_time = now

                # 4. Input Handling
                key = cv2.waitKey(1) & 0xFF
                if key == ord(' '):
                    if not self.is_recording:
                        self.start_recording()
                    else:
                        self.stop_and_save()
                elif key == ord('h') or key == ord('H'):
                    self.go_home()
                elif key == ord('q') or key == ord('Q'):
                    self.stop_and_save()
                    self.running = False
                    
        finally:
            self.running = False
            try:
                self.pipeline.stop()
            except: pass
            cv2.destroyAllWindows()
            print("Done.")

def main():
    parser = argparse.ArgumentParser(description='RoboTwin Real Robot Data Collection')
    parser.add_argument('--master_l', type=str, default='can0', help='Left master arm CAN port')
    parser.add_argument('--master_r', type=str, default='can2', help='Right master arm CAN port')
    parser.add_argument('--follower_l', type=str, default='can3', help='Left follower arm CAN port')
    parser.add_argument('--follower_r', type=str, default='can1', help='Right follower arm CAN port')
    parser.add_argument('--save_dir', type=str, default='./data/real_data_robotwin_collected', help='Save root directory')
    parser.add_argument('--freq', type=float, default=50.0, help='Control loop frequency (Hz)')
    parser.add_argument('--record_freq', type=float, default=30.0, help='Data recording frequency (Hz)')
    parser.add_argument('--gripper_max_width', type=float, default=0.08, help='Max gripper width in meters for normalization')
    parser.add_argument('--pcd_num_points', type=int, default=4096, help='Number of points in point cloud')
    
    # Point cloud filtering ranges
    parser.add_argument('--x_range', type=float, nargs=2, default=[0.0, 0.4], help='X range for point cloud filtering (min max)')
    parser.add_argument('--y_range', type=float, nargs=2, default=[-0.2, 0.3], help='Y range for point cloud filtering (min max)')
    parser.add_argument('--z_range', type=float, nargs=2, default=[0.1, 1.0], help='Z range for point cloud filtering (min max)')
    parser.add_argument('--vis_pcd', action='store_true', help='Enable real-time 3D point cloud visualization')

    args = parser.parse_args()
    
    collector = RoboTwinRealCollector(args)
    collector.run()

if __name__ == "__main__":
    main()
