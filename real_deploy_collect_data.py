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

    def _depth_to_pointcloud(self, depth_img, color_img, num_points=1024):
        """
        Convert depth image to point cloud with colors.
        
        Args:
            depth_img: Raw depth image (uint16, in depth units)
            color_img: BGR color image (uint8)
            num_points: Number of points to sample (default 1024 for DP3)
            
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
        
        # Valid depth mask (filter out invalid depth values)
        valid_mask = (depth_m > 0.1) & (depth_m < 2.0)  # 0.1m to 2m range
        
        # Back-project to 3D
        z = depth_m[valid_mask]
        x = (u[valid_mask] - self.cx) * z / self.fx
        y = (v[valid_mask] - self.cy) * z / self.fy
        
        # Get pixel coordinates for visualization
        pixel_u = u[valid_mask]
        pixel_v = v[valid_mask]
        
        # Get colors (convert BGR to RGB and normalize to [0,1])
        colors = color_img[valid_mask][:, ::-1].astype(np.float32) / 255.0
        
        # Stack points
        points = np.stack([x, y, z], axis=-1)
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

    def _create_pointcloud_visualization(self, color_img, sampled_pixel_coords):
        """Create a visualization image showing sampled point cloud points on the RGB image."""
        vis_img = color_img.copy()
        if sampled_pixel_coords is not None and len(sampled_pixel_coords) > 0:
            for u, v in sampled_pixel_coords.astype(int):
                if 0 <= u < self.img_width and 0 <= v < self.img_height:
                    cv2.circle(vis_img, (u, v), 1, (0, 255, 0), -1)  # Green dots
        return vis_img

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
                
                # Generate point cloud for visualization
                pointcloud, sampled_pixel_coords = self._depth_to_pointcloud(
                    depth_img, color_img, num_points=self.args.pcd_num_points
                )
                
                # 2. Create visualization panels
                # Main camera view with overlays
                display_img = color_img.copy()
                
                # Point cloud visualization (sampled points on image)
                pcd_vis_img = self._create_pointcloud_visualization(color_img, sampled_pixel_coords)
                
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
                
                # Left arm 7D pose (position + quaternion)
                y_offset += line_height + 10
                pose_L = state["pose_L"]
                cv2.putText(display_img, f"L Pose: [{pose_L[0]:.3f}, {pose_L[1]:.3f}, {pose_L[2]:.3f}]",
                            (20, y_offset), font, font_scale, (255, 200, 0), 1)
                y_offset += line_height
                cv2.putText(display_img, f"L Quat: [{pose_L[3]:.3f}, {pose_L[4]:.3f}, {pose_L[5]:.3f}, {pose_L[6]:.3f}]",
                            (20, y_offset), font, font_scale, (255, 200, 0), 1)
                y_offset += line_height
                cv2.putText(display_img, f"L Grip: {state['width_L']:.3f}",
                            (20, y_offset), font, font_scale, (255, 200, 0), 1)
                
                # Right arm 7D pose
                y_offset += line_height + 5
                pose_R = state["pose_R"]
                cv2.putText(display_img, f"R Pose: [{pose_R[0]:.3f}, {pose_R[1]:.3f}, {pose_R[2]:.3f}]",
                            (20, y_offset), font, font_scale, (0, 200, 255), 1)
                y_offset += line_height
                cv2.putText(display_img, f"R Quat: [{pose_R[3]:.3f}, {pose_R[4]:.3f}, {pose_R[5]:.3f}, {pose_R[6]:.3f}]",
                            (20, y_offset), font, font_scale, (0, 200, 255), 1)
                y_offset += line_height
                cv2.putText(display_img, f"R Grip: {state['width_R']:.3f}",
                            (20, y_offset), font, font_scale, (0, 200, 255), 1)
                
                # Point cloud info on pcd visualization
                cv2.putText(pcd_vis_img, f"Point Cloud: {self.args.pcd_num_points} pts",
                            (20, 30), font, font_scale, (0, 255, 0), 1)
                
                # Show both windows
                cv2.imshow("RealSense Feed", display_img)
                cv2.imshow("Point Cloud Sampling", pcd_vis_img)

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
    parser.add_argument('--pcd_num_points', type=int, default=1024, help='Number of points in point cloud')
    
    args = parser.parse_args()
    
    collector = RoboTwinRealCollector(args)
    collector.run()

if __name__ == "__main__":
    main()
