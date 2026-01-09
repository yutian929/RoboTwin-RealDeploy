#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DP3 Real Robot Deployment Script
Deploy trained DP3 model on real robot hardware.

Usage:
    python real_deploy_dp3.py --task_name my_task --checkpoint_num 500 --expert_data_num 21
"""

import sys
import os
import numpy as np
import time
import argparse
import threading
from copy import deepcopy

# Add paths
sys.path.append('./')
sys.path.append('./policy/DP3')
sys.path.append('./policy/DP3/3D-Diffusion-Policy')

import cv2
import pyrealsense2 as rs
import transforms3d as t3d

# ============== VISUALIZATION CONFIG ==============
# Which arm to visualize: "left" or "right"
VIS_ARM = "right"

# Whether to require user to press SPACE before executing each action chunk
REQUIRE_SPACE_TO_EXECUTE = False
# ==================================================

# Hardware Interface
try:
    from fucking_arx_mujoco.real.real_single_arm import RealSingleArm
except ImportError:
    print("[Error] Could not import robot interface.")
    sys.exit(1)


class RealDP3Deployer:
    def __init__(self, args):
        self.args = args
        self.running = True
        
        # Gripper normalization
        self.gripper_max_width = args.gripper_max_width
        
        # Robot state buffer
        self.state_lock = threading.Lock()
        self.current_state = {
            "qpos_L": np.zeros(6), "width_L": 0.0,
            "qpos_R": np.zeros(6), "width_R": 0.0,
        }
        
        # Async Inference State
        self.inference_lock = threading.Lock()
        self.vis_lock = threading.Lock()
        self.latest_plan = None
        self.latest_vis_img = None
        self.latest_obs_disp = None # For display text
        self.plan_ready_event = threading.Event()
        
        # Initialize hardware
        self._init_arms()
        self._init_camera()
        
        # Load DP3 model
        self._load_model()
        
        # Start Inference Thread
        self.inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        # self.inference_thread.start() # Start in run()

    def _init_arms(self):
        print(">>> Initializing ARX Arms...")
        try:
            self.left_arm = RealSingleArm(
                can_port=self.args.follower_l, 
                arm_type=0, 
                max_velocity=300, 
                max_acceleration=800
            )
            self.right_arm = RealSingleArm(
                can_port=self.args.follower_r, 
                arm_type=0, 
                max_velocity=300, 
                max_acceleration=800
            )
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
            
            color_stream = profile.get_stream(rs.stream.color)
            intr = color_stream.as_video_stream_profile().get_intrinsics()
            self.fx, self.fy = intr.fx, intr.fy
            self.cx, self.cy = intr.ppx, intr.ppy
            
            depth_sensor = profile.get_device().first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()
            
            self.align = rs.align(rs.stream.color)
            print(">>> RealSense Initialized.")
        except Exception as e:
            print(f"[Error] Camera init failed: {e}")
            sys.exit(1)
            
    def _load_model(self):
        print(">>> Loading DP3 Model...")
        
        # Import DP3 components
        from policy.DP3.deploy_policy import get_model, reset_model
        
        usr_args = {
            'task_name': self.args.task_name,
            'config_name': 'robot_dp3',
            'checkpoint_num': self.args.checkpoint_num,
            'expert_data_num': self.args.expert_data_num,
            'ckpt_setting': self.args.ckpt_setting,
            'seed': self.args.seed,
            'use_rgb': False,
        }
        
        self.model = get_model(usr_args)
        self.reset_model = reset_model
        print(">>> DP3 Model Loaded.")
        
    def _depth_to_pointcloud(self, depth_img, color_img, num_points=1024, xyz_range=None):
        """Convert depth image to point cloud."""
        depth_m = depth_img.astype(np.float32) * self.depth_scale
        
        v, u = np.meshgrid(
            np.arange(self.img_height),
            np.arange(self.img_width),
            indexing='ij'
        )
        
        if xyz_range is None:
            valid_mask = (depth_m > 0.1) & (depth_m < 2.0)
        else:
            z_min, z_max = xyz_range.get('z', [0.1, 2.0])
            valid_mask = (depth_m > z_min) & (depth_m < z_max)
        
        z = depth_m[valid_mask]
        x = (u[valid_mask] - self.cx) * z / self.fx
        y = (v[valid_mask] - self.cy) * z / self.fy
        
        colors = color_img[valid_mask][:, ::-1].astype(np.float32) / 255.0
        
        if xyz_range is not None:
            x_min, x_max = xyz_range.get('x', [-np.inf, np.inf])
            y_min, y_max = xyz_range.get('y', [-np.inf, np.inf])
            
            range_mask = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)
            x = x[range_mask]
            y = y[range_mask]
            z = z[range_mask]
            colors = colors[range_mask]
        
        points = np.stack([x, y, z], axis=-1)
        pointcloud = np.hstack([points, colors])
        
        if len(pointcloud) > num_points:
            indices = np.random.choice(len(pointcloud), num_points, replace=False)
            pointcloud = pointcloud[indices]
        elif len(pointcloud) < num_points:
            padding = np.zeros((num_points - len(pointcloud), 6))
            pointcloud = np.vstack([pointcloud, padding])
        
        return pointcloud.astype(np.float32)
    
    def _normalize_gripper(self, width):
        if width is None:
            return 0.0
        return np.clip(width / self.gripper_max_width, 0.0, 1.0)
        
    def get_observation(self):
        """Get current observation for policy."""
        # Get camera frames
        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        
        color_img = np.asanyarray(color_frame.get_data())
        depth_img = np.asanyarray(depth_frame.get_data())
        
        # Generate point cloud
        xyz_range = {
            'x': self.args.x_range,
            'y': self.args.y_range,
            'z': self.args.z_range
        }
        pointcloud = self._depth_to_pointcloud(depth_img, color_img, num_points=self.args.pcd_num_points, xyz_range=xyz_range)
        
        # Get robot state
        left_qpos = np.array(self.left_arm.get_joint_positions() or np.zeros(6))
        right_qpos = np.array(self.right_arm.get_joint_positions() or np.zeros(6))
        left_gripper = self._normalize_gripper(self.left_arm.get_gripper_width())
        right_gripper = self._normalize_gripper(self.right_arm.get_gripper_width())
        
        # Construct observation (DP3 format)
        obs = {
            'agent_pos': np.concatenate([
                left_qpos, [left_gripper],
                right_qpos, [right_gripper]
            ]).astype(np.float32),
            'point_cloud': pointcloud
        }
        
        return obs, color_img
    
    def execute_action(self, action):
        """Execute a single action step."""
        # Parse action vector (14-dim)
        left_qpos = action[:6]
        left_gripper = action[6]
        right_qpos = action[7:13]
        right_gripper = action[13]
        
        # Send commands to robot
        self.left_arm.set_joint_positions(left_qpos.tolist())
        self.left_arm.set_gripper_width(left_gripper * self.gripper_max_width)
        self.right_arm.set_joint_positions(right_qpos.tolist())
        self.right_arm.set_gripper_width(right_gripper * self.gripper_max_width)

    def _draw_action_visualization(self, display_img, actions, current_action_idx, step):
        """
        Draw predicted action sequence visualization on the image.
        
        Args:
            display_img: Image to draw on (modified in-place)
            actions: Array of predicted actions (N, 14)
            current_action_idx: Index of the action being executed (highlighted in green)
            step: Current step number
        """
        # Select which arm to visualize based on VIS_ARM config
        if VIS_ARM == "left":
            arm_slice = slice(0, 6)
            gripper_idx = 6
            arm_label = "Left Arm"
        else:  # right
            arm_slice = slice(7, 13)
            gripper_idx = 13
            arm_label = "Right Arm"
        
        # Drawing parameters
        panel_x = 10
        panel_y = 50
        row_height = 18
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        
        # Draw mode indicator
        mode_text = "Mode: SPACE to Execute" if REQUIRE_SPACE_TO_EXECUTE else "Mode: Auto Execute"
        mode_color = (0, 255, 255) if REQUIRE_SPACE_TO_EXECUTE else (0, 255, 0)
        cv2.putText(display_img, mode_text, (panel_x, 20), font, 0.5, mode_color, 1)
        
        # Draw title
        cv2.putText(display_img, f"Step: {step} | {arm_label}", 
                    (panel_x, panel_y - 10), font, 0.5, (255, 255, 255), 1)
        
        # Draw each action in the sequence
        for action_idx, action in enumerate(actions):
            # Check if this is the current action to execute
            is_current = (action_idx == current_action_idx)
            
            # Get arm values
            arm_values = action[arm_slice]
            gripper_value = action[gripper_idx]
            
            # Format: "Action N: [j1, j2, j3, j4, j5, j6, grip]"
            values_str = f"[{arm_values[0]:+.2f}, {arm_values[1]:+.2f}, {arm_values[2]:+.2f}, {arm_values[3]:+.2f}, {arm_values[4]:+.2f}, {arm_values[5]:+.2f}, {gripper_value:.2f}]"
            
            if is_current:
                text_color = (0, 255, 0)  # Green for current action
                action_text = f">>> A{action_idx}: {values_str}"
            else:
                text_color = (200, 200, 200)  # Gray for future actions
                action_text = f"    A{action_idx}: {values_str}"
            
            y_row = panel_y + action_idx * row_height
            cv2.putText(display_img, action_text, (panel_x, y_row), 
                        font, font_scale, text_color, 1)
        
        # Instructions at bottom
        inst_y = display_img.shape[0] - 10
        if REQUIRE_SPACE_TO_EXECUTE:
            cv2.putText(display_img, "[SPACE] Execute | [Q] Quit", 
                        (panel_x, inst_y), font, 0.5, (0, 255, 255), 1)
        else:
            cv2.putText(display_img, "[Q] Quit", 
                        (panel_x, inst_y), font, 0.5, (0, 255, 255), 1)
        
    def _inference_loop(self):
        """Background thread for continuous inference."""
        print(">>> Inference Thread Started.")
        while self.running:
            try:
                # 1. Get observation (accessing camera)
                # Note: get_observation consumes a frame from pipeline
                obs, color_img = self.get_observation()
                
                # Update Visualization buffer immediately
                with self.vis_lock:
                    self.latest_vis_img = color_img.copy()
                
                # 2. Update Model Obs
                # Assuming update_obs handles history internally
                self.model.update_obs(obs)
                
                # 3. Model Inference
                actions = self.model.get_action()
                
                # 4. Update Shared Plan
                with self.inference_lock:
                    self.latest_plan = actions
                    self.plan_ready_event.set()
                
                # Small sleep to prevent pegging CPU if inference is excessively fast (unlikely for DP3)
                # time.sleep(0.001) 
                
            except Exception as e:
                print(f"[Inference Error] {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.1)

    def run(self, max_steps=300):
        """Main deployment loop (Execution Thread)."""
        print("\n" + "="*50)
        print("  DP3 Real Robot Deployment (Async)")
        print(f"  Visualizing: {VIS_ARM.upper()} arm")
        print(f"  Execution Steps per Plan: {self.args.exec_steps}")
        print(f"  Execution Mode: {'Manual (SPACE)' if REQUIRE_SPACE_TO_EXECUTE else 'Auto'}")
        print("  Press 'Q' to quit")
        print("="*50 + "\n")
        
        self.reset_model(self.model)
        
        # Start Inference
        self.inference_thread.start()
        
        # Wait for first plan
        print("Waiting for first plan...")
        while self.running:
            self.plan_ready_event.wait(timeout=0.1)
            if self.plan_ready_event.is_set():
                break
            # checking quit
            with self.vis_lock:
                if self.latest_vis_img is not None:
                     cv2.imshow("DP3 Deployment", self.latest_vis_img)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                self.running = False
                return

        step = 0
        
        try:
            while self.running and step < max_steps:
                
                # 1. Wait for fresh plan (non-blocking loop for UI responsiveness)
                while self.running and not self.plan_ready_event.is_set():
                     # Update visualization while waiting
                     with self.vis_lock:
                        if self.latest_vis_img is not None:
                             cv2.imshow("DP3 Deployment", self.latest_vis_img)
                     if (cv2.waitKey(1) & 0xFF) == ord('q'):
                        self.running = False
                        return
                     time.sleep(0.001)

                if not self.running: break

                # 2. Consume the plan
                with self.inference_lock:
                    current_plan = self.latest_plan
                    # Mark as consumed so we wait for next update
                    self.plan_ready_event.clear() 
                
                # Get Visualization Image
                with self.vis_lock:
                    display_img = self.latest_vis_img.copy() if self.latest_vis_img is not None else np.zeros((480, 640, 3), dtype=np.uint8)

                # Visualize Plan (using the plan we are about to execute)
                self._draw_action_visualization(display_img, current_plan, current_action_idx=-1, step=step)
                cv2.imshow("DP3 Deployment", display_img)
                
                # If manual mode, wait for SPACE to execute
                if REQUIRE_SPACE_TO_EXECUTE:
                    print(f"Step {step}: Press SPACE to execute {self.args.exec_steps} steps...")
                    while True:
                        key = cv2.waitKey(30) & 0xFF
                        if key == ord(' '):
                            break
                        elif key == ord('q'):
                            self.running = False
                            break
                        
                        # Continuously update display from inference thread
                        with self.vis_lock:
                            if self.latest_vis_img is not None:
                                display_img = self.latest_vis_img.copy()
                        
                        # Show potential plan
                        with self.inference_lock:
                            vis_plan = self.latest_plan
                            # If plan updated while waiting, visualize NEW plan but we still hold 'current_plan' for execution?
                            # Actually, if we waited for space, we might want to refresh 'current_plan' to the very latest inference?
                            # Yes, that would be better "Real Time" behavior.
                            if vis_plan is not self.latest_plan: # Reference check
                                # This logic is tricky, let's just show what we HAVE in 'vis_plan'
                                pass

                        self._draw_action_visualization(display_img, vis_plan, current_action_idx=-1, step=step)
                        cv2.putText(display_img, "WAITING FOR SPACE...", 
                                    (display_img.shape[1]//2 - 100, display_img.shape[0]//2), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.imshow("DP3 Deployment", display_img)
                    
                    if not self.running:
                        break
                        
                    # Refresh plan after Space if possible? 
                    # If we waited 5 seconds, the 'current_plan' we grabbed is 5 seconds old.
                    # We should grab the latest one again.
                    with self.inference_lock:
                        current_plan = self.latest_plan
                        self.plan_ready_event.clear()

                
                # Execute 'exec_steps' from the plan
                # Usually we execute the FIRST N steps of the generated plan
                # because the plan starts from "current state" (at time of inference).
                
                n_execute = min(self.args.exec_steps, len(current_plan))
                
                for i in range(n_execute):
                    if not self.running:
                        break
                        
                    action = current_plan[i]
                    
                    # Visualization with current action highlighted
                    with self.vis_lock:
                        if self.latest_vis_img is not None:
                            display_img = self.latest_vis_img.copy()
                            
                    self._draw_action_visualization(display_img, current_plan, current_action_idx=i, step=step)
                    cv2.imshow("DP3 Deployment", display_img)
                    
                    # Execute action
                    self.execute_action(action)
                    time.sleep(1.0 / 30.0)  # ~30Hz control loop pacing
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        self.running = False
                        break
                        
                    step += 1
                
                print(f"Executed {n_execute} steps. Total: {step}/{max_steps}", end='\r')
                
        finally:
            self.running = False # Stop inference thread
            if self.inference_thread.is_alive():
                self.inference_thread.join(timeout=1.0)
            self.cleanup()
            
        print(f"\nDeployment complete. Total steps: {step}")
        
    def cleanup(self):
        self.pipeline.stop()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='DP3 Real Robot Deployment')
    parser.add_argument('--task_name', type=str, required=True, help='Task name')
    parser.add_argument('--checkpoint_num', type=int, default=500, help='Checkpoint number')
    parser.add_argument('--expert_data_num', type=int, required=True, help='Number of expert episodes')
    parser.add_argument('--ckpt_setting', type=str, default='default', help='Checkpoint setting')
    parser.add_argument('--seed', type=int, default=0, help='Seed')
    parser.add_argument('--follower_l', type=str, default='can3', help='Left follower CAN port')
    parser.add_argument('--follower_r', type=str, default='can1', help='Right follower CAN port')
    parser.add_argument('--gripper_max_width', type=float, default=0.08, help='Max gripper width')
    parser.add_argument('--max_steps', type=int, default=300, help='Max deployment steps')
    parser.add_argument('--exec_steps', type=int, default=4, help='Number of steps to execute per inference')
    parser.add_argument('--pcd_num_points', type=int, default=2048, help='Number of points in point cloud (inference only)')
    
    # Point cloud filtering ranges (Matched with data collection)
    parser.add_argument('--x_range', type=float, nargs=2, default=[-0.1940, 0.1570], help='X range for point cloud filtering (min max)')
    parser.add_argument('--y_range', type=float, nargs=2, default=[-0.0380, 0.0870], help='Y range for point cloud filtering (min max)')
    parser.add_argument('--z_range', type=float, nargs=2, default=[0.5930, 0.9600], help='Z range for point cloud filtering (min max)')

    args = parser.parse_args()
    
    deployer = RealDP3Deployer(args)
    deployer.run(max_steps=args.max_steps)


if __name__ == "__main__":
    main()
