import sys
import os
import torch
import hydra
from omegaconf import OmegaConf
import pathlib
import argparse
import dill
import numpy as np
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add paths to sys.path
# We assume the script is run from project root or checks current file location
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# If script is in root, CURRENT_DIR is root.
ROOT_DIR = CURRENT_DIR

# Add policy paths
sys.path.append(os.path.join(ROOT_DIR, 'policy/DP3/3D-Diffusion-Policy'))
sys.path.append(os.path.join(ROOT_DIR, 'policy/DP3'))

# Now we can import from the policy codebase
try:
    from train_dp3 import TrainDP3Workspace
    from diffusion_policy_3d.env_runner.robot_runner import RobotRunner
except ImportError as e:
    print(f"Error importing DP3 modules: {e}")
    print("Make sure you are running this script from the RoboTwin-RealDeploy root directory.")
    sys.exit(1)

def get_dp3_model(ckpt_path, device='cuda:0'):
    """
    Initializes DP3 model and loads checkpoint.
    """
    # Define config path relative to the expected location
    config_dir = os.path.join(ROOT_DIR, "policy/DP3/3D-Diffusion-Policy/diffusion_policy_3d/config")
    
    # Initialize Hydra
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    with hydra.initialize(config_path=os.path.relpath(config_dir, os.getcwd()), version_base='1.2'):
        # Compose config. 
        # Using 'robot_dp3' as base config which uses 'demo_task' by default.
        # This provides standard shapes (action: 14, etc).
        cfg = hydra.compose(config_name="robot_dp3")
    
    # Configure for inference
    OmegaConf.set_struct(cfg, False)
    cfg.training.device = device
    cfg.training.use_ema = True # Default to using EMA if available, assumed common for DP3
    OmegaConf.set_struct(cfg, True)

    # Initialize Workspace (builds model architecture)
    workspace = TrainDP3Workspace(cfg)
    
    # Load Checkpoint
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")
    
    print(f"Loading checkpoint: {ckpt_path}")
    workspace.load_checkpoint(path=pathlib.Path(ckpt_path))
    
    # Select Model (EMA or regular)
    model = workspace.model
    if cfg.training.use_ema and workspace.ema_model is not None:
        print("Using EMA model.")
        model = workspace.ema_model
    else:
        print("Using standard model.")
    
    model.eval()
    model.to(device)
    
    return model, cfg

class DP3Predictor:
    def __init__(self, ckpt_path, device='cuda:0'):
        self.device = device
        self.model, self.cfg = get_dp3_model(ckpt_path, device)
        
        # Setup Runner for observation handling (stacking, etc)
        self.n_obs_steps = self.cfg.n_obs_steps
        self.n_action_steps = self.cfg.n_action_steps
        
        # RobotRunner handles observation buffer and stacking
        self.env_runner = RobotRunner(n_obs_steps=self.n_obs_steps, n_action_steps=self.n_action_steps)
        self.env_runner.reset_obs() 
        
    def predict(self, obs_dict):
        """
        Run inference.
        
        Args:
            obs_dict: Dictionary containing observations.
                      Must match shape expected by model (numpy arrays).
                      {'agent_pos': (14,), 'point_cloud': (N, 6)}
        
        Returns:
            action: Predicted action sequence.
        """
        # RobotRunner.get_action handles:
        # 1. Update obs buffer
        # 2. Stack observations (history)
        # 3. Convert to tensor and move to device
        # 4. Run model.predict_action
        
        action = self.env_runner.get_action(self.model, obs_dict)
        return action
    
    def reset(self):
        self.env_runner.reset_obs()

def load_episode_data(episode_path):
    """
    Loads all .pkl files from an episode directory.
    Sorts them numerically.
    """
    if not os.path.exists(episode_path):
        raise FileNotFoundError(f"Episode path not found: {episode_path}")
        
    pkl_files = sorted([f for f in os.listdir(episode_path) if f.endswith('.pkl')], 
                       key=lambda x: int(x.split('.')[0]))
    
    data = []
    print(f"Loading {len(pkl_files)} steps from {episode_path}...")
    for pkl_f in tqdm(pkl_files, desc="Loading Episode Data"):
        try:
            with open(os.path.join(episode_path, pkl_f), 'rb') as f:
                data.append(pickle.load(f))
        except Exception as e:
            print(f"Failed to load {pkl_f}: {e}")
            
    return data

def process_point_cloud(point_cloud, target_num_points=2048):
    """
    Simple sampling/padding for point cloud.
    """
    num_points = point_cloud.shape[0]
    if num_points == target_num_points:
        return point_cloud
    elif num_points > target_num_points:
        # Sample
        indices = np.random.choice(num_points, target_num_points, replace=False)
        return point_cloud[indices]
    else:
        # Pad with zeros
        padded_pc = np.zeros((target_num_points, 6), dtype=point_cloud.dtype)
        padded_pc[:num_points] = point_cloud
        return padded_pc

def evaluate_episode(predictor, episode_path):
    """
    Evaluates the model on a recorded episode.
    """
    print(f"\\nEvaluating on episode: {episode_path}")
    raw_data = load_episode_data(episode_path)
    if not raw_data:
        print("No data loaded.")
        return None, None
    
    T = len(raw_data)
    # We need t+1 for ground truth, so we can evaluate up to T-1
    
    predictions = []
    ground_truths = []
    
    predictor.reset()
    
    # Loop
    # Note: data[t] is state at t. data[t+1] has the action taken at t (resulting in t+1 state).
    # See process_data.py logic: action[t] = vector[t+1].
    
    for t in tqdm(range(T - 1), desc="Inference"):
        step_data = raw_data[t]
        
        # Prepare observation
        # Assuming joint_action vector is the state (agent_pos)
        agent_pos = step_data['joint_action']['vector'].astype(np.float32)
        
        # Point Cloud
        # Use existing point cloud if available or reconstruct (assuming raw data has pointcloud)
        if 'pointcloud' in step_data:
            pc = step_data['pointcloud'].astype(np.float32)
        elif 'point_cloud' in step_data:
            pc = step_data['point_cloud'].astype(np.float32)
        else:
            # print("Warning: No pointcloud field found!")
            pc = np.zeros((2048, 6), dtype=np.float32)
            
        # Resize Point Cloud
        pc = process_point_cloud(pc, target_num_points=2048) # Match model config
        
        obs_dict = {
            'agent_pos': agent_pos,
            'point_cloud': pc
        }
        
        # Predict
        action_seq = predictor.predict(obs_dict) # Shape: (H, D)
        
        # Take the first action as the immediate next step prediction
        # (Open-Loop vs Closed-Loop consideration: Here we are doing open-loop
        # single step prediction evaluation on recorded trajectory)
        pred_action = action_seq[0]
        
        predictions.append(pred_action)
        
        # Ground Truth Action
        # The action executed at state t resulted in state t+1
        gt_action = raw_data[t+1]['joint_action']['vector'].astype(np.float32)
        ground_truths.append(gt_action)
        
    predictions = np.array(predictions)
    ground_truths = np.array(ground_truths)
    
    # Calculate Metrics
    mse = np.mean((predictions - ground_truths) ** 2)
    print(f"\\nTotal MSE: {mse:.6f}")
    
    return predictions, ground_truths

def plot_results(preds, gts, save_dir="eval_results"):
    os.makedirs(save_dir, exist_ok=True)
    
    dims = preds.shape[1]
    
    # 1. Overall Error Plot
    errors = np.abs(preds - gts)
    mean_errors = np.mean(errors, axis=1) # Average over dimensions per step
    
    plt.figure(figsize=(10, 6))
    plt.plot(mean_errors, label='Mean Absolute Error')
    plt.title("Prediction Error over Time")
    plt.xlabel("Step")
    plt.ylabel("MAE")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'error_over_time.png'))
    print(f"Saved error plot to {os.path.join(save_dir, 'error_over_time.png')}")
    
    # 2. Per Dimension Plot (Just plot first 6 for clarity - Left Arm Joints usually)
    plt.figure(figsize=(15, 10))
    for i in range(min(6, dims)):
        plt.subplot(3, 2, i+1)
        plt.plot(gts[:, i], label='Ground Truth', color='black', linewidth=2)
        plt.plot(preds[:, i], label='Prediction', color='red', linestyle='--')
        plt.title(f"Dimension {i}")
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'traj_comparison_dims_0_5.png'))
    print(f"Saved trajectory plot to {os.path.join(save_dir, 'traj_comparison_dims_0_5.png')}")

    # 3. Gripper Plot (Last dimension usually)
    # Usually 14 dims: Left Arm (6), Left Gripper (1), Right Arm (6), Right Gripper (1) -> 6+1+6+1 = 14
    # Indices: 0-5 (L Arm), 6 (L Grip), 7-12 (R Arm), 13 (R Grip)
    if dims >= 14:
        plt.figure(figsize=(15, 6))
        
        plt.subplot(1, 2, 1)
        # Clipping grippers for better visualization if they are 0-1
        plt.plot(gts[:, 6], label='GT Left Gripper', color='black')
        plt.plot(preds[:, 6], label='Pred Left Gripper', color='red', linestyle='--')
        plt.title("Left Gripper (Dim 6)")
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(gts[:, 13], label='GT Right Gripper', color='black')
        plt.plot(preds[:, 13], label='Pred Right Gripper', color='red', linestyle='--')
        plt.title("Right Gripper (Dim 13)")
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'traj_comparison_grippers.png'))
        print(f"Saved gripper plot to {os.path.join(save_dir, 'traj_comparison_grippers.png')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eval DP3 with explicit checkpoint path")
    parser.add_argument("-c", "--checkpoint", type=str, required=True, help="Path to checkpoint (.ckpt)")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use (default: cuda:0)")
    parser.add_argument("-e", "--episode", type=str, help="Path to episode directory (containing .pkl files) for evaluation")
    parser.add_argument("-o", "--output", type=str, default="eval_results", help="Directory to save evaluation plots")
    
    args = parser.parse_args()
    
    try:
        predictor = DP3Predictor(args.checkpoint, args.device)
        print("\\n\033[32mDP3 Model loaded successfully!\033[0m")
        
        if args.episode:
            # Full Episode Evaluation
            preds, gts = evaluate_episode(predictor, args.episode)
            if preds is not None:
                plot_results(preds, gts, save_dir=args.output)
            
        else:
            # Dummy Check
            print("\\nRunning dummy inference check...")
            dummy_obs = {
                'agent_pos': np.zeros(14, dtype=np.float32),
                'point_cloud': np.zeros((2048, 6), dtype=np.float32)
            }
            predictor.reset()
            import time
            t0 = time.time()
            action = predictor.predict(dummy_obs)
            t1 = time.time()
            print(f"Inference time: {t1-t0:.4f}s")
            print(f"Output Action shape: {action.shape}")
            print("Model is ready for deployment. Pass --episode to evaluate on real data.")
        
    except Exception as e:
        print(f"\\n\033[31mError loading model or running inference: {e}\033[0m")
        import traceback
        traceback.print_exc()
