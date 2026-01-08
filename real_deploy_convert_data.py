#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert real robot collected PKL data to HDF5 format for DP3 training.
This script avoids importing sapien by implementing conversion functions directly.

Usage:
    python real_deploy_convert_data.py --task_name my_task --task_config default
    
Output structure (compatible with DP3 process_data.sh):
    ./data/{task_name}/{task_config}/data/episode{N}.hdf5
    ./data/{task_name}/{task_config}/video/episode{N}.mp4

Then run DP3 training:
    cd policy/DP3
    bash process_data.sh {task_name} {task_config} {num_episodes}
    bash train.sh {task_name} {task_config} {num_episodes} {seed} {gpu_id}
"""

import os
import pickle
import numpy as np
import h5py
import cv2
import argparse


def images_encoding(imgs):
    """Encode images to JPEG bytes for HDF5 storage."""
    encode_data = []
    max_len = 0
    for i in range(len(imgs)):
        success, encoded_image = cv2.imencode(".jpg", imgs[i])
        jpeg_data = encoded_image.tobytes()
        encode_data.append(jpeg_data)
        max_len = max(max_len, len(jpeg_data))
    # Padding to same length
    padded_data = []
    for i in range(len(imgs)):
        padded_data.append(encode_data[i].ljust(max_len, b"\0"))
    return padded_data, max_len


def images_to_video(images, out_path, fps=30):
    """Convert image sequence to video."""
    if len(images) == 0:
        print(f"Warning: No images to create video at {out_path}")
        return
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    h, w = images[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    
    for img in images:
        # Ensure BGR format for cv2
        if img.shape[-1] == 3:
            out.write(img)
        else:
            out.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    out.release()
    print(f"  Video saved: {out_path}")


def parse_dict_structure(data):
    """Parse dictionary structure to create empty lists for accumulation."""
    if isinstance(data, dict):
        parsed = {}
        for key, value in data.items():
            if isinstance(value, dict):
                parsed[key] = parse_dict_structure(value)
            elif isinstance(value, np.ndarray):
                parsed[key] = []
            else:
                parsed[key] = []
        return parsed
    else:
        return []


def append_data_to_structure(data_structure, data):
    """Append data to the accumulated structure."""
    for key in data_structure:
        if key in data:
            if isinstance(data_structure[key], list):
                data_structure[key].append(data[key])
            elif isinstance(data_structure[key], dict):
                append_data_to_structure(data_structure[key], data[key])


def load_pkl_file(pkl_path):
    """Load a pickle file."""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return data


def create_hdf5_from_dict(hdf5_group, data_dict):
    """Recursively create HDF5 datasets from dictionary."""
    for key, value in data_dict.items():
        if isinstance(value, dict):
            subgroup = hdf5_group.create_group(key)
            create_hdf5_from_dict(subgroup, value)
        elif isinstance(value, list):
            value = np.array(value)
            if "rgb" in key:
                encode_data, max_len = images_encoding(value)
                hdf5_group.create_dataset(key, data=encode_data, dtype=f"S{max_len}")
            else:
                hdf5_group.create_dataset(key, data=value)


def pkl_files_to_hdf5_and_video(pkl_files, hdf5_path, video_path):
    """Convert PKL files to HDF5 and video."""
    # Parse structure from first file
    data_list = parse_dict_structure(load_pkl_file(pkl_files[0]))
    
    # Accumulate all data
    for pkl_file_path in pkl_files:
        pkl_file = load_pkl_file(pkl_file_path)
        append_data_to_structure(data_list, pkl_file)
    
    # Create video from RGB images
    rgb_images = np.array(data_list["observation"]["head_camera"]["rgb"])
    images_to_video(rgb_images, video_path)
    
    # Create HDF5 file
    os.makedirs(os.path.dirname(hdf5_path), exist_ok=True)
    with h5py.File(hdf5_path, "w") as f:
        create_hdf5_from_dict(f, data_list)
    print(f"  HDF5 saved: {hdf5_path}")


def process_folder_to_hdf5_video(folder_path, hdf5_path, video_path):
    """Process a folder of PKL files to HDF5 and video."""
    # Find all PKL files with numeric names
    pkl_files = []
    for fname in os.listdir(folder_path):
        if fname.endswith(".pkl") and fname[:-4].isdigit():
            pkl_files.append((int(fname[:-4]), os.path.join(folder_path, fname)))
    
    if not pkl_files:
        raise FileNotFoundError(f"No valid .pkl files found in {folder_path}")
    
    # Sort by frame number
    pkl_files.sort()
    pkl_files = [f[1] for f in pkl_files]
    
    # Verify sequential numbering
    expected = 0
    for f in pkl_files:
        num = int(os.path.basename(f)[:-4])
        if num != expected:
            raise ValueError(f"Missing file {expected}.pkl in {folder_path}")
        expected += 1
    
    print(f"  Found {len(pkl_files)} frames")
    pkl_files_to_hdf5_and_video(pkl_files, hdf5_path, video_path)


def main():
    parser = argparse.ArgumentParser(description='Convert PKL data to HDF5 for DP3 training')
    parser.add_argument('--task_name', type=str, required=True,
                        help='Task name (used in output path and DP3 training)')
    parser.add_argument('--task_config', type=str, default='default',
                        help='Task config name (default: default)')
    parser.add_argument('--data_root', type=str, default='./data/real_data_robotwin_collected',
                        help='Root directory containing episode folders')
    args = parser.parse_args()
    
    task_name = args.task_name
    task_config = args.task_config
    data_root = args.data_root
    
    # Output directory structure: ./data/{task_name}/{task_config}/
    output_dir = f"./data/{task_name}/{task_config}"
    
    # Create output directories
    os.makedirs(f"{output_dir}/data", exist_ok=True)
    os.makedirs(f"{output_dir}/video", exist_ok=True)
    
    # Find all episode directories
    episode_dirs = sorted([d for d in os.listdir(data_root) if d.startswith("episode_")])
    
    if not episode_dirs:
        print(f"No episode directories found in {data_root}")
        return
    
    num_episodes = len(episode_dirs)
    
    print(f"Task Name: {task_name}")
    print(f"Task Config: {task_config}")
    print(f"Found {num_episodes} episodes in {data_root}")
    print(f"Output directory: {output_dir}")
    print("=" * 50)
    
    # Convert each episode
    for i, ep_dir in enumerate(episode_dirs):
        ep_path = os.path.join(data_root, ep_dir)
        hdf5_path = f"{output_dir}/data/episode{i}.hdf5"
        video_path = f"{output_dir}/video/episode{i}.mp4"
        
        print(f"\n[{i+1}/{num_episodes}] Converting {ep_dir} -> episode{i}")
        try:
            process_folder_to_hdf5_video(ep_path, hdf5_path, video_path)
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    print("\n" + "=" * 50)
    print(f"Conversion complete! {num_episodes} episodes converted.")
    print(f"HDF5 files: {output_dir}/data/")
    print(f"Videos: {output_dir}/video/")
    print("\n" + "=" * 50)
    print("Next steps for DP3 training:")
    print(f"  cd policy/DP3")
    print(f"  bash process_data.sh {task_name} {task_config} {num_episodes}")
    print(f"  bash train.sh {task_name} {task_config} {num_episodes} 0 0")


if __name__ == "__main__":
    main()
