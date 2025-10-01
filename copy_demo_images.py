#!/usr/bin/env python3
"""
Script to copy images from dataset to examples folder based on demo_config.yaml
"""

import glob
import os
import shutil

import yaml


def load_demo_config(config_path):
    """Load demo configuration from YAML file"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def copy_images_to_examples(config, examples_base_dir):
    """Copy images based on demo config to examples folder"""

    # Extract configuration values
    data_path = config["data_path"]
    current_time = config["current_time"]
    ego_id = config["ego_id"]
    contributer_id = config["contributer_id"]
    frame_indices = config["frame_idx"]

    print("Configuration:")
    print(f"  Data path: {data_path}")
    print(f"  Time: {current_time}")
    print(f"  Ego ID: {ego_id}")
    print(f"  Contributer ID: {contributer_id}")
    print(f"  Frames: {frame_indices}")

    # Create scene folder name based on config
    scene_name = f"demo_scene_{current_time}"
    scene_dir = os.path.join(examples_base_dir, scene_name)

    # Create the scene directory
    if os.path.exists(scene_dir):
        shutil.rmtree(scene_dir)
    os.makedirs(scene_dir)
    print(f"\nCreated scene directory: {scene_dir}")

    # Paths to the vehicle directories
    dataset_dir = os.path.join(data_path, current_time)
    ego_dir = os.path.join(dataset_dir, str(ego_id))
    contributer_dir = os.path.join(dataset_dir, str(contributer_id))

    print("\nLooking for images in:")
    print(f"  Ego vehicle: {ego_dir}")
    print(f"  Contributer vehicle: {contributer_dir}")

    copied_files = []

    # Copy images for each frame
    for frame_idx in frame_indices:
        frame_str = f"{frame_idx:06d}"  # Format as 6-digit number with leading zeros

        # Look for ego vehicle images (all camera angles)
        ego_pattern = os.path.join(ego_dir, f"{frame_str}_camera*.png")
        ego_files = glob.glob(ego_pattern)

        for ego_file in ego_files:
            if os.path.exists(ego_file):
                filename = os.path.basename(ego_file)
                # Rename to include ego prefix
                new_name = f"ego_{ego_id}_{filename}"
                dest_path = os.path.join(scene_dir, new_name)
                shutil.copy2(ego_file, dest_path)
                copied_files.append(new_name)
                print(f"  ✅ Copied: {filename} -> {new_name}")

        # Look for contributer vehicle images (all camera angles)
        contributer_pattern = os.path.join(contributer_dir, f"{frame_str}_camera*.png")
        contributer_files = glob.glob(contributer_pattern)

        for contributer_file in contributer_files:
            if os.path.exists(contributer_file):
                filename = os.path.basename(contributer_file)
                # Rename to include contributer prefix
                new_name = f"contributer_{contributer_id}_{filename}"
                dest_path = os.path.join(scene_dir, new_name)
                shutil.copy2(contributer_file, dest_path)
                copied_files.append(new_name)
                print(f"  ✅ Copied: {filename} -> {new_name}")

    print(f"\n🎉 Successfully copied {len(copied_files)} images to {scene_name}")
    print(f"📁 Scene location: {scene_dir}")

    return scene_dir, len(copied_files)


def main():
    # Configuration
    config_path = "/users/chengpo/map-anything/demo_config.yaml"
    examples_dir = "/users/chengpo/map-anything/examples"

    # Check if config file exists
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        return False

    # Load configuration
    try:
        config = load_demo_config(config_path)
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return False

    # Create examples directory if it doesn't exist
    if not os.path.exists(examples_dir):
        os.makedirs(examples_dir)
        print(f"Created examples directory: {examples_dir}")

    # Copy images
    try:
        scene_dir, num_files = copy_images_to_examples(config, examples_dir)

        if num_files > 0:
            print("\n✅ Task completed successfully!")
            print("📋 Next steps:")
            print("1. Run the Gradio app: python scripts/gradio_app.py")
            print("2. Look for the new scene in the 'Example Scenes' section")
            print("3. Click on the thumbnail to load the demo scene")
            return True
        else:
            print(
                "\n⚠️  No images were copied. Please check the paths and frame indices."
            )
            return False

    except Exception as e:
        print(f"❌ Error copying images: {e}")
        return False


if __name__ == "__main__":
    main()
