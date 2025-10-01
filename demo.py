# Optional config for better memory efficiency
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Required imports
import struct

import numpy as np
import torch
import yaml

from mapanything.models import MapAnything
from mapanything.utils.image import load_images


def pack_rgb(r, g, b):
    """Pack r, g, b into a single float32"""
    rgb_int = (int(r) << 16) | (int(g) << 8) | int(b)
    return struct.unpack("f", struct.pack("I", rgb_int))[0]


def generate_pcd(pts3d, rgb, mask, filename):
    pts3d = pts3d.cpu().numpy()
    rgb = (rgb * 255).cpu().numpy().astype(np.uint8)
    mask = mask.cpu().numpy().astype(bool)

    print(rgb[30:40])

    pts3d = pts3d[mask]
    rgb = rgb[mask]

    print(pts3d.shape, rgb.shape, mask.shape)
    """Generate a PLY point cloud file from 3D points and RGB colors."""
    with open(filename, "w") as f:
        # Header
        f.write(
            """# .PCD v0.7 - Point Cloud Data file format
VERSION 0.7
FIELDS x y z rgb
SIZE 4 4 4 4
TYPE F F F F
COUNT 1 1 1 1
WIDTH {}
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS {}
DATA ascii
""".format(len(pts3d), len(pts3d))
        )
        # Data
        for pt, color in zip(pts3d, rgb):
            rgb_float = pack_rgb(*color)
            f.write(f"{pt[0]} {pt[1]} {pt[2]} {rgb_float}\n")


# Get inference device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Init model - This requries internet access or the huggingface hub cache to be pre-downloaded
# For Apache 2.0 license model, use "facebook/map-anything-apache"
model = MapAnything.from_pretrained("facebook/map-anything").to(device)

# Load images from OPV2V
with open("demo_config.yaml", "r") as f:
    config = yaml.safe_load(f)

cam_id_list = [0, 3]
image_names_ori = [
    f"{idx:06d}_camera{cam_idx}.png"
    for idx in config["frame_idx"]
    for cam_idx in cam_id_list
]

image_names = [
    os.path.join(
        config["data_path"], config["current_time"], str(config["ego_id"]), img_name
    )
    for img_name in image_names_ori
]
# image_names += [os.path.join(config["data_path"], config["current_time"], str(config["contributer_id"]), img_name) for img_name in image_names_ori]

views = load_images(image_names)

# Run inference
predictions = model.infer(
    views,  # Input views
    memory_efficient_inference=False,  # Trades off speed for more views (up to 2000 views on 140 GB)
    use_amp=True,  # Use mixed precision inference (recommended)
    amp_dtype="bf16",  # bf16 inference (recommended; falls back to fp16 if bf16 not supported)
    apply_mask=True,  # Apply masking to dense geometry outputs
    mask_edges=True,  # Remove edge artifacts by using normals and depth
    apply_confidence_mask=False,  # Filter low-confidence regions
    confidence_percentile=10,  # Remove bottom 10 percentile confidence pixels
)
print(type(predictions))
pts3d_all = torch.cat([pred["pts3d"] for pred in predictions], dim=0)
pts3d_all = pts3d_all.reshape(-1, 3)  # (B*H*W, 3)

mask_all = torch.cat([pred["mask"] for pred in predictions], dim=0)
mask_all = mask_all.reshape(-1)  # (B*H*W,)

rgb_all = torch.cat([view["img"] for view in views], dim=0)
print(rgb_all[0])
rgb_all = rgb_all.permute(0, 2, 3, 1).reshape(-1, 3)  # (B*H*W, 3)


generate_pcd(pts3d_all, rgb_all, mask_all, "output.pcd")

# Access results for each view - Complete list of metric outputs
# for i, pred in enumerate(predictions):
#     # Geometry outputs
#     pts3d = pred["pts3d"]                     # 3D points in world coordinates (B, H, W, 3)
#     pts3d_cam = pred["pts3d_cam"]             # 3D points in camera coordinates (B, H, W, 3)
#     depth_z = pred["depth_z"]                 # Z-depth in camera frame (B, H, W, 1)
#     depth_along_ray = pred["depth_along_ray"] # Depth along ray in camera frame (B, H, W, 1)

#     # Camera outputs
#     ray_directions = pred["ray_directions"]   # Ray directions in camera frame (B, H, W, 3)
#     intrinsics = pred["intrinsics"]           # Recovered pinhole camera intrinsics (B, 3, 3)
#     camera_poses = pred["camera_poses"]       # OpenCV (+X - Right, +Y - Down, +Z - Forward) cam2world poses in world frame (B, 4, 4)
#     cam_trans = pred["cam_trans"]             # OpenCV (+X - Right, +Y - Down, +Z - Forward) cam2world translation in world frame (B, 3)
#     cam_quats = pred["cam_quats"]             # OpenCV (+X - Right, +Y - Down, +Z - Forward) cam2world quaternion in world frame (B, 4)

#     # Quality and masking
#     confidence = pred["conf"]                 # Per-pixel confidence scores (B, H, W)
#     mask = pred["mask"]                       # Combined validity mask (B, H, W, 1)
#     non_ambiguous_mask = pred["non_ambiguous_mask"]                # Non-ambiguous regions (B, H, W)
#     non_ambiguous_mask_logits = pred["non_ambiguous_mask_logits"]  # Mask logits (B, H, W)

#     # Scaling
#     metric_scaling_factor = pred["metric_scaling_factor"]  # Applied metric scaling (B,)

#     # Original input
#     img_no_norm = pred["img_no_norm"]         # Denormalized input images for visualization (B, H, W, 3)
