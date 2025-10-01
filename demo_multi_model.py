# Optional config for better memory efficiency
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Required imports
import struct

import numpy as np
import torch
import yaml

from mapanything.models import MapAnything
from mapanything.utils.image import preprocess_inputs
from opv2v_data_preprocess import opv2v_data_prepare


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

imgs, intrinsics, extrinsics = opv2v_data_prepare(
    "demo_config.yaml", cam_dirs=[0], has_contributer=False
)

print(len(imgs), len(intrinsics), len(extrinsics))
views_example = [
    {
        "img": torch.tensor(imgs[i]),
        "intrinsics": torch.tensor(intrinsics[i]),
        "camera_poses": torch.tensor(extrinsics[i]),
    }
    for i in range(len(imgs))
]

# Preprocess inputs to the expected format
processed_views = preprocess_inputs(views_example)
print("🚕🚕🚕🚕", processed_views[0]["img"].shape)

# Run inference with any combination of inputs
predictions = model.infer(
    processed_views,  # Any combination of input views
    memory_efficient_inference=False,  # Trades off speed for more views (up to 2000 views on 140 GB)
    use_amp=True,  # Use mixed precision inference (recommended)
    amp_dtype="bf16",  # bf16 inference (recommended; falls back to fp16 if bf16 not supported)
    apply_mask=True,  # Apply masking to dense geometry outputs
    mask_edges=True,  # Remove edge artifacts by using normals and depth
    apply_confidence_mask=False,  # Filter low-confidence regions
    confidence_percentile=10,  # Remove bottom 10 percentile confidence pixels
    # Control which inputs to use/ignore
    # By default, all inputs are used when provided
    # If is_metric_scale flag is not provided, all inputs are assumed to be in metric scale
    ignore_calibration_inputs=False,
    ignore_depth_inputs=False,
    ignore_pose_inputs=False,
    ignore_depth_scale_inputs=False,
    ignore_pose_scale_inputs=False,
)

pts3d_all = torch.cat([pred["pts3d"] for pred in predictions], dim=0)
print(pts3d_all.shape)
pts3d_all = pts3d_all.reshape(-1, 3)  # (B*H*W, 3)

mask_all = torch.cat([pred["mask"] for pred in predictions], dim=0)
mask_all = mask_all.reshape(-1)  # (B*H*W,)


rgb_all = torch.cat([view["img"] for view in processed_views], dim=0)
print(rgb_all.shape)

rgb_all = rgb_all.permute(0, 2, 3, 1).reshape(-1, 3)  # (B*H*W, 3)

generate_pcd(pts3d_all, rgb_all, mask_all, "output.pcd")
