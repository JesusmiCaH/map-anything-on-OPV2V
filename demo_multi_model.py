# Optional config for better memory efficiency
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Required imports
import argparse
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


def generate_pcd(pts3d, rgb, mask, filename, gt_pcds=None):
    pts3d = pts3d.cpu().numpy()
    rgb = (rgb * 255).cpu().numpy().astype(np.uint8)
    mask = mask.cpu().numpy().astype(bool)

    pts3d = pts3d[mask]
    rgb = rgb[mask]

    points_count = len(pts3d) if gt_pcds is None else len(pts3d) + len(gt_pcds[0])
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
""".format(points_count, points_count)
        )
        # Data
        for pt, color in zip(pts3d, rgb):
            rgb_float = pack_rgb(255, 0, 0)
            f.write(f"{pt[0]} {pt[1]} {pt[2]} {rgb_float}\n")
            # f.write(f"{pt[0]} {pt[1]} {pt[2]}\n")

        if gt_pcds is not None:
            gt_pcd = gt_pcds[0]  # Get the first ground truth point cloud
            for pt in gt_pcd:
                rgb_float = pack_rgb(0, 255, 0)
                f.write(f"{pt[0]} {pt[1]} {pt[2]} {rgb_float}\n")


def main(args):
    # Get inference device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Init model - This requries internet access or the huggingface hub cache to be pre-downloaded
    # For Apache 2.0 license model, use "facebook/map-anything-apache"
    model = MapAnything.from_pretrained("facebook/map-anything").to(device)

    imgs, intrinsics, extrinsics, gt_pcds, T_gt2pred = opv2v_data_prepare(
        f"{args.datapath}/{args.current_time}/config.yaml",
        cam_dirs=[0, 1, 2, 3],
        has_contributer=True,
    )

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
    # print("🚕🚕🚕🚕", processed_views[0]["img"].shape)

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

    pts3d_all = pts3d_all.reshape(-1, 3)  # (B*H*W, 3)
    T_pred2gt = torch.tensor(np.linalg.inv(T_gt2pred)).float()  # (4, 4)
    pts3d_all_homogeneous = torch.cat(
        [pts3d_all, torch.ones((pts3d_all.shape[0], 1), device=pts3d_all.device)], dim=1
    )  # (B*H*W, 4)

    pts3d_all = (T_pred2gt.to(pts3d_all.device) @ pts3d_all_homogeneous.T).T[
        :, :3
    ]  # (B*H*W, 3)

    mask_all = torch.cat([pred["mask"] for pred in predictions], dim=0)
    mask_all = mask_all.reshape(-1)  # (B*H*W,)

    rgb_all = torch.cat([view["img"] for view in processed_views], dim=0)

    rgb_all = rgb_all.permute(0, 2, 3, 1).reshape(-1, 3)  # (B*H*W, 3)

    # Ensure base folder exists
    os.makedirs("/data/test", exist_ok=True)
    config = yaml.safe_load(
        open(f"{args.datapath}/{args.current_time}/config.yaml", "r")
    )

    # Build output directory and file path
    output_dir = os.path.join(
        "/data/test",
        args.current_time,
        str(config["ego_id"]),
    )
    os.makedirs(output_dir, exist_ok=True)

    idx = config["frame_idx"][0]
    output_path = os.path.join(output_dir, f"{idx:06d}.pcd")

    # Save PCD to the requested location
    generate_pcd(pts3d_all, rgb_all, mask_all, output_path)

    # Copy f"{idx:06d}.yaml" from source to output directory
    import shutil

    src_yaml_path = os.path.join(
        args.datapath,
        args.current_time,
        str(config["ego_id"]),
        f"{idx:06d}.yaml",
    )
    dst_yaml_path = os.path.join(output_dir, f"{idx:06d}.yaml")
    shutil.copyfile(src_yaml_path, dst_yaml_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datapath",
        type=str,
        default="/data/train/",
        help="Path to the OPV2V data package",
    )
    parser.add_argument(
        "--current_time",
        type=str,
        default="2021_08_18_09_02_56",
        help="Current time folder name, e.g. 2021_08_18_09_02_56",
    )

    args = parser.parse_args()
    main(args)
