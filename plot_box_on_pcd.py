import os
import struct

import numpy as np
import open3d as o3d
import torch


def pack_rgb(r, g, b):
    """Pack r, g, b into a single float32"""
    rgb_int = (int(r) << 16) | (int(g) << 8) | int(b)
    return struct.unpack("f", struct.pack("I", rgb_int))[0]


def colorize_pcd(pcd: np.ndarray, color: list) -> np.ndarray:
    """Colorize point cloud with a given RGB color.

    Args:
        pcd (np.ndarray): Point cloud of shape (N, 3).
        color (list): RGB color as a list of three floats in [0, 1].

    Returns:
        np.ndarray: Colored point cloud of shape (N, 6) where the last three
                    columns are the RGB colors.
    """
    num_points = pcd.shape[0]
    color_array = np.tile(pack_rgb(*color), (num_points, 1)).astype(np.float32)

    colored_pcd = np.hstack((pcd, color_array))
    print(colored_pcd.dtype, color_array.dtype)

    return colored_pcd


def visualize_pcd(filename, *pcds: np.ndarray):
    points_count = sum(len(pcd) for pcd in pcds)

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
        for pcd in pcds:
            for pt in pcd:
                if len(pt) == 3:
                    f.write(f"{pt[0]} {pt[1]} {pt[2]}\n")
                elif len(pt) == 4:
                    f.write(f"{pt[0]} {pt[1]} {pt[2]} {pt[3]}\n")


def main():
    testset_path = "/data/test"
    # list all folders in testset_path
    folder_list = os.listdir(testset_path)
    folder_list = sorted(os.listdir(testset_path))

    print(folder_list)
    box_path = "../OpenCOOD/checkpoints/pointpillar_late_fusion/boxes"

    idx = 0
    for folder in folder_list:
        print(f"Processing folder: {folder}")

        folder_full_path = os.path.join(testset_path, folder)

        # Find pcd file in folder_full_path
        pcd_files = []
        for sub in os.listdir(folder_full_path):
            sub_path = os.path.join(folder_full_path, sub)
            if os.path.isdir(sub_path):
                for f in os.listdir(sub_path):
                    if f.endswith(".pcd"):
                        pcd_files.append(os.path.join(sub, f))

        print(pcd_files, folder_full_path)
        for pcd_file in pcd_files:
            pcd_full_path = os.path.join(folder_full_path, pcd_file)
            print("📂", pcd_full_path)

            # Load pcd points
            pc = o3d.io.read_point_cloud(pcd_full_path)
            pcd_points = np.asarray(pc.points, dtype=np.float32)
            pcd_colored = colorize_pcd(pcd_points, [255, 255, 255])  # White for GT

            box_file = os.path.join(box_path, f"{idx:05d}.pt")
            boxes_data = torch.load(box_file)
            pred_box_tensor = boxes_data["pred_box_tensor"].numpy()
            box_points = pred_box_tensor.reshape(-1, 3)
            box_colored = colorize_pcd(box_points, [255, 0, 0])  # Red for Predicted
            gt_box_tensor = boxes_data["gt_box_tensor"].numpy()
            gt_box_points = gt_box_tensor.reshape(-1, 3)
            gt_box_colored = colorize_pcd(
                gt_box_points, [0, 255, 0]
            )  # Blue for GT Boxes
            print(f"{folder_full_path}/pcd_with_boxes_{idx:05d}.pcd")

            visualize_pcd(
                f"{folder_full_path}/pcd_with_boxes_{idx:05d}.pcd",
                pcd_colored,
                box_colored,
                gt_box_colored,
            )

            idx += 1


if __name__ == "__main__":
    main()
