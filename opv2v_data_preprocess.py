import os

import numpy as np
import open3d as o3d
import yaml
from PIL import Image


def euler_matrix(roll, yaw, pitch, degrees=True):
    """
    Build a rotation matrix from Euler angles in Z (yaw) -> Y (pitch) -> X (roll) order.
    Returns a 3x3 matrix that rotates camera coordinates into world coordinates.
    """
    if degrees:
        roll = np.deg2rad(roll)
        yaw = np.deg2rad(yaw)
        pitch = np.deg2rad(pitch)

    Rx = np.array(
        [[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]],
        dtype=float,
    )

    Ry = np.array(
        [
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)],
        ],
        dtype=float,
    )

    Rz = np.array(
        [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]],
        dtype=float,
    )

    # Apply yaw, then pitch, then roll
    R = Rz @ Ry @ Rx
    return R


def carla_pose_to_cam2world(cords, degrees=True, to_opencv_axes=False):
    """
    Convert CARLA cords to a 4x4 cam2world matrix.

    cords: [x, y, z, roll, yaw, pitch] in CARLA world coordinates.
    degrees: if True, input angles are in degrees, otherwise radians.
    to_opencv_axes: if True, convert the CARLA camera axis convention
                    (x forward, y right, z down)
                    to OpenCV convention (x right, y down, z forward).

    Returns:
        T (4x4): Homogeneous transformation matrix (cam->world).
    """
    x, y, z, roll, yaw, pitch = cords

    # Axis conversion: CARLA_cam (x forward, y right, z down) -> OpenCV_cam (x right, y down, z forward)
    # Mapping:
    #   x_cv =  y_carla
    #   y_cv =  z_carla
    #   z_cv =  x_carla

    # Build cam->world rotation with CARLA convention
    R_carla = euler_matrix(roll, pitch, yaw, degrees=degrees)

    if to_opencv_axes:
        # Transform OpenCV camera to world
        # R_world_cv = R_world_carla * (cv->carla)
        R = R_carla
    else:
        R = R_carla

    t = np.array([y, -z, x], dtype=float).reshape(3, 1)
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = t.flatten()

    return T


def opv2v_data_prepare(config_file, cam_dirs=[0, 3], has_contributer=True):
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
        # get the folder of config_file as the base folder
        base_folder = os.path.dirname(config_file)

    imgs = []
    intrinsics = []
    extrinsics = []

    car_list = ["ego", "contributer"] if has_contributer else ["ego"]

    pcds = []
    for i, idx in enumerate(config["frame_idx"]):
        for car in car_list:
            for dir in cam_dirs:
                img_path = os.path.join(
                    base_folder,
                    str(config[f"{car}_id"]),
                    f"{idx:06d}_camera{dir}.png",
                )
                with Image.open(img_path) as im:
                    im = im.convert("RGB")
                    arr = np.array(im, dtype=np.uint8)  # (H,W,3)
                imgs.append(arr)
                # print("🌈", car, idx, dir)
                with open(
                    os.path.join(
                        base_folder,
                        str(config[f"{car}_id"]),
                        f"{idx:06d}.yaml",
                    ),
                    "r",
                ) as f:
                    cam_params = yaml.safe_load(f)
                    intrinsics.append(cam_params[f"camera{dir}"]["intrinsic"])
                    extrinsics.append(
                        carla_pose_to_cam2world(cam_params[f"camera{dir}"]["cords"])
                    )

        if i == 0:
            yaml_filename = os.path.join(
                base_folder,
                str(config["ego_id"]),
                f"{config['frame_idx'][i]:06d}.yaml",
            )
            with open(yaml_filename, "r") as f:
                cam_params = yaml.safe_load(f)
                ext_cam2lidar = cam_params[f"camera{0}"]["extrinsic"]

        pcd_filename = os.path.join(
            base_folder,
            str(config["ego_id"]),
            f"{idx:06d}.pcd",
        )

        pc = o3d.io.read_point_cloud(pcd_filename)
        pcd_points = np.asarray(pc.points, dtype=np.float32)

        R_for_gt = np.array(
            [
                [0, 1, 0],
                [0, 0, -1],
                [1, 0, 0],
            ],
            dtype=float,
        )

        T_for_gt = np.eye(4, dtype=float)
        T_for_gt[:3, :3] = R_for_gt
        T_gt2pred = T_for_gt @ ext_cam2lidar

        if pcd_points.size == 0:
            raise ValueError(f"Empty point cloud: {pcd_filename}")

        # optional: collect per-frame point clouds
        pcds.append(pcd_points)

    return imgs, intrinsics, extrinsics, pcds, T_gt2pred.astype(np.float32)


# # Example usage
# example_cords = [
#     141.35067749023438,
#     -388.642578125,
#     1.0410505533218384,
#     0.07589337974786758,
#     174.18048095703125,
#     0.20690691471099854,
# ]

# T_cam2world_carla_axes = carla_pose_to_cam2world(
#     example_cords, degrees=True, to_opencv_axes=False
# )
# T_cam2world_opencv_axes = carla_pose_to_cam2world(
#     example_cords, degrees=True, to_opencv_axes=True
# )
