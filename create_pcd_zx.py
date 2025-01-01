import os
import numpy as np
from PIL import Image
import open3d as o3d
from scipy.spatial.transform import Rotation as R


def create_pcd_from_raw_data(rgb_dir, depth_dir, height, width, xyz_path, rotation_path,save_dir):
    # Camera intrinsics (example)
    hfov = float(90) * np.pi / 180.
    focal_length = width / (2 * np.tan(hfov / 2))
    K = np.array([
                [focal_length, 0., width/2],
                [0., focal_length, height/2],
                [0., 0.,  1]], dtype=np.float32)

    # Get camera poses
    # some staticmethod functions
    def get_transformation(xyz, rotatangle):
        roll1,pitch1,yaw1 = -rotatangle[0],rotatangle[1], (rotatangle[2]-1.5708)
        r_1 = R.from_euler('yxz', [yaw1,pitch1,roll1])
        r_matrix_1 = r_1.as_matrix()
        xyz_1 = np.array([ -xyz[1], xyz[2],xyz[0]])
        transformation1 = np.eye(4)
        transformation1[:3,:3]=r_matrix_1
        transformation1[:3,3] = xyz_1.T
        # transformation1 = np.linalg.inv(transformation1)

        return transformation1
    
    xyzs = np.loadtxt(xyz_path)
    rotatangles = np.loadtxt(rotation_path)

    poses = []
    for i in range(len(xyzs)):
        transformation_i = get_transformation(xyzs[i], rotatangles[i])
        poses.append(transformation_i)
    poses = np.array(poses)

    print(f"Loaded {len(poses)} poses.")

    def depth_to_pointcloud(depth, K):
        """
        Convert a depth image into a point cloud in the camera frame.
        depth: HxW array of depth values (float)
        K: Camera intrinsic matrix 3x3
        Returns: Nx3 array of 3D points in camera coordinates
        """
        H, W = depth.shape
        x, y = np.meshgrid(np.arange(W), np.arange(H))
        x = x.astype(np.float32)
        y = y.astype(np.float32)

        # Flatten
        x = x.reshape(-1)
        y = y.reshape(-1)
        d = depth.reshape(-1)

        # Filter out invalid depths
        valid = d > 0
        x = x[valid]
        y = y[valid]
        d = d[valid]

        # Unproject to camera coordinates
        X = (x - K[0, 2]) / K[0, 0] * d
        Y = (y - K[1, 2]) / K[1, 1] * d
        Z = d

        points = np.stack([X, Y, Z], axis=1)
        return points
    
    all_points = []
    all_colors = []

    rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith('.png') or f.endswith('.jpg')])
    depth_files = sorted([f for f in os.listdir(depth_dir) if f.endswith('.npy')])
    assert len(rgb_files) == len(depth_files) == poses.shape[0], "Mismatch in number of RGB, depth frames, and poses."

    print(f"Found {len(rgb_files)} RGB and depth frames.")

    for i, (rgb_file, depth_file) in enumerate(zip(rgb_files, depth_files)):
        print(f"Processing frame {i + 1}/{len(rgb_files)}: RGB={rgb_file}, Depth={depth_file}")
        
        rgb_path = os.path.join(rgb_dir, rgb_file)
        rgb_image = Image.open(rgb_path).resize((width, height), Image.BILINEAR)
        rgb = np.array(rgb_image)

        if rgb.ndim == 3 and rgb.shape[2] == 4:
            rgb = rgb[:, :, :3]
        
        depth_path = os.path.join(depth_dir, depth_file)
        depth = np.load(depth_path).astype(np.float32)
        assert depth.shape == (height, width), f"Depth shape mismatch: expected ({height}, {width}), got {depth.shape}"
        
        cam_points = depth_to_pointcloud(depth, K)

        valid_indices = np.where(depth.reshape(-1) > 0)[0]
        y_idx, x_idx = np.divmod(valid_indices, width)
        point_colors = rgb[y_idx, x_idx] / 255.0
        
        cam_to_world = poses[i]
        
        ones = np.ones((cam_points.shape[0], 1), dtype=np.float32)
        cam_points_h = np.concatenate([cam_points, ones], axis=1)
        world_points_h = (cam_to_world @ cam_points_h.T).T
        world_points = world_points_h[:, :3] / world_points_h[:, 3:4]
        
        all_points.append(world_points)
        all_colors.append(point_colors)

    all_points = np.concatenate(all_points, axis=0)
    all_colors = np.concatenate(all_colors, axis=0)

    print(f"Total points aggregated: {all_points.shape[0]}")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    pcd.colors = o3d.utility.Vector3dVector(all_colors)

    # output_path = "./output_pointcloud.ply"
    output_path = os.path.join(save_dir, "output_pointcloud.ply")
    o3d.io.write_point_cloud(output_path, pcd)
    print(f"Point cloud saved to {output_path}")
    return all_points, all_colors, poses

# --------------------------------------------
# User-defined inputs:
# --------------------------------------------

# Path to your data
rgb_dir = "./example_zhixuan/mp3d_gendata1/"
depth_dir = "./example_zhixuan/mp3d_gendata1"

# Camera intrinsics (example)
height = 512
width = 512
hfov = float(90) * np.pi / 180.
focal_length = width / (2 * np.tan(hfov / 2))
K = np.array([
            [focal_length, 0., width/2],
            [0., focal_length, height/2],
            [0., 0.,  1]], dtype=np.float32)

# Get camera poses
# some staticmethod functions
def get_transformation(xyz, rotatangle):
    roll1,pitch1,yaw1 = -rotatangle[0],rotatangle[1], (rotatangle[2]-1.5708)
    r_1 = R.from_euler('yxz', [yaw1,pitch1,roll1])
    r_matrix_1 = r_1.as_matrix()
    xyz_1 = np.array([ -xyz[1], xyz[2],xyz[0]])
    transformation1 = np.eye(4)
    transformation1[:3,:3]=r_matrix_1
    transformation1[:3,3] = xyz_1.T
    # transformation1 = np.linalg.inv(transformation1)

    return transformation1

xyz_path = "./example_zhixuan/mp3d_data1/position.txt"
rotation_path = "./example_zhixuan/mp3d_data1/rotation.txt"
xyzs = np.loadtxt(xyz_path)
rotatangles = np.loadtxt(rotation_path)

poses = []
for i in range(len(xyzs)):
    transformation_i = get_transformation(xyzs[i], rotatangles[i])
    poses.append(transformation_i)
poses = np.array(poses)

print(f"Loaded {len(poses)} poses.")

# --------------------------------------------
# Function to unproject depth to 3D points
# --------------------------------------------
def depth_to_pointcloud(depth, K):
    """
    Convert a depth image into a point cloud in the camera frame.
    depth: HxW array of depth values (float)
    K: Camera intrinsic matrix 3x3
    Returns: Nx3 array of 3D points in camera coordinates
    """
    H, W = depth.shape
    x, y = np.meshgrid(np.arange(W), np.arange(H))
    x = x.astype(np.float32)
    y = y.astype(np.float32)

    # Flatten
    x = x.reshape(-1)
    y = y.reshape(-1)
    d = depth.reshape(-1)

    # Filter out invalid depths
    valid = d > 0
    x = x[valid]
    y = y[valid]
    d = d[valid]

    # Unproject to camera coordinates
    X = (x - K[0, 2]) / K[0, 0] * d
    Y = (y - K[1, 2]) / K[1, 1] * d
    Z = d

    points = np.stack([X, Y, Z], axis=1)  # Nx3
    return points

# # --------------------------------------------
# # Aggregate point cloud from all frames
# # --------------------------------------------
# all_points = []
# all_colors = []

# # Get sorted image file lists
# # --------------------------------------------
# # Modification: Changed depth file extension to .npy
# # --------------------------------------------
# rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith('.png') or f.endswith('.jpg')])
# depth_files = sorted([f for f in os.listdir(depth_dir) if f.endswith('.npy')])

# # Check that we have the same number of RGB and depth frames and matches the pose count
# assert len(rgb_files) == len(depth_files) == poses.shape[0], "Mismatch in number of RGB, depth frames, and poses."

# print(f"Found {len(rgb_files)} RGB and depth frames.")

# for i, (rgb_file, depth_file) in enumerate(zip(rgb_files, depth_files)):
#     print(f"Processing frame {i + 1}/{len(rgb_files)}: RGB={rgb_file}, Depth={depth_file}")
    
#     # Load RGB image
#     rgb_path = os.path.join(rgb_dir, rgb_file)
#     # rgb = np.array(Image.open(rgb_path))  # HxWx3, uint8
#     rgb_image = Image.open(rgb_path).resize((width, height), Image.BILINEAR)  # Modification 2: Resize
#     rgb = np.array(rgb_image)  # HxWx3, uint8

#     if rgb.ndim == 3 and rgb.shape[2] == 4:
#         rgb = rgb[:, :, :3]
    
#     # Load Depth from .npy file
#     depth_path = os.path.join(depth_dir, depth_file)
#     depth = np.load(depth_path).astype(np.float32)  # Shape: (512, 512)
    
#     # Ensure depth shape matches expected dimensions
#     assert depth.shape == (height, width), f"Depth shape mismatch: expected ({height}, {width}), got {depth.shape}"
    
#     # Convert depth to camera space points
#     cam_points = depth_to_pointcloud(depth, K)  # Nx3
    
#     # Get corresponding colors
#     # --------------------------------------------
#     # Modification: Changed how valid indices are computed to match .npy depth loading
#     # --------------------------------------------
#     valid_indices = np.where(depth.reshape(-1) > 0)[0]
#     y_idx, x_idx = np.divmod(valid_indices, width)
#     point_colors = rgb[y_idx, x_idx] / 255.0  # Nx3, normalized
    
#     # Load camera pose (camera-to-world)
#     cam_to_world = poses[i]  # 4x4 matrix
    
#     # Convert points to homogeneous coordinates
#     ones = np.ones((cam_points.shape[0], 1), dtype=np.float32)
#     cam_points_h = np.concatenate([cam_points, ones], axis=1)  # Nx4
    
#     # Transform to world coordinates
#     world_points_h = (cam_to_world @ cam_points_h.T).T  # Nx4
#     world_points = world_points_h[:, :3] / world_points_h[:, 3:4]  # Nx3
    
#     # Append to all points and colors
#     all_points.append(world_points)
#     all_colors.append(point_colors)

# # Concatenate all points and colors
# all_points = np.concatenate(all_points, axis=0)
# all_colors = np.concatenate(all_colors, axis=0)

# print(f"Total points aggregated: {all_points.shape[0]}")

# # --------------------------------------------
# # Create Open3D point cloud and save
# # --------------------------------------------
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(all_points)
# pcd.colors = o3d.utility.Vector3dVector(all_colors)

# # Save the point cloud to a file
# output_path = "./example_zhixuan/mp3d_gendata1/output_pointcloud.ply"
# o3d.io.write_point_cloud(output_path, pcd)
# print(f"Point cloud saved to {output_path}")