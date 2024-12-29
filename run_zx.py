import os
import argparse
from PIL import Image

from luciddreamer import LucidDreamer
from create_pcd_zx import create_pcd_from_raw_data

if __name__ == "__main__":
    ### option
    parser = argparse.ArgumentParser(description='Arguments for LucidDreamer')
    parser.add_argument('--save_dir', '-s', type=str, default='./example_zhixuan/mp3d_gendata1/', help='Save directory')
    parser.add_argument('--rgb_dir', type=str, default='./example_zhixuan/mp3d_gendata1/', help='RGB image directory')
    parser.add_argument('--depth_dir', type=str, default='./example_zhixuan/mp3d_gendata1/', help='Depth image directory')
    parser.add_argument('--xyz_path', type=str, default='./example_zhixuan/mp3d_gendata1/position.txt', help='Path to xyz coordinates')
    parser.add_argument('--rotation_path', type=str, default='./example_zhixuan/mp3d_gendata1/rotation.txt', help='Path to rotation coordinates')
    args = parser.parse_args()

    height = 512
    width = 512

    all_points, all_colors, poses = create_pcd_from_raw_data(args.rgb_dir, args.depth_dir, height, width, args.xyz_path, args.rotation_path)
    ld = LucidDreamer(for_gradio=False, save_dir=args.save_dir)
    ld.create_zx(all_points, all_colors, args.rgb_dir, poses, width, height)
    
