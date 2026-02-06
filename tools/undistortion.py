import sys
import json
import argparse
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
from hand_tracking_toolkit import camera
from multiprocessing import Pool

g_map_x = None
g_map_y = None
g_rgb_dir = None
g_mask_dir = None
g_rgb_out = None
g_mask_out = None
g_scene_gt = None
g_scene_gt_undist = None

cv2.setNumThreads(0)
num_workers = 32 # Customize your settings.

def build_undistortion_map(src_camera, dst_camera):
    W, H = dst_camera.width, dst_camera.height
    px, py = np.meshgrid(np.arange(W), np.arange(H))
    dst_win_pts = np.column_stack((px.flatten(), py.flatten()))
    
    dst_eye_pts = dst_camera.window_to_eye(dst_win_pts)
    world_pts = dst_camera.eye_to_world(dst_eye_pts)
    src_eye_pts = src_camera.world_to_eye(world_pts)
    src_win_pts = src_camera.eye_to_window(src_eye_pts)

    mask = src_eye_pts[:, 2] < 0
    src_win_pts[mask] = -1
    src_win_pts = src_win_pts.astype(np.float32)

    map_x = src_win_pts[:, 0].reshape((H, W))
    map_y = src_win_pts[:, 1].reshape((H, W))
    return map_x, map_y

def init_worker(map_x, map_y, rgb_dir, mask_dir, rgb_out, mask_out, scene_gt, scene_gt_undist):
    global g_map_x, g_map_y, g_rgb_dir, g_mask_dir, g_rgb_out, g_mask_out, g_scene_gt, g_scene_gt_undist
    g_map_x = map_x
    g_map_y = map_y
    g_rgb_dir = rgb_dir
    g_mask_dir = mask_dir
    g_rgb_out = rgb_out
    g_mask_out = mask_out
    g_scene_gt = scene_gt
    g_scene_gt_undist = scene_gt_undist

def process_single_frame(frame_id):
    try:
        str_id = str(frame_id)
        
        raw_path = g_rgb_dir / f"{frame_id:06d}.png"
        if not raw_path.exists():
            raise FileNotFoundError(f"[Error] Missing RGB file: {raw_path}")

        raw_image = cv2.imread(str(raw_path))
        if raw_image is None:
            raise ValueError(f"[Error] Failed to read image: {raw_path}")

        undist_image = cv2.remap(raw_image, g_map_x, g_map_y, interpolation=cv2.INTER_LINEAR)
        
        encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), 1]
        cv2.imwrite(str(g_rgb_out / f"{frame_id:06d}.png"), undist_image, encode_param)

        if str_id not in g_scene_gt:
            return None

        current_objects = g_scene_gt[str_id]
        expected_count = len(g_scene_gt_undist.get(str_id, []))
        valid_idx = 0

        for original_idx, _ in enumerate(current_objects):
            mask_path = g_mask_dir / f"{frame_id:06d}_{original_idx:06d}.png"
            if not mask_path.exists():
                raise FileNotFoundError(f"[Error] Missing Mask file: {mask_path}")

            raw_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if raw_mask is None:
                 raise ValueError(f"[Error] Failed to read mask: {mask_path}")

            undist_mask = cv2.remap(raw_mask, g_map_x, g_map_y, interpolation=cv2.INTER_NEAREST)
            
            if cv2.countNonZero(undist_mask) < 10:
                continue
                
            new_name = f"{frame_id:06d}_{valid_idx:06d}.png"
            cv2.imwrite(str(g_mask_out / new_name), undist_mask, encode_param)
            valid_idx += 1
            
        if valid_idx != expected_count:
            return f"[Warning] Frame {frame_id}: Mismatch! Generated {valid_idx} vs Expect {expected_count}."
        
        return None

    except Exception as e:
        return f"[Exception] Frame {frame_id}: {str(e)}"

def process_scene_parallel(target_dir, stream_key='214-1'):
    target_dir = Path(target_dir)
    rgb_dir = target_dir / "rgb"
    mask_dir = target_dir / "mask"
    rgb_out = target_dir / "rgb_undist"
    mask_out = target_dir / "mask_undist"
    
    paths = {
        "camera": target_dir / "scene_camera.json",
        "gt": target_dir / "scene_gt.json",
        "gt_undist": target_dir / "scene_gt_undist.json"
    }
    
    missing_files = [p.name for p in paths.values() if not p.exists()]
    if missing_files:
        print(f"\n[Critical Error] Missing JSON files in {target_dir.name}: {missing_files}")
        sys.exit(1)

    rgb_out.mkdir(exist_ok=True)
    mask_out.mkdir(exist_ok=True)

    with open(paths["camera"], "r") as f: scene_camera = json.load(f)
    with open(paths["gt"], "r") as f: scene_gt = json.load(f)
    with open(paths["gt_undist"], "r") as f: scene_gt_undist = json.load(f)

    print(f"[{target_dir.name}] Building Map...", end=" ")
    
    first_id = list(scene_camera.keys())[0]
    calib = scene_camera[first_id]["cam_model"]
    
    camera_raw = {
        stream_key: {
            "T_world_from_camera": {
                "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                "translation_xyz": [0.0, 0.0, 0.0]
            },
            "calibration": calib
        }
    }
    static_src = camera.from_json(camera_raw[stream_key])
    static_dst = camera.PinholePlaneCameraModel(
        width=static_src.width,
        height=static_src.height,
        f=[static_src.f[0], static_src.f[1]],
        c=static_src.c,
        distort_coeffs=[],
        T_world_from_eye=static_src.T_world_from_eye,
    )
    map_x, map_y = build_undistortion_map(static_src, static_dst)
    print("Done. Processing...")

    frame_ids = sorted([int(k) for k in scene_camera.keys()])
    
    with Pool(processes=num_workers, initializer=init_worker, 
              initargs=(map_x, map_y, rgb_dir, mask_dir, rgb_out, mask_out, scene_gt, scene_gt_undist)) as pool:
        
        for result in tqdm(pool.imap_unordered(process_single_frame, frame_ids, chunksize=10), 
                           total=len(frame_ids), desc=f"Scene {target_dir.name}"):
            if result:
                if "[Exception]" in result or "[Error]" in result:
                    print(f"\n{result}")
                    print("Aborting due to critical error.")
                    pool.terminate()
                    sys.exit(1)
                else:
                    tqdm.write(result)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate undistorted images/masks for EgoXtreme dataset.")
    parser.add_argument('--data_dir', type=str, required=True, help='Root directory (e.g., ./data/train)')
    parser.add_argument('--scene_id', type=str, default=None, help='Specific scene ID to process')
    parser.add_argument('--all', action='store_true', help='Process all scenes')
    
    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    if not data_dir.exists():
        print(f"[Error] Directory not found: {data_dir}")
        sys.exit(1)

    if args.all:
        target_list = sorted([p for p in data_dir.iterdir() if p.is_dir()])
    elif args.scene_id:
        scene_path = data_dir / args.scene_id
        if scene_path.exists():
            target_list = [scene_path]
        else:
            print(f"[Error] Scene {args.scene_id} not found in {data_dir}")
            sys.exit(1)
    else:
        print("Usage: python undistortion.py --data_dir [PATH] (--all | --scene_id [ID])")
        sys.exit(1)

    for scene_path in target_list:
        process_scene_parallel(scene_path)
    
    print("\nAll tasks completed successfully.")
