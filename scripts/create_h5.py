import sys
import os
import argparse
import h5py
import cv2
import numpy as np
from tqdm import tqdm
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from utils.video_utils import extract_pose

def create_h5(raw_dir: str, out_h5: str, skip: int = 1):
    video_list = []
    for root, _, files in os.walk(raw_dir):
        for fname in files:
            if not fname.lower().endswith('.mp4'):
                continue
            path = os.path.join(root, fname)
            folder = os.path.basename(root).lower()
            if folder == 'violence':
                label = 1
            elif folder == 'nonviolence':
                label = 0
            else:
                label = 0
            video_list.append((path, fname, label))

    if not video_list:
        print(f"No .mp4 files found under {raw_dir}")
        return

    with h5py.File(out_h5, 'w') as h5f:
        try:
            for video_path, fname, label in tqdm(video_list, desc="Processing videos"):
                cap = cv2.VideoCapture(video_path)
                frames = []
                idx = 0

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if idx % skip == 0:
                        frames.append(frame)
                    idx += 1
                cap.release()

                if not frames:
                    tqdm.write(f"[WARN] no frames read from {video_path}")
                    continue

                poses = np.stack([extract_pose(f) for f in frames], axis=0)  
                grp_name = os.path.splitext(fname)[0]
                grp = h5f.create_group(grp_name)
                grp.create_dataset('pose', data=poses, compression='gzip')
                grp.attrs['label'] = label

            print(f"\nDone: processed {len(video_list)} videos → saved to {out_h5}")
        except KeyboardInterrupt:
            print("\nInterrupted by user. Exiting cleanly.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Create HDF5 of pose sequences from raw videos"
    )
    parser.add_argument(
        '--raw_dir', required=True,
        help="Root directory containing raw .mp4 videos (will recurse subfolders)"
    )
    parser.add_argument(
        '--out_h5', required=True,
        help="Output path for the compressed HDF5 file"
    )
    parser.add_argument(
        '--skip', type=int, default=1,
        help="Frame-skip interval: 1=every frame, 15=every 15th frame, etc."
    )
    args = parser.parse_args()
    create_h5(args.raw_dir, args.out_h5, args.skip)
