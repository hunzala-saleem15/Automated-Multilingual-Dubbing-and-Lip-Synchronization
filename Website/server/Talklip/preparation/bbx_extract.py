import argparse
import math
import os
from os import path
import sys
import cv2
import numpy as np
from tqdm import tqdm

sys.path.append(os.getcwd().replace('preparation', ''))
import face_detection


def smooth_bbxs(bbxs, window=5):
    """Simple temporal smoothing to reduce jitter / ghosting."""
    if len(bbxs) == 0:
        return bbxs

    bbxs = np.array(bbxs, dtype=np.float32)
    smoothed = bbxs.copy()

    half = window // 2
    for i in range(len(bbxs)):
        start = max(0, i - half)
        end = min(len(bbxs), i + half + 1)
        smoothed[i] = np.mean(bbxs[start:end], axis=0)

    return smoothed


def process_video_file(samplename, args, fa):
    vfile = '{}/{}.mp4'.format(args.video_root, samplename)
    video_stream = cv2.VideoCapture(vfile)

    frames = []
    while True:
        still_reading, frame = video_stream.read()
        if not still_reading:
            video_stream.release()
            break
        frames.append(frame)

    if len(frames) == 0:
        print(f"[WARNING] Empty video: {vfile}")
        return

    height, width, _ = frames[0].shape

    # Ensure output directory exists
    os.makedirs(args.bbx_root, exist_ok=True)
    out_path = path.join(args.bbx_root, samplename + '.npy')

    batches = [
        frames[i : i + args.batch_size]
        for i in range(0, len(frames), args.batch_size)
    ]

    bbxs = []
    last_valid_bbx = None

    # Default fallback box (center 96x96) – only used if NO face is ever detected
    htmp = int((height - 96) / 2)
    wtmp = int((width - 96) / 2)
    default_bbx = [wtmp, htmp, wtmp + 96, htmp + 96]

    for fb in batches:
        preds = fa.get_detections_for_batch(np.asarray(fb))

        for j, f in enumerate(preds):
            if f is None:
                # Prefer last valid box instead of jumping to center (reduces ghosting)
                if last_valid_bbx is not None:
                    x1, y1, x2, y2 = last_valid_bbx
                else:
                    x1, y1, x2, y2 = default_bbx
            else:
                x1, y1, x2, y2 = f
                last_valid_bbx = [x1, y1, x2, y2]

            # Clamp to image boundaries
            x1 = max(0, min(int(x1), width - 1))
            y1 = max(0, min(int(y1), height - 1))
            x2 = max(x1 + 1, min(int(x2), width))
            y2 = max(y1 + 1, min(int(y2), height))

            # Enforce minimum size (helps croppatch stability)
            if (x2 - x1) < 20 or (y2 - y1) < 20:
                if last_valid_bbx is not None:
                    x1, y1, x2, y2 = last_valid_bbx
                else:
                    x1, y1, x2, y2 = default_bbx

            bbxs.append([x1, y1, x2, y2])

    bbxs = np.array(bbxs, dtype=np.float32)

    # Temporal smoothing (strongly recommended against ghosting)
    bbxs = smooth_bbxs(bbxs, window=5)

    np.save(out_path, bbxs)
    print(f"[OK] Saved BBX: {out_path}  ({len(bbxs)} frames)")


def main(args, fa):
    print(
        'Started processing of {}-th rank for {} on CPU'.format(
            args.rank, args.video_root
        )
    )

    with open(args.filelist) as f:
        lines = f.readlines()

    filelist = [line.strip().split()[0] for line in lines]

    nlength = math.ceil(len(filelist) / args.nshard)
    start_id, end_id = nlength * args.rank, nlength * (args.rank + 1)
    filelist = filelist[start_id:end_id]
    print('process {}-{}'.format(start_id, end_id))

    for vfile in tqdm(filelist):
        process_video_file(vfile, args, fa)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # CPU par OOM error se bachne ke liye default=8 set kiya hai,
    # High-RAM par aap CLI se '--batch_size 32' bhej sakte hain.
    parser.add_argument(
        '--batch_size',
        help='Face detection batch size',
        default=8,
        type=int,
    )
    parser.add_argument(
        '--filelist',
        help="Path of a file list containing all samples' name",
        required=True,
        type=str,
    )
    parser.add_argument(
        '--video_root', help='Root folder of video', required=True, type=str
    )
    parser.add_argument(
        '--bbx_root',
        help='Root folder of bounding boxes of faces',
        required=True,
        type=str,
    )
    parser.add_argument(
        '--rank',
        help='the rank of the current thread in the preprocessing',
        default=1,
        type=int,
    )
    parser.add_argument(
        '--nshard',
        help='How many threads are used in the preprocessing',
        default=1,
        type=int,
    )
    parser.add_argument(
        '--gpu',
        help='the rank of the current thread in the preprocessing',
        default=1,
        type=int,
    )

    args = parser.parse_args()

    s3fd_weights = (
        r'D:\Website\server\Talklip\face_detection\detection\sfd\s3fd.pth'
    )

    if not path.isfile(s3fd_weights):
        raise FileNotFoundError(
            f'Save the s3fd model to {s3fd_weights} before running this script!'
        )

    args.rank -= 1

    fa = face_detection.FaceAlignment(
        face_detection.LandmarksType._2D, flip_input=False, device='cpu'
    )

    main(args, fa)