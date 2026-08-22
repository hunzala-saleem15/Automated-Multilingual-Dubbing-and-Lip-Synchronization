from os import path
import numpy as np
import argparse, os, cv2
from tqdm import tqdm
import math
import time

import sys
sys.path.append(os.getcwd().replace('preparation', ''))
import face_detection


def process_video_file(samplename, args, fa):
    vfile = '{}/{}.mp4'.format(args.video_root, samplename)
    print("Opening:", vfile)
    
    video_stream = cv2.VideoCapture(vfile)

    frames = []
    while True:
        still_reading, frame = video_stream.read()
        if not still_reading:
            video_stream.release()
            break
        frames.append(frame)

    if not frames:
        print(f"Warning: No frames read from {vfile}. Skipping.")
        return

    height, width, _ = frames[0].shape
    print("Frames:", len(frames))

    fulldir = path.join(args.bbx_root, samplename)
    os.makedirs(os.path.dirname(fulldir), exist_ok=True)

    batches = [frames[i:i + args.batch_size] for i in range(0, len(frames), args.batch_size)]

    bbxs = list()
    print("Starting Detection...")

    for i, fb in enumerate(batches):
        print(f"Processing Batch {i+1}/{len(batches)}")
        start_time = time.time()

        preds = fa.get_detections_for_batch(np.asarray(fb))

        elapsed = time.time() - start_time
        print(f"Batch {i+1} Done (Took {elapsed:.2f}s)")

        for j, f in enumerate(preds):
            if f is None:
                htmp = int((height - 96) / 2)
                wtmp = int((width - 96) / 2)
                x1, y1, x2, y2 = wtmp, htmp, wtmp + 96, htmp + 96
            else:
                x1, y1, x2, y2 = f

                pad_x = int((x2 - x1) * 0.35)
                pad_y = int((y2 - y1) * 0.35)

                x1 = max(0, int(x1 - pad_x))
                y1 = max(0, int(y1 - pad_y))
                x2 = min(width, int(x2 + pad_x))
                y2 = min(height, int(y2 + pad_y))

            bbxs.append([x1, y1, x2, y2])

    bbxs = np.array(bbxs)
    np.save(fulldir + '.npy', bbxs)


def main(args, fa):
    print('Started processing of {}-th rank for {} on {} GPUs'.format(args.rank, args.video_root, args.gpu))

    with open(args.filelist) as f:
        lines = f.readlines()

    filelist = [line.strip().split()[0] for line in lines]

    nlength = math.ceil(len(filelist) / args.nshard)
    start_id, end_id = nlength * args.rank, nlength * (args.rank + 1)
    filelist = filelist[start_id: end_id]
    print('process {}-{}'.format(start_id, end_id))

    for vfile in tqdm(filelist):
        process_video_file(vfile, args, fa)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Batch size update: default=4 -> default=32
    parser.add_argument('--batch_size', help='Single GPU Face detection batch size', default=1, type=int)
    parser.add_argument('--filelist', help="Path of a file list containing all samples' name", required=True, type=str)
    parser.add_argument("--video_root", help="Root folder of video", required=True, type=str)
    parser.add_argument('--bbx_root', help="Root folder of bounding boxes of faces", required=True, type=str)
    parser.add_argument("--rank", help="the rank of the current thread in the preprocessing ", default=1, type=int)
    parser.add_argument("--nshard", help="How many threads are used in the preprocessing ", default=1, type=int)
    parser.add_argument("--gpu", help="the rank of the current thread in the preprocessing ", default=1, type=int)

    args = parser.parse_args()

    s3fd_weights = r'D:\Website\server\Talklip\face_detection\detection\sfd\s3fd.pth'
    if not path.isfile(s3fd_weights):
        raise FileNotFoundError(f'Save the s3fd model to {s3fd_weights} before running this script!')

    args.rank -= 1

    print("Loading FaceAlignment...")
    fa = face_detection.FaceAlignment(
        face_detection.LandmarksType._2D,
        flip_input=False,
        device='cpu'
    )
    print("FaceAlignment Loaded")

    main(args, fa)