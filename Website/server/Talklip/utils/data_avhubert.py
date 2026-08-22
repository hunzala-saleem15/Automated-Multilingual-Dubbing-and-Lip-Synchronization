import random
import cv2
import numpy as np
import torch
from torchvision import transforms


def collater_audio(audios, audio_size):
    audio_feat_shape = list(audios[0].shape[1:])
    collated_audios = audios[0].new_zeros(
        [len(audios), audio_size] + audio_feat_shape
    )
    padding_mask = torch.BoolTensor(len(audios), audio_size).fill_(False)

    for i, audio in enumerate(audios):
        diff = len(audio) - audio_size
        if diff == 0:
            collated_audios[i] = audio
        elif diff < 0:
            collated_audios[i] = torch.cat(
                [audio, audio.new_full([-diff] + audio_feat_shape, 0.0)]
            )
            padding_mask[i, diff:] = True
        else:
            import sys
            sys.exit("Audio segment is longer than the longest")

    if len(audios[0].shape) == 2:
        collated_audios = collated_audios.transpose(1, 2)  # [B, T, F] -> [B, F, T]
    else:
        collated_audios = collated_audios.permute(
            (0, 4, 1, 2, 3)
        ).contiguous()  # [B, T, H, W, C] -> [B, C, T, H, W]

    return collated_audios, padding_mask


class Compose(object):
    """Compose several preprocess together.
    Args:
        preprocess (list of ``Preprocess`` objects): list of preprocess to compose.
    """

    def __init__(self, preprocess):
        self.preprocess = preprocess

    def __call__(self, sample):
        for t in self.preprocess:
            sample = t(sample)
        return sample

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.preprocess:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class Normalize(object):
    """Normalize a ndarray image with mean and standard deviation."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, frames):
        frames = (frames - self.mean) / self.std
        return frames

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(
            self.mean, self.std
        )


class CenterCrop(object):
    """Crop the given image at the center"""

    def __init__(self, size):
        self.size = size

    def __call__(self, frames):
        t, h, w = frames.shape
        th, tw = self.size
        delta_w = int(round((w - tw) / 2.0))
        delta_h = int(round((h - th) / 2.0))
        frames = frames[:, delta_h : delta_h + th, delta_w : delta_w + tw]
        return frames


class RandomCrop(object):
    """Crop the given image randomly"""

    def __init__(self, size):
        self.size = size

    def __call__(self, frames):
        t, h, w = frames.shape
        th, tw = self.size
        delta_w = random.randint(0, w - tw)
        delta_h = random.randint(0, h - th)
        frames = frames[:, delta_h : delta_h + th, delta_w : delta_w + tw]
        return frames

    def __repr__(self):
        return self.__class__.__name__ + "(size={0})".format(self.size)


class HorizontalFlip(object):
    """Flip image horizontally."""

    def __init__(self, flip_ratio):
        self.flip_ratio = flip_ratio

    def __call__(self, frames):
        t, h, w = frames.shape
        if random.random() < self.flip_ratio:
            for index in range(t):
                frames[index] = cv2.flip(frames[index], 1)
        return frames


transform = Compose(
    [
        Normalize(0.0, 255.0),
        CenterCrop((88, 88)),
        Normalize(0.421, 0.165),
    ]
)


def rgb2gray(g, dim):
    glist = g.split([1, 1, 1], dim=dim)
    return 0.299 * glist[2] + 0.587 * glist[1] + 0.114 * glist[0]


def affine_trans(imgs, video_size):
    videoSeq = list()
    for i, img in enumerate(imgs):
        new_images = list()
        for j, frame in enumerate(img):
            frame = rgb2gray(frame, 2).squeeze(dim=-1)
            new_images.append(frame)
        new_images = torch.stack(new_images, dim=0)
        videoSeq.append(transform(new_images).unsqueeze(dim=-1))
    collated_videos, padding_mask = collater_audio(videoSeq, video_size)
    return collated_videos


def emb_roi2im(pickedimg, imgs, bbxs, predictions, device):
    """
    Blends predicted face/lip patches back into the full-frame video
    using a Gaussian Feathered Alpha Mask to eliminate hard edges.
    """
    processed_videos = []
    pred_idx = 0

    for b in range(len(imgs)):
        video_frames = imgs[b]
        sample_bbxs = bbxs[b]
        output_frames = []

        for idx in range(len(video_frames)):
            # ---------- Safe Tensor → NumPy ----------
            if isinstance(video_frames[idx], torch.Tensor):
                frame = video_frames[idx].detach().cpu().numpy()
            else:
                frame = video_frames[idx]
            frame = frame.copy()

            bbx = sample_bbxs[idx]

            if bbx is None:
                output_frames.append(torch.from_numpy(frame).to(device))
                continue

            if isinstance(bbx, torch.Tensor):
                bbx = bbx.detach().cpu().numpy()

            bbx = np.asarray(bbx, dtype=np.float32).reshape(-1)

            if bbx.size < 4 or np.all(bbx == 0):
                output_frames.append(torch.from_numpy(frame).to(device))
                continue

            # ---------- Bounding box ----------
            x0, y0, x1, y1 = [int(v) for v in bbx[:4]]

            h_f, w_f = frame.shape[:2]
            x0 = max(0, min(x0, w_f - 1))
            y0 = max(0, min(y0, h_f - 1))
            x1 = max(x0 + 1, min(x1, w_f))
            y1 = max(y0 + 1, min(y1, h_f))

            target_w = x1 - x0
            target_h = y1 - y0

            if target_w < 2 or target_h < 2:
                output_frames.append(torch.from_numpy(frame).to(device))
                continue

            # ---------- Prediction index safety ----------
            if pred_idx >= len(predictions):
                output_frames.append(torch.from_numpy(frame).to(device))
                continue

            pred_patch = predictions[pred_idx].cpu().detach().numpy()

            # (C, H, W) → (H, W, C)
            if pred_patch.ndim == 3 and pred_patch.shape[0] == 3:
                pred_patch = np.transpose(pred_patch, (1, 2, 0))

            if pred_patch.max() <= 1.0:
                pred_patch = (pred_patch * 255.0).astype(np.uint8)
            else:
                pred_patch = pred_patch.astype(np.uint8)

            # ---------- Resize ----------
            resized_pred = cv2.resize(
                pred_patch, (target_w, target_h), interpolation=cv2.INTER_LINEAR
            )

            # Extra safety against tiny size mismatch
            original_crop = frame[y0:y1, x0:x1]
            if (
                original_crop.shape[0] != resized_pred.shape[0]
                or original_crop.shape[1] != resized_pred.shape[1]
            ):
                resized_pred = cv2.resize(
                    resized_pred,
                    (original_crop.shape[1], original_crop.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )

            # ---------- Gaussian Feathered Ellipse Mask ----------
            mask = np.zeros((target_h, target_w), dtype=np.float32)
            cv2.ellipse(
                mask,
                (target_w // 2, int(target_h * 0.55)),
                (int(target_w * 0.45), int(target_h * 0.35)),
                0, 0, 360, 1, -1,
            )

            k_w = max(3, (target_w // 5) | 1)
            k_h = max(3, (target_h // 5) | 1)
            mask = cv2.GaussianBlur(mask, (k_w, k_h), 0)
            mask = mask[..., None]

            # ---------- Soft Alpha Blending ----------
            original_crop = original_crop.astype(np.float32)
            generated_crop = resized_pred.astype(np.float32)

            blended = generated_crop * mask + original_crop * (1.0 - mask)
            frame[y0:y1, x0:x1] = np.clip(blended, 0, 255).astype(np.uint8)

            output_frames.append(torch.from_numpy(frame).to(device))
            pred_idx += 1

        processed_videos.append(output_frames)

    return processed_videos


def images2avhubert(pickedimg, imgs, bbxs, pre, video_size, device):
    imgs = emb_roi2im(pickedimg, imgs, bbxs, pre, device)
    processed_img = affine_trans(imgs, video_size).to(device)
    return processed_img