from torchvision import transforms
import torch
import cv2
import random


def collater_audio(audios, audio_size):
    audio_feat_shape = list(audios[0].shape[1:])
    collated_audios = audios[0].new_zeros([len(audios), audio_size]+audio_feat_shape)
    padding_mask = (
        torch.BoolTensor(len(audios), audio_size).fill_(False) #
    )
    for i, audio in enumerate(audios):
        diff = len(audio) - audio_size
        if diff == 0:
            collated_audios[i] = audio
        elif diff < 0:
            collated_audios[i] = torch.cat(
                [audio, audio.new_full([-diff]+audio_feat_shape, 0.0)]
            )
            padding_mask[i, diff:] = True
        else:
            import sys
            sys.exit('Audio segment is longer than the loggest')
    if len(audios[0].shape) == 2:
        collated_audios = collated_audios.transpose(1, 2) # [B, T, F] -> [B, F, T]
    else:
        collated_audios = collated_audios.permute((0, 4, 1, 2, 3)).contiguous() # [B, T, H, W, C] -> [B, C, T, H, W]
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
        format_string = self.__class__.__name__ + '('
        for t in self.preprocess:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class Normalize(object):
    """Normalize a ndarray image with mean and standard deviation.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, frames):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized Tensor image.
        """
        frames = (frames - self.mean) / self.std
        return frames

    def __repr__(self):
        return self.__class__.__name__+'(mean={0}, std={1})'.format(self.mean, self.std)


class CenterCrop(object):
    """Crop the given image at the center
    """
    def __init__(self, size):
        self.size = size

    def __call__(self, frames):
        """
        Args:
            img (numpy.ndarray): Images to be cropped.
        Returns:
            numpy.ndarray: Cropped image.
        """
        t, h, w = frames.shape
        th, tw = self.size
        delta_w = int(round((w - tw))/2.)
        delta_h = int(round((h - th))/2.)
        frames = frames[:, delta_h:delta_h+th, delta_w:delta_w+tw]
        return frames


class RandomCrop(object):
    """Crop the given image at the center
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, frames):
        """
        Args:
            img (numpy.ndarray): Images to be cropped.
        Returns:
            numpy.ndarray: Cropped image.
        """
        t, h, w = frames.shape
        th, tw = self.size
        delta_w = random.randint(0, w-tw)
        delta_h = random.randint(0, h-th)
        frames = frames[:, delta_h:delta_h+th, delta_w:delta_w+tw]
        return frames

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class HorizontalFlip(object):
    """Flip image horizontally.
    """

    def __init__(self, flip_ratio):
        self.flip_ratio = flip_ratio

    def __call__(self, frames):
        """
        Args:
            img (numpy.ndarray): Images to be flipped with a probability flip_ratio
        Returns:
            numpy.ndarray: Cropped image.
        """
        t, h, w = frames.shape
        if random.random() < self.flip_ratio:
            for index in range(t):
                frames[index] = cv2.flip(frames[index], 1)
        return frames


transform = Compose([
    Normalize(0.0, 255.0),
    CenterCrop((88, 88)),
    Normalize(0.421, 0.165)])


def rgb2gray(g, dim):
    glist = g.split([1,1,1], dim=dim)
    return 0.299 * glist[2] + 0.587 * glist[1] + 0.114 * glist[0]


def affine_trans(imgs, video_size):
    h, w, _ = imgs[0][0].shape
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


import cv2
import torch
import numpy as np

import cv2
import torch
import numpy as np

import cv2
import torch
import numpy as np

def emb_roi2im(pickedimg, imgs, bbxs, prediction, device):
    """
    Robust emb_roi2im function handling nested bbx lists/tensors.
    """
    pre = prediction
    trackid = 0
    
    for i in range(len(imgs)):
        img = imgs[i]
        bbx_raw = bbxs[i]
        
        # Helper to flatten nested lists/tensors to 1D numpy array
        if isinstance(bbx_raw, torch.Tensor):
            bbx_arr = bbx_raw.detach().cpu().numpy().flatten()
        else:
            bbx_arr = np.array(bbx_raw).flatten()

        # Extract coordinates safely from 1D array
        x1, y1, x2, y2 = float(bbx_arr[0]), float(bbx_arr[1]), float(bbx_arr[2]), float(bbx_arr[3])
        
        target_w = int(x2 - x1)
        target_h = int(y2 - y1)
        
        if target_w <= 0 or target_h <= 0:
            continue

        for j in range(len(pickedimg[i])):
            frame_tensor = pre[trackid + j] * 255.0
            
            # Convert Tensor (C, H, W) to NumPy Array (H, W, C) for OpenCV
            frame_np = frame_tensor.detach().clamp(0, 255).byte().cpu().numpy()
            if frame_np.ndim == 3 and frame_np.shape[0] in [1, 3]:
                frame_np = np.transpose(frame_np, (1, 2, 0))

            # OpenCV Resize
            resized_np = cv2.resize(frame_np, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

            # Convert back to Torch Tensor
            resized = torch.from_numpy(resized_np).float().to(device)
            
            ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)
            
            if resized.ndim == 2:
                resized = resized.unsqueeze(-1)
                
            imgs[i][j][iy1:iy2, ix1:ix2] = resized
            
        trackid += len(pickedimg[i])

    return imgs


def images2avhubert(pickedimg, imgs, bbxs, pre, video_size, device):
    imgs = emb_roi2im(pickedimg, imgs, bbxs, pre, device)
    processed_img = affine_trans(imgs, video_size).to(device)
    return processed_img


