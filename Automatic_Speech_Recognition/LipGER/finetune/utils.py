#!/usr/bin/env python

import cv2
import random
import numpy as np
import torch
from pathlib import Path
import torch.nn.functional as F

__all__ = ['Compose', 'Normalize', 'CenterCrop', 'RgbToGray', 'RandomCrop',
           'HorizontalFlip', 'AddNoise', 'NormalizeUtterance',
           'chunked_cross_entropy', 'check_valid_checkpoint_dir',
           'LipEncoderWrapper']

# ===========================
# PREPROCESSING CLASSES
# ===========================

class Compose(object):
    def __init__(self, preprocess):
        self.preprocess = preprocess

    def __call__(self, sample):
        for t in self.preprocess:
            sample = t(sample)
        return sample

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.preprocess:
            format_string += '\n    {0}'.format(t)
        format_string += '\n)'
        return format_string


class RgbToGray(object):
    def __call__(self, frames):
        frames = np.stack([cv2.cvtColor(_, cv2.COLOR_RGB2GRAY) for _ in frames], axis=0)
        return frames

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, frames):
        frames = (frames - self.mean) / self.std
        return frames

    def __repr__(self):
        return self.__class__.__name__+'(mean={0}, std={1})'.format(self.mean, self.std)


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, frames):
        t, h, w = frames.shape
        th, tw = self.size
        delta_w = int(round((w - tw))/2.)
        delta_h = int(round((h - th))/2.)
        frames = frames[:, delta_h:delta_h+th, delta_w:delta_w+tw]
        return frames


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, frames):
        t, h, w = frames.shape
        th, tw = self.size
        delta_w = random.randint(0, w-tw)
        delta_h = random.randint(0, h-th)
        frames = frames[:, delta_h:delta_h+th, delta_w:delta_w+tw]
        return frames

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class HorizontalFlip(object):
    def __init__(self, flip_ratio):
        self.flip_ratio = flip_ratio

    def __call__(self, frames):
        t, h, w = frames.shape
        if random.random() < self.flip_ratio:
            for index in range(t):
                frames[index] = cv2.flip(frames[index], 1)
        return frames


class NormalizeUtterance():
    def __call__(self, signal):
        signal_std = 0. if np.std(signal)==0. else np.std(signal)
        signal_mean = np.mean(signal)
        return (signal - signal_mean) / signal_std


class AddNoise(object):
    def __init__(self, noise, snr_levels=[-5, 0, 5, 10, 15, 20, 9999]):
        assert noise.dtype in [np.float32, np.float64], "noise only supports float data type"
        self.noise = noise
        self.snr_levels = snr_levels

    def get_power(self, clip):
        clip2 = clip.copy()
        clip2 = clip2 **2
        return np.sum(clip2) / (len(clip2) * 1.0)

    def __call__(self, signal):
        assert signal.dtype in [np.float32, np.float64], "signal only supports float32/64 data type"
        snr_target = random.choice(self.snr_levels)
        if snr_target == 9999:
            return signal
        else:
            start_idx = random.randint(0, len(self.noise)-len(signal))
            noise_clip = self.noise[start_idx:start_idx+len(signal)]
            sig_power = self.get_power(signal)
            noise_clip_power = self.get_power(noise_clip)
            factor = (sig_power / noise_clip_power ) / (10**(snr_target / 10.0))
            desired_signal = (signal + noise_clip*np.sqrt(factor)).astype(np.float32)
            return desired_signal


# ===========================
# LipGER FUNCTIONS
# ===========================

def chunked_cross_entropy(logits, labels, chunk_size=1024):
    B, T, V = logits.size()
    logits = logits.view(-1, V)
    labels = labels.view(-1)
    loss_fn = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    for start in range(0, logits.size(0), chunk_size):
        end = start + chunk_size
        total_loss += loss_fn(logits[start:end], labels[start:end])
    return total_loss / (logits.size(0) / chunk_size)


def check_valid_checkpoint_dir(dir_path):
    if not Path(dir_path).exists():
        raise FileNotFoundError(f"Checkpoint directory {dir_path} not found!")


# ===========================
# LIP ENCODER WRAPPER
# ===========================

class LipEncoderWrapper(torch.nn.Module):
    """
    Wrap pre-trained lip encoder checkpoint for feature extraction.
    Input: video frames [T, H, W, C] or [T, H, W] (grayscale)
    Output: lip embeddings [T, D]
    """
    def __init__(self, lipencoder_path, device='cuda'):
        super().__init__()
        self.device = device
        self.model = torch.load(lipencoder_path, map_location=device)
        self.model.eval()
        self.model.to(device)

    @torch.no_grad()
    def forward(self, frames):
        """
        frames: np.ndarray or torch.Tensor [T,H,W,C] or [T,H,W]
        returns: torch.Tensor [T, feature_dim]
        """
        if isinstance(frames, np.ndarray):
            frames = torch.tensor(frames, dtype=torch.float32)

        # [T,H,W] → [T,1,H,W]
        if frames.ndim == 3:
            frames = frames.unsqueeze(1)

        frames = frames.to(self.device)
        emb = self.model(frames)  # depends on encoder forward
        return emb
