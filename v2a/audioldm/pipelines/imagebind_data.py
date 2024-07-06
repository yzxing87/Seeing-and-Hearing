#!/usr/bin/env python3
# Portions Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math

import torch
import torch.nn as nn
import torchaudio
from PIL import Image
from pytorchvideo import transforms as pv_transforms
from pytorchvideo.data.clip_sampling import ConstantClipsPerVideoSampler
from pytorchvideo.data.encoded_video import EncodedVideo
from torchvision import transforms
from torchvision.transforms._transforms_video import NormalizeVideo
import imageio, os
import numpy as np 
from imagebind.imagebind.models.multimodal_preprocessors import SimpleTokenizer

DEFAULT_AUDIO_FRAME_SHIFT_MS = 10  # in milliseconds

# BPE_PATH = "bpe/bpe_simple_vocab_16e6.txt.gz"
BPE_PATH = "imagebind/bpe/bpe_simple_vocab_16e6.txt.gz"


def waveform2melspec(waveform, sample_rate, num_mel_bins, target_length):
    # Based on https://github.com/YuanGongND/ast/blob/d7d8b4b8e06cdaeb6c843cdb38794c1c7692234c/src/dataloader.py#L102
    waveform -= waveform.mean()
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform,
        htk_compat=True,
        sample_frequency=sample_rate,
        use_energy=False,
        window_type="hanning",
        num_mel_bins=num_mel_bins,
        dither=0.0,
        frame_length=25,
        frame_shift=DEFAULT_AUDIO_FRAME_SHIFT_MS,
    )
    # Convert to [mel_bins, num_frames] shape
    fbank = fbank.transpose(0, 1)
    # Pad to target_length
    n_frames = fbank.size(1)
    p = target_length - n_frames
    # if p is too large (say >20%), flash a warning
    # if abs(p) / n_frames > 0.2:
    #     logging.warning(
    #         "Large gap between audio n_frames(%d) and "
    #         "target_length (%d). Is the audio_target_length "
    #         "setting correct?",
    #         n_frames,
    #         target_length,
    #     )
    # cut and pad
    if p > 0:
        fbank = torch.nn.functional.pad(fbank, (0, p), mode="constant", value=0)
    elif p < 0:
        fbank = fbank[:, 0:target_length]
    # Convert to [1, mel_bins, num_frames] shape, essentially like a 1
    # channel image
    fbank = fbank.unsqueeze(0)
    return fbank


def get_clip_timepoints(clip_sampler, duration):
    # Read out all clips in this video
    all_clips_timepoints = []
    is_last_clip = False
    end = 0.0
    while not is_last_clip:
        start, end, _, _, is_last_clip = clip_sampler(end, duration, annotation=None)
        all_clips_timepoints.append((start, end))
    return all_clips_timepoints


def load_and_transform_vision_data(image_paths, device):
    if image_paths is None:
        return None

    image_outputs = []
    for image_path in image_paths:
        data_transform = transforms.Compose(
            [
                transforms.Resize(
                    224, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
        with open(image_path, "rb") as fopen:
            image = Image.open(fopen).convert("RGB")

        image = data_transform(image).to(device)
        print('image in load_and_transform: ', image.shape, image.max(), image.min())
        image_outputs.append(image)
    return torch.stack(image_outputs, dim=0)


def load_and_transform_vision_data_from_tensor(image_tensor, device):
    # image_tensor: [1,3,512,512]
    # image_outputs = []

    data_transform = transforms.Compose(
        [
            # transforms.Resize(
            #     224, interpolation=transforms.InterpolationMode.BICUBIC
            # ),
            # transforms.Resize(
                # (224, 224)
            # ),
            # transforms.CenterCrop(224),
            # transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )


    image = data_transform(image_tensor).to(device)
    print('image in load_and_transform: ', image.shape, image.max(), image.min())
    # image_outputs.append(image)
    # return torch.stack(image_outputs, dim=0)
    return image



def load_and_transform_text(text, device):
    if text is None:
        return None
    tokenizer = SimpleTokenizer(bpe_path=BPE_PATH)
    tokens = [tokenizer(t).unsqueeze(0).to(device) for t in text]
    tokens = torch.cat(tokens, dim=0)
    return tokens


def load_and_transform_audio_data(
    audio_paths,
    device,
    num_mel_bins=128,
    target_length=204,
    sample_rate=16000,
    clip_duration=2,
    clips_per_video=3,
    mean=-4.268,
    std=9.138,
):
    if audio_paths is None:
        return None

    audio_outputs = []
    clip_sampler = ConstantClipsPerVideoSampler(
        clip_duration=clip_duration, clips_per_video=clips_per_video
    )

    for audio_path in audio_paths:
        # 1. load audio as waveform
        waveform, sr = torchaudio.load(audio_path)
        # 2. sample waveform with target sample_rate
        if sample_rate != sr:
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=sr, new_freq=sample_rate
            )
        
        # 3. get clip timepoints
        all_clips_timepoints = get_clip_timepoints(
            clip_sampler, waveform.size(1) / sample_rate
        )

        # 4. clip audio into clips and convert to melspec
        all_clips = []
        for clip_timepoints in all_clips_timepoints:
            # clip waveform
            waveform_clip = waveform[
                :,
                int(clip_timepoints[0] * sample_rate) : int(
                    clip_timepoints[1] * sample_rate
                ),
            ]
            # convert to melspec
            waveform_melspec = waveform2melspec(
                waveform_clip, sample_rate, num_mel_bins, target_length
            )
            all_clips.append(waveform_melspec)

        # 5. normalize
        normalize = transforms.Normalize(mean=mean, std=std)
        all_clips = [normalize(ac).to(device) for ac in all_clips]

        # 6. stack all clips
        all_clips = torch.stack(all_clips, dim=0)
        audio_outputs.append(all_clips)

    return torch.stack(audio_outputs, dim=0)



def load_and_transform_audio_data_from_waveform(
    waveform,
    device,
    num_mel_bins=128,
    target_length=204,
    org_sample_rate=16000,
    sample_rate=16000,
    clip_duration=2,
    clips_per_video=3,
    mean=-4.268,
    std=9.138,
):
    audio_outputs = []
    clip_sampler = ConstantClipsPerVideoSampler(
        clip_duration=clip_duration, clips_per_video=clips_per_video
    )

    # # 1. load audio as waveform
    # waveform, sr = torchaudio.load(audio_path)
    # 2. sample waveform with target sample_rate
    if sample_rate != org_sample_rate:
        waveform = torchaudio.functional.resample(
            waveform, orig_freq=org_sample_rate, new_freq=sample_rate
        )
    
    # 3. get clip timepoints
    all_clips_timepoints = get_clip_timepoints(
        clip_sampler, waveform.size(1) / sample_rate
    )

    # 4. clip audio into clips and convert to melspec
    all_clips = []
    for clip_timepoints in all_clips_timepoints:
        # clip waveform 
        waveform_clip = waveform[
            :,
            int(clip_timepoints[0] * sample_rate) : int(
                clip_timepoints[1] * sample_rate
            ),
        ]
        # convert to melspec
        waveform_melspec = waveform2melspec(
            waveform_clip, sample_rate, num_mel_bins, target_length
        )
        all_clips.append(waveform_melspec)

    # 5. normalize
    normalize = transforms.Normalize(mean=mean, std=std)
    all_clips = [normalize(ac).to(device) for ac in all_clips]

    # 6. stack all clips
    all_clips = torch.stack(all_clips, dim=0)
    audio_outputs.append(all_clips)

    return torch.stack(audio_outputs, dim=0)


def crop_boxes(boxes, x_offset, y_offset):
    """
    Perform crop on the bounding boxes given the offsets.
    Args:
        boxes (ndarray or None): bounding boxes to perform crop. The dimension
            is `num boxes` x 4.
        x_offset (int): cropping offset in the x axis.
        y_offset (int): cropping offset in the y axis.
    Returns:
        cropped_boxes (ndarray or None): the cropped boxes with dimension of
            `num boxes` x 4.
    """
    cropped_boxes = boxes.copy()
    cropped_boxes[:, [0, 2]] = boxes[:, [0, 2]] - x_offset
    cropped_boxes[:, [1, 3]] = boxes[:, [1, 3]] - y_offset

    return cropped_boxes


def uniform_crop(images, size, spatial_idx, boxes=None, scale_size=None):
    """
    Perform uniform spatial sampling on the images and corresponding boxes.
    Args:
        images (tensor): images to perform uniform crop. The dimension is
            `num frames` x `channel` x `height` x `width`.
        size (int): size of height and weight to crop the images.
        spatial_idx (int): 0, 1, or 2 for left, center, and right crop if width
            is larger than height. Or 0, 1, or 2 for top, center, and bottom
            crop if height is larger than width.
        boxes (ndarray or None): optional. Corresponding boxes to images.
            Dimension is `num boxes` x 4.
        scale_size (int): optinal. If not None, resize the images to scale_size before
            performing any crop.
    Returns:
        cropped (tensor): images with dimension of
            `num frames` x `channel` x `size` x `size`.
        cropped_boxes (ndarray or None): the cropped boxes with dimension of
            `num boxes` x 4.
    """
    assert spatial_idx in [0, 1, 2]
    ndim = len(images.shape)
    if ndim == 3:
        images = images.unsqueeze(0)
    height = images.shape[2]
    width = images.shape[3]

    if scale_size is not None:
        if width <= height:
            width, height = scale_size, int(height / width * scale_size)
        else:
            width, height = int(width / height * scale_size), scale_size
        images = torch.nn.functional.interpolate(
            images,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )

    y_offset = int(math.ceil((height - size) / 2))
    x_offset = int(math.ceil((width - size) / 2))

    if height > width:
        if spatial_idx == 0:
            y_offset = 0
        elif spatial_idx == 2:
            y_offset = height - size
    else:
        if spatial_idx == 0:
            x_offset = 0
        elif spatial_idx == 2:
            x_offset = width - size
    cropped = images[:, :, y_offset : y_offset + size, x_offset : x_offset + size]
    cropped_boxes = crop_boxes(boxes, x_offset, y_offset) if boxes is not None else None
    if ndim == 3:
        cropped = cropped.squeeze(0)
    return cropped, cropped_boxes


class SpatialCrop(nn.Module):
    """
    Convert the video into 3 smaller clips spatially. Must be used after the
        temporal crops to get spatial crops, and should be used with
        -2 in the spatial crop at the slowfast augmentation stage (so full
        frames are passed in here). Will return a larger list with the
        3x spatial crops as well.
    """

    def __init__(self, crop_size: int = 224, num_crops: int = 3):
        super().__init__()
        self.crop_size = crop_size
        if num_crops == 3:
            self.crops_to_ext = [0, 1, 2]
            self.flipped_crops_to_ext = []
        elif num_crops == 1:
            self.crops_to_ext = [1]
            self.flipped_crops_to_ext = []
        else:
            raise NotImplementedError("Nothing else supported yet")

    def forward(self, videos):
        """
        Args:
            videos: A list of C, T, H, W videos.
        Returns:
            videos: A list with 3x the number of elements. Each video converted
                to C, T, H', W' by spatial cropping.
        """
        assert isinstance(videos, list), "Must be a list of videos after temporal crops"
        assert all([video.ndim == 4 for video in videos]), "Must be (C,T,H,W)"
        res = []
        for video in videos:
            for spatial_idx in self.crops_to_ext:
                res.append(uniform_crop(video, self.crop_size, spatial_idx)[0])
            if not self.flipped_crops_to_ext:
                continue
            flipped_video = transforms.functional.hflip(video)
            for spatial_idx in self.flipped_crops_to_ext:
                res.append(uniform_crop(flipped_video, self.crop_size, spatial_idx)[0])
        return res


def load_and_transform_video_data(
    video_paths,
    device,
    clip_duration=2,
    clips_per_video=5,
    sample_rate=16000,
    n_samples_per_clip=2,
):
    if video_paths is None:
        return None

    video_outputs = []
    video_transform = transforms.Compose(
        [
            pv_transforms.ShortSideScale(224),
            NormalizeVideo(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )

    clip_sampler = ConstantClipsPerVideoSampler(
        clip_duration=clip_duration, clips_per_video=clips_per_video
    )
    frame_sampler = pv_transforms.UniformTemporalSubsample(num_samples=n_samples_per_clip)

    resize = transforms.Resize((224, 224))


    for video_path in video_paths:
        video = EncodedVideo.from_path(
            video_path,
            decoder="decord",
            decode_audio=False,
            # **{"sample_rate": sample_rate},
        )

        all_clips_timepoints = get_clip_timepoints(clip_sampler, video.duration)
        # print('all_clips_timepoints: ', all_clips_timepoints)

        all_video = []
        for i_time, clip_timepoints in enumerate(all_clips_timepoints):
            # Read the clip, get frames
            clip = video.get_clip(clip_timepoints[0], clip_timepoints[1])
            if clip is None:
                raise ValueError("No clip found")
            video_clip = frame_sampler(clip["video"])
            video_clip = video_clip / 255.0  # since this is float, need 0-1

            all_video.append(video_clip)

        all_video_save = all_video.copy()
        all_video = [video_transform(clip) for clip in all_video]
        # all_video = SpatialCrop(224, num_crops=3)(all_video)
        all_video = [resize(vid) for vid in all_video]
        all_video_save = [resize(vid) for vid in all_video_save]

        all_video = torch.stack(all_video, dim=0)
        video_outputs.append(all_video)

    return torch.stack(video_outputs, dim=0).to(device)


def clip_video(video_tensor, t1, t2, video_fps):
    # video_tensor: [3, 16, 64, 64]
    start = int(t1*video_fps)
    end = int(t2*video_fps)
    return {'video': video_tensor[:, start:end, ...]}


# import random 

# def my_uniform_sampler(video_tensor, num_samples):
#     # video_tensor: [3, 16, 64, 64]
#     video_length = video_tensor.shape[1] # 16
#     # uniform sampling
#     indices = sorted(random.sample(range(video_length), num_samples)) 
#     return video_tensor[:, indices, ...]


# # [1, 3, 4, 224, 224]
# def load_and_transform_video_data_from_tensor_real(
#     video_tensor, # [1, 16, 64, 64, 3]
#     device,
#     # video_duration, # this is the video length in seconds
#     clip_duration=2,
#     clips_per_video=5,
#     sample_rate=16000,
#     n_samples_per_clip=2,
#     video_fps = 10,
# ):
#     video_outputs = []
#     # requires CTHW 
#     video_transform = transforms.Compose(
#         [
#             pv_transforms.ShortSideScale(224),
#             NormalizeVideo(
#                 mean=(0.48145466, 0.4578275, 0.40821073),
#                 std=(0.26862954, 0.26130258, 0.27577711),
#             ),
#         ]
#     )
#     # video_tensor = video_tensor.permute(0, 4, 1, 2, 3)

#     if len(video_tensor.shape) == 5:
#         video_length = video_tensor.shape[1] 
#     elif len(video_tensor.shape) == 4:
#         video_length = video_tensor.shape[0]
#     else:
#         raise ValueError('video tensor shape is not correct')
#     video_duration = video_length / video_fps # 1.6s 

#     clip_sampler = ConstantClipsPerVideoSampler(
#         clip_duration=clip_duration, clips_per_video=clips_per_video
#     )
#     frame_sampler = pv_transforms.UniformTemporalSubsample(num_samples=n_samples_per_clip)

#     resize = transforms.Resize((224, 224))

#     all_clips_timepoints = get_clip_timepoints(clip_sampler, video_duration) 

#     all_video = []
#     for vd_tensor in video_tensor:
#         for i_time, clip_timepoints in enumerate(all_clips_timepoints):
#             # Read the clip, get frames
#             # clip = video.get_clip(clip_timepoints[0], clip_timepoints[1])
#             clip = clip_video(vd_tensor, clip_timepoints[0], clip_timepoints[1], video_fps) # give starting and ending point, return a clip
#             # print('clip: ', clip.keys()) # video, audio, start_time, end_time 
#             # print('clip video shape: ', clip['video'].shape, type(clip['video'])) # [3, 17, 64, 64], torch.Tensor
#             if clip is None:
#                 raise ValueError("No clip found")
#             # video_clip = frame_sampler(clip["video"])
#             video_clip = my_uniform_sampler(clip["video"], num_samples=n_samples_per_clip)

#             video_clip = video_clip / 255.0  # since this is float, need 0-1

#             # print()
#             all_video.append(video_clip)

#     all_video_save = all_video.copy()
#     all_video = [video_transform(clip) for clip in all_video]
#     # all_video = SpatialCrop(224, num_crops=3)(all_video)
#     all_video = [resize(vid) for vid in all_video]
#     all_video_save = [resize(vid) for vid in all_video_save]


#     all_video = torch.stack(all_video, dim=0)
#     video_outputs.append(all_video)

#     return torch.stack(video_outputs, dim=0).to(device)
