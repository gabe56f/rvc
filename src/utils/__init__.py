from typing import Tuple, Literal

import torch
import torch.nn.functional as F
import numpy as np
import ffmpeg

from .mel_utils import (
    torch_dynamic_range_compression,
    torch_dynamic_range_decompression,
    torch_spectral_normalize,
    torch_spectral_denormalize,
    torch_spectrogram,
    torch_spec_to_mel,
    torch_mel_spectrogram,
)

from . import spec_utils

from .train_utils import (
    EpochRecorder,
    load_checkpoint,
    save_checkpoint,
    latest_checkpoint_path,
    feature_loss,
    discriminator_loss,
    generator_loss,
    kl_loss,
)

from .data_utils import (
    TextAudioLoaderMultiNSFsid,
    TextAudioCollateMultiNSFsid,
    DistributedBucketSampler,
)


def load_audio(file: str, sr: int) -> Tuple[np.ndarray, int]:
    audio, _ = (
        ffmpeg.input(file, threads=0)
        .output("-", format="f32le", acodec="pcm_f32le", ac=1, ar=sr)
        .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
    )
    audio = np.frombuffer(audio, np.float32).flatten()
    # audio_max = np.abs(audio).max() / 0.95
    # if audio_max > 1:
    #     audio /= audio_max
    return audio, sr


def get_rms(
    y: torch.Tensor,
    frame_length: int = 2048,
    hop_length: int = 512,
    pad_mode: Literal["constant"] = "constant",
) -> torch.Tensor:
    padding_left = frame_length // 2
    padding_right = frame_length - padding_left
    y = F.pad(y, (padding_left, padding_right), mode=pad_mode)

    x = y.unfold(-1, frame_length, hop_length)

    power = torch.mean(torch.square(torch.abs(x)), dim=-1, keepdim=True)

    rms = torch.sqrt(power)

    return rms
