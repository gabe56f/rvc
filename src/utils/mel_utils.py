import logging

import torch
import torch.nn.functional as F
import torch.utils.data
from librosa.filters import mel as librosa_mel


logger = logging.getLogger(__name__)

mel_basis = {}
hann_window = {}


def torch_dynamic_range_compression(
    x: torch.Tensor, C: int = 1, clip_val: float = 1e-5
) -> torch.Tensor:
    return torch.log(torch.clamp(x, min=clip_val) * C)


def torch_dynamic_range_decompression(x: torch.Tensor, C: int = 1) -> torch.Tensor:
    return torch.exp(x) / C


def torch_spectral_normalize(magnitudes: torch.Tensor) -> torch.Tensor:
    return torch_dynamic_range_compression(magnitudes)


def torch_spectral_denormalize(magnitudes: torch.Tensor) -> torch.Tensor:
    return torch_dynamic_range_decompression(magnitudes)


def torch_spectrogram(
    y: torch.Tensor,
    n_fft: int,
    hop_length: int,
    window_size: int,
    center: bool = False,
) -> torch.Tensor:
    if torch.min(y) < -1.07:
        logger.debug("min value is ", torch.min(y).item())
    if torch.max(y) > 1.07:
        logger.debug("max value is ", torch.max(y).item())

    global hann_window
    if y.dtype not in hann_window:
        hann_window[y.dtype] = torch.hann_window(window_size).to(y.device, y.dtype)

    y = F.pad(
        y.unsqueeze(1),
        (
            int((n_fft - hop_length) / 2),
            int((n_fft - hop_length) / 2),
        ),
        mode="reflect",
    )
    y = y.squeeze(1)
    spec = torch.stft(
        y,
        n_fft,
        hop_length,
        window_size,
        hann_window[y.dtype],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    spec = torch.view_as_real(spec)
    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
    return spec


def torch_spec_to_mel(
    spec: torch.Tensor,
    n_fft: int,
    n_mels: int,
    sampling_rate: int,
    fmin: float,
    fmax: float,
) -> torch.Tensor:
    global mel_basis
    if spec.dtype not in mel_basis:
        mel = librosa_mel(
            sr=sampling_rate, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax
        )
        mel_basis[spec.dtype] = torch.from_numpy(mel).to(spec.device, spec.dtype)

    melspec = torch.matmul(mel_basis[spec.dtype], spec)
    melspec = torch_spectral_normalize(melspec)
    return melspec


def torch_mel_spectrogram(
    y: torch.Tensor,
    n_fft: int,
    n_mels: int,
    sampling_rate: int,
    hop_length: int,
    window_size: int,
    fmin: float,
    fmax: float,
    center: bool = False,
) -> torch.Tensor:
    spec = torch_spectrogram(
        y, n_fft, sampling_rate, hop_length, window_size, center=center
    )
    mel = torch_spec_to_mel(spec, n_fft, n_mels, sampling_rate, fmin, fmax)
    return mel
