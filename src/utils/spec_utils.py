from typing import TYPE_CHECKING, Optional, Tuple
import math

import numpy as np
import torch
from torchaudio import functional as Fa

if TYPE_CHECKING:
    from ..models.separation.uvr import UVRParameters


def istft(spec: torch.Tensor, hop_len: int) -> torch.Tensor:
    if len(spec.shape) == 2:
        n_fft = spec.shape[0] * 2 - 2
    else:
        n_fft = spec.shape[1] * 2 - 2
    wave_tensor: torch.Tensor = torch.istft(
        spec,
        n_fft=n_fft,  # Calculate n_fft from spectrogram shape
        hop_length=hop_len,
        window=torch.hann_window(
            n_fft, device=spec.device
        ),  # Assuming Hann window was used
        center=True,  # Match librosa's default behavior
        normalized=False,  # Match librosa's default behavior
        onesided=True,  # Match librosa's default behavior
        length=None,  # Adjust if you need to specify output length
    )
    return wave_tensor


def stft(wave: torch.Tensor, hop_len: int, n_fft: int) -> np.ndarray:
    spec_tensor = torch.stft(
        wave,
        n_fft=n_fft,
        hop_length=hop_len,
        window=torch.hann_window(
            n_fft, device=wave.device
        ),  # Assuming Hann window is used
        center=True,  # Match librosa's default behavior
        normalized=False,  # Match librosa's default behavior
        onesided=True,  # Match librosa's default behavior
        return_complex=True,
    )
    return spec_tensor.cpu().numpy()


def crop_center(h1: np.ndarray, h2: np.ndarray) -> np.ndarray:
    h1_shape = h1.size()
    h2_shape = h2.size()

    if h1_shape[3] == h2_shape[3]:
        return h1
    elif h1_shape[3] < h2_shape[3]:
        raise ValueError("h1_shape[3] must be greater than h2_shape[3]")

    # s_freq = (h2_shape[2] - h1_shape[2]) // 2
    # e_freq = s_freq + h1_shape[2]
    s_time = (h1_shape[3] - h2_shape[3]) // 2
    e_time = s_time + h2_shape[3]
    h1 = h1[:, :, :, s_time:e_time]

    return h1


def wave_to_spectrogram(
    wave: np.ndarray,
    hop_length: int,
    n_fft: int,
    mid_side: bool = False,
    mid_side_b2: bool = False,
    reverse: bool = False,
    device="cuda",
) -> np.ndarray:
    wave = torch.from_numpy(wave).to(device)
    if reverse:
        wave_left = torch.flip(wave[0])
        wave_right = torch.flip(wave[1])
    elif mid_side:
        wave_left = (wave[0] + wave[1]) / 2
        wave_right = wave[0] - wave[1]
    elif mid_side_b2:
        wave_left = (wave[0] + wave[1] * 1.25) / 2
        wave_right = wave[0] * 1.25 - wave[1]
    else:
        wave_left = wave[0]
        wave_right = wave[1]

    spec_left = stft(wave_left, hop_length, n_fft)
    spec_right = stft(wave_right, hop_length, n_fft)

    spec = np.asfortranarray([spec_left, spec_right])
    return spec


def combine_spectrograms(specs: np.ndarray, params: "UVRParameters") -> np.ndarray:
    min_length = min([specs[i].shape[2] for i in specs])
    spec_c = np.zeros(
        shape=(2, params.param["bins"] + 1, min_length), dtype=np.complex64
    )
    offset = 0
    bands_n = len(params.param["band"])

    for d in range(1, bands_n + 1):
        crop_start = params.param["band"][d]["crop_start"]
        crop_stop = params.param["band"][d]["crop_stop"]
        height = crop_stop - crop_start
        spec_c[:, offset : offset + height, :min_length] = specs[d][
            :, crop_start:crop_stop, :min_length
        ]
        offset += height

    if offset > params.param["bins"]:
        raise ValueError("Too many bins")

    # lowpass fiter
    if (
        params.param["pre_filter_start"] > 0
    ):  # and mp.param['band'][bands_n]['res_type'] in ['scipy', 'polyphase']:
        if bands_n == 1:
            spec_c = fft_lp_filter(
                spec_c,
                params.param["pre_filter_start"],
                params.param["pre_filter_stop"],
            )
        else:
            gp = 1
            for b in range(
                params.param["pre_filter_start"] + 1, params.param["pre_filter_stop"]
            ):
                g = math.pow(
                    10, -(b - params.param["pre_filter_start"]) * (3.5 - gp) / 20.0
                )
                gp = g
                spec_c[:, b, :] *= g

    return np.asfortranarray(spec_c)


def spectrogram_to_image(spec: np.ndarray, mode: str = "magnitude") -> np.ndarray:
    if mode == "magnitude":
        if np.iscomplexobj(spec):
            y = np.abs(spec)
        else:
            y = spec
        y = np.log10(y**2 + 1e-8)
    elif mode == "phase":
        if np.iscomplexobj(spec):
            y = np.angle(spec)
        else:
            y = spec

    y -= y.min()
    y *= 255 / y.max()
    img = np.uint8(y)

    if y.ndim == 3:
        img = img.transpose(1, 2, 0)
        img = np.concatenate([np.max(img, axis=2, keepdims=True), img], axis=2)

    return img


def mask_silence(
    mag: np.ndarray,
    ref: np.ndarray,
    thres: float = 0.2,
    min_range: int = 64,
    fade_size: int = 32,
) -> np.ndarray:
    if min_range < fade_size * 2:
        raise ValueError("min_range must be >= fade_area * 2")

    mag = mag.copy()

    idx = np.where(ref.mean(axis=(0, 1)) < thres)[0]
    starts = np.insert(idx[np.where(np.diff(idx) != 1)[0] + 1], 0, idx[0])
    ends = np.append(idx[np.where(np.diff(idx) != 1)[0]], idx[-1])
    uninformative = np.where(ends - starts > min_range)[0]
    if len(uninformative) > 0:
        starts = starts[uninformative]
        ends = ends[uninformative]
        old_e = None
        for s, e in zip(starts, ends):
            if old_e is not None and s - old_e < fade_size:
                s = old_e - fade_size * 2

            if s != 0:
                weight = np.linspace(0, 1, fade_size)
                mag[:, :, s : s + fade_size] += weight * ref[:, :, s : s + fade_size]
            else:
                s -= fade_size

            if e != mag.shape[2]:
                weight = np.linspace(1, 0, fade_size)
                mag[:, :, e - fade_size : e] += weight * ref[:, :, e - fade_size : e]
            else:
                e += fade_size

            mag[:, :, s + fade_size : e - fade_size] += ref[
                :, :, s + fade_size : e - fade_size
            ]
            old_e = e

    return mag


def align_wave_head_and_tail(
    a: np.ndarray, b: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    length = min([a[0].size, b[0].size])

    return a[:length, :length], b[:length, :length]


def spectrogram_to_wave(
    spec: torch.Tensor,
    hop_length: int,
    mid_side: bool,
    mid_side_b2: bool,
    reverse: bool,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    spec_left = spec[0]
    spec_right = spec[1]

    wave_left = istft(spec_left, hop_length)
    wave_right = istft(spec_right, hop_length)

    if reverse:
        return torch.stack([torch.flip(wave_left), torch.flip(wave_right)]).to(dtype)
    elif mid_side:
        return torch.stack([(wave_left + wave_right) / 2, wave_left - wave_right]).to(
            dtype
        )
    elif mid_side_b2:
        return torch.stack(
            [wave_right / 1.25 + 0.4 * wave_left, wave_left / 1.25 - 0.4 * wave_right]
        ).to(dtype)
    else:
        return torch.stack([wave_left, wave_right]).to(dtype)


def cmb_spectrogram_to_wave(
    spec: np.ndarray,
    params: "UVRParameters",
    extra_bins_h: Optional[int] = None,
    extra_bins: Optional[np.ndarray] = None,
    device="cuda",
    dtype=torch.bfloat16,
) -> np.ndarray:
    bands_n = len(params.param["band"])
    offset = 0

    spec_m = torch.from_numpy(spec).to(device, dtype=torch.complex64)
    if extra_bins_h:
        extra_bins = torch.from_numpy(extra_bins).to(device, dtype=torch.complex64)
    for d in range(1, bands_n + 1):
        bp = params.param["band"][d]
        spec_s = torch.zeros(
            size=(2, bp["n_fft"] // 2 + 1, spec_m.shape[2]),
            dtype=torch.complex64,
            device=device,
        )
        h = bp["crop_stop"] - bp["crop_start"]
        spec_s[:, bp["crop_start"] : bp["crop_stop"], :] = spec_m[
            :, offset : offset + h, :
        ]

        offset += h
        if d == bands_n:  # higher
            if extra_bins_h:  # if --high_end_process bypass
                max_bin = bp["n_fft"] // 2
                spec_s[:, max_bin - extra_bins_h : max_bin, :] = extra_bins[
                    :, :extra_bins_h, :
                ]
            if bp["hpf_start"] > 0:
                spec_s = fft_hp_filter(spec_s, bp["hpf_start"], bp["hpf_stop"] - 1)
            if bands_n == 1:
                wave = spectrogram_to_wave(
                    spec_s,
                    bp["hl"],
                    params.param["mid_side"],
                    params.param["mid_side_b2"],
                    params.param["reverse"],
                    dtype,
                )
            else:
                wave = wave + spectrogram_to_wave(
                    spec_s,
                    bp["hl"],
                    params.param["mid_side"],
                    params.param["mid_side_b2"],
                    params.param["reverse"],
                    dtype,
                )
        else:
            sr = params.param["band"][d + 1]["sr"]
            if d == 1:  # lower
                spec_s = fft_lp_filter(spec_s, bp["lpf_start"], bp["lpf_stop"])
                wave = Fa.resample(
                    spectrogram_to_wave(
                        spec_s,
                        bp["hl"],
                        params.param["mid_side"],
                        params.param["mid_side_b2"],
                        params.param["reverse"],
                        dtype,
                    ),
                    bp["sr"],
                    sr,
                )
            else:  # mid
                spec_s = fft_hp_filter(spec_s, bp["hpf_start"], bp["hpf_stop"] - 1)
                spec_s = fft_lp_filter(spec_s, bp["lpf_start"], bp["lpf_stop"])
                wave2 = wave + spectrogram_to_wave(
                    spec_s,
                    bp["hl"],
                    params.param["mid_side"],
                    params.param["mid_side_b2"],
                    params.param["reverse"],
                    dtype,
                )
                wave = Fa.resample(wave2, bp["sr"], sr)

    return wave.cpu().float().numpy().T  # transpose


def fft_lp_filter(spec, bin_start, bin_stop):
    g = 1.0
    for b in range(bin_start, bin_stop):
        g -= 1 / (bin_stop - bin_start)
        spec[:, b, :] = g * spec[:, b, :]

    spec[:, bin_stop:, :] *= 0

    return spec


def fft_hp_filter(spec, bin_start, bin_stop):
    g = 1.0
    for b in range(bin_start, bin_stop, -1):
        g -= 1 / (bin_start - bin_stop)
        spec[:, b, :] = g * spec[:, b, :]

    spec[:, 0 : bin_stop + 1, :] *= 0

    return spec


def mirroring(a, spec_m, input_high_end, mp):
    if "mirroring" == a:
        mirror = np.flip(
            np.abs(
                spec_m[
                    :,
                    mp.param["pre_filter_start"]
                    - 10
                    - input_high_end.shape[1] : mp.param["pre_filter_start"]
                    - 10,
                    :,
                ]
            ),
            1,
        )
        mirror = mirror * np.exp(1.0j * np.angle(input_high_end))

        return np.where(
            np.abs(input_high_end) <= np.abs(mirror), input_high_end, mirror
        )

    if "mirroring2" == a:
        mirror = np.flip(
            np.abs(
                spec_m[
                    :,
                    mp.param["pre_filter_start"]
                    - 10
                    - input_high_end.shape[1] : mp.param["pre_filter_start"]
                    - 10,
                    :,
                ]
            ),
            1,
        )
        mi = np.multiply(mirror, input_high_end * 1.7)

        return np.where(np.abs(input_high_end) <= np.abs(mi), input_high_end, mi)


def ensembling(a, specs):
    for i in range(1, len(specs)):
        if i == 1:
            spec = specs[0]

        ln = min([spec.shape[2], specs[i].shape[2]])
        spec = spec[:, :, :ln]
        specs[i] = specs[i][:, :, :ln]

        if "min_mag" == a:
            spec = np.where(np.abs(specs[i]) <= np.abs(spec), specs[i], spec)
        if "max_mag" == a:
            spec = np.where(np.abs(specs[i]) >= np.abs(spec), specs[i], spec)

    return spec
