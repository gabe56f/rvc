from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict

import torch
import numpy as np
import librosa


OPT_BATCH_SIZE = 1


@dataclass
class STFTConfig:
    n_fft: int = 8192
    hop_length: int = 1024
    num_audio_channels: int = 2

    dim_f: int = 4096
    dim_t: int = 256
    chunk_size: int = 261120

    sample_rate: int = 44100
    min_mean_abs = 1e-3

    bottleneck_factor: int = 4
    growth: int = 128

    num_blocks_per_scale: int = 2
    num_channels: int = 128
    num_scales: int = 5
    num_subbands: int = 4
    scale: List[int] = field(default_factory=lambda: [2, 2])


class STFT:
    def __init__(self, config: Optional[STFTConfig]) -> None:
        config = config or STFTConfig()
        self.n_fft = config.n_fft
        self.hop_length = config.hop_length
        self.window = torch.hann_window(window_length=self.n_fft, periodic=True)
        self.dim_f = config.dim_f

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        window = self.window.to(device=x.device)
        batch_dims = x.shape[:-2]
        c, t = x.shape[-2:]
        x = x.reshape([-1, t])
        x = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            center=True,
            return_complex=True,  # *2 is undesirable.
        )
        x = torch.view_as_real(x).to(dtype)
        x = x.permute([0, 3, 1, 2])
        x = x.reshape([*batch_dims, c, 2, -1, x.shape[-1]]).reshape(
            [*batch_dims, c * 2, -1, x.shape[-1]]
        )
        return x[..., : self.dim_f, :]

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        window = self.window.to(x.device)
        batch_dims = x.shape[:-3]
        c, f, t = x.shape[-3:]
        n = self.n_fft // 2 + 1
        f_pad = torch.zeros([*batch_dims, c, n - f, t]).to(x.device)
        x = torch.cat([x, f_pad], -2)
        x = x.reshape([*batch_dims, c // 2, 2, n, t]).reshape([-1, 2, n, t])
        x = x.permute([0, 2, 3, 1])
        x = x[..., 0] + x[..., 1] * 1.0j
        x = torch.istft(
            x, n_fft=self.n_fft, hop_length=self.hop_length, window=window, center=True
        )
        x = x.reshape([*batch_dims, 2, -1])
        return x.to(dtype)


class Upscale(torch.nn.Module):
    def __init__(self, in_c, out_c, scale):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.InstanceNorm2d(in_c, affine=True),
            torch.nn.GELU(),
            torch.nn.ConvTranspose2d(
                in_channels=in_c,
                out_channels=out_c,
                kernel_size=scale,
                stride=scale,
                bias=False,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Downscale(torch.nn.Module):
    def __init__(self, in_c, out_c, scale):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.InstanceNorm2d(in_c, affine=True),
            torch.nn.GELU(),
            torch.nn.Conv2d(
                in_channels=in_c,
                out_channels=out_c,
                kernel_size=scale,
                stride=scale,
                bias=False,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class MDX(torch.nn.Module):
    def __init__(
        self, in_channels, channels, num_blocks_per_scale, fps, bottleneck
    ) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList()
        for _ in range(num_blocks_per_scale):
            block = torch.nn.Module()

            block.tfc1 = torch.nn.Sequential(
                torch.nn.InstanceNorm2d(in_channels, affine=True),
                torch.nn.GELU(),
                torch.nn.Conv2d(in_channels, channels, 3, 1, 1, bias=False),
            )

            block.tdf = torch.nn.Sequential(
                torch.nn.InstanceNorm2d(channels, affine=True),
                torch.nn.GELU(),
                torch.nn.Linear(fps, fps // bottleneck, bias=False),
                torch.nn.InstanceNorm2d(channels, affine=True),
                torch.nn.GELU(),
                torch.nn.Linear(fps // bottleneck, fps, bias=False),
            )

            block.tfc2 = torch.nn.Sequential(
                torch.nn.InstanceNorm2d(channels, affine=True),
                torch.nn.GELU(),
                torch.nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            )

            block.shortcut = torch.nn.Conv2d(in_channels, channels, 1, 1, 0, bias=False)

            self.blocks.append(block)
            in_channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            s = block.shortcut(x)
            x = block.tfc1(x)
            x = x + block.tdf(x)
            x = block.tfc2(x)
            x = x + s
        return x


class Network(torch.nn.Module):
    def __init__(self, config: Optional[STFTConfig]) -> None:
        super().__init__()
        config = config or STFTConfig()
        self.config = config
        self.num_target_instruments = 2  # "vocal" and "others"
        self.num_subbands = config.num_subbands

        dim_c = self.num_subbands * config.num_audio_channels * 2
        num_blocks = config.num_scales
        scale = config.scale
        channels = config.num_channels
        growth = config.growth
        bottleneck = config.bottleneck_factor
        f = config.dim_f // self.num_subbands

        self.first_conv = torch.nn.Conv2d(dim_c, channels, 1, 1, 0, bias=False)

        self.encoder_blocks = torch.nn.ModuleList()
        for _ in range(num_blocks):
            block = torch.nn.Module()
            block.tfc_tdf = MDX(
                channels, channels, config.num_blocks_per_scale, f, bottleneck
            )
            block.downscale = Downscale(channels, channels + growth, scale)
            f = f // scale[1]
            channels += growth
            self.encoder_blocks.append(block)

        self.bottleneck_block = MDX(
            channels, channels, config.num_blocks_per_scale, f, bottleneck
        )

        self.decoder_blocks = torch.nn.ModuleList()
        for _ in range(num_blocks):
            block = torch.nn.Module()
            block.upscale = Upscale(channels, channels - growth, scale)
            channels -= growth
            f *= scale[1]
            block.tfc_tdf = MDX(
                2 * channels, channels, config.num_blocks_per_scale, f, bottleneck
            )
            self.decoder_blocks.append(block)

        self.final_conv = torch.nn.Sequential(
            torch.nn.Conv2d(channels + dim_c, channels, 1, 1, 0, bias=False),
            torch.nn.GELU(),
            torch.nn.Conv2d(
                channels, self.num_target_instruments * dim_c, 1, 1, 0, bias=False
            ),
        )

        self.stft = STFT(config)

    def cac2cws(self, x: torch.Tensor) -> torch.Tensor:
        k = self.num_subbands
        b, c, f, t = x.shape
        return x.reshape(b, c, k, f // k, t).reshape(b, c * k, f // k, t)

    def cws2cac(self, x: torch.Tensor) -> torch.Tensor:
        k = self.num_subbands
        b, c, f, t = x.shape
        return x.reshape(b, c // k, k, f, t).reshape(b, c // k, k * f, t)

    def __call__(self, path: str) -> Tuple[int, Dict[str, np.ndarray]]:
        """
        Split audio into vocals and accompaniment using the model specified.
        """
        first_param = next(self.parameters())
        dtype = first_param.dtype
        device = first_param.device

        torch.cuda.empty_cache()

        if isinstance(path, str):
            mix, sr = librosa.load(path, sr=self.config.sample_rate, mono=False)
        else:
            mix = path
            sr = self.config.sample_rate

        C = self.config.chunk_size
        fade_size = C // 10
        step = int(C // 4)
        border = C - step

        length_init = mix.shape[-1]

        if length_init > 2 * border and (border > 0):
            mix = np.pad(mix, ((0, 0), (border, border)), mode="reflect")
        mix = torch.tensor(mix, dtype=dtype)

        fadein = torch.linspace(0, 1, fade_size, dtype=dtype)
        fadeout = torch.linspace(1, 0, fade_size, dtype=dtype)

        window_start = torch.ones(C, dtype=dtype)
        window_middle = torch.ones(C, dtype=dtype)
        window_finish = torch.ones(C, dtype=dtype)
        window_start[-fade_size:] *= fadeout
        window_finish[:fade_size] *= fadein
        window_middle[-fade_size:] *= fadeout
        window_middle[:fade_size] *= fadein

        req_shape = (2,) + tuple(mix.shape)
        result = torch.zeros(req_shape, dtype=dtype)
        counter = torch.zeros(req_shape, dtype=dtype)
        i = 0
        batch_data = []
        batch_locations = []
        with torch.inference_mode():
            while i < mix.shape[1]:
                part = mix[:, i : i + C].to(device, dtype=dtype)
                length = part.shape[-1]
                if length < C:
                    if length > C // 2 + 1:
                        part = torch.nn.functional.pad(
                            input=part, pad=(0, C - length), mode="reflect"
                        )
                    else:
                        part = torch.nn.functional.pad(
                            input=part,
                            pad=(0, C - length, 0, 0),
                            mode="constant",
                            value=0,
                        )
                batch_data.append(part)
                batch_locations.append((i, length))
                i += step

                if len(batch_data) >= OPT_BATCH_SIZE or (i >= mix.shape[1]):
                    arr = torch.stack(batch_data, dim=0)
                    x = self.forward(arr)

                    window = window_middle
                    if i - step == 0:
                        window = window_start
                    elif i >= mix.shape[1]:
                        window = window_finish

                    for j in range(len(batch_locations)):
                        start, length = batch_locations[j]
                        result[..., start : start + length] += (
                            x[j][..., :length].cpu() * window[..., :length]
                        )
                        counter[..., start : start + length] += window[..., :length]
                    batch_data = []
                    batch_locations = []
            estimated_sources = result / counter
            torch.nan_to_num(estimated_sources)

            if length_init > 2 * border and (border > 0):
                estimated_sources = estimated_sources[..., border:-border]
        estimated_sources = estimated_sources.cpu().float().numpy()
        return sr, {
            "vocals": estimated_sources[0],
            "accompaniment": estimated_sources[1],
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stft(x)
        mix = x = self.cac2cws(x)
        first_conv_out = x = self.first_conv(x)
        x = x.transpose(-1, -2)
        encoder_outputs = []
        for block in self.encoder_blocks:
            x = block.tfc_tdf(x)
            encoder_outputs.append(x)
            x = block.downscale(x)
        x = self.bottleneck_block(x)
        for block in self.decoder_blocks:
            x = block.upscale(x)
            x = torch.cat([x, encoder_outputs.pop()], 1)
            x = block.tfc_tdf(x)
        x = x.transpose(-1, -2)
        x = x * first_conv_out  # reduce artifacts
        x = self.final_conv(torch.cat([mix, x], 1))
        x = self.cws2cac(x)
        b, _, f, t = x.shape
        x = x.reshape(b, self.num_target_instruments, -1, f, t)

        x = self.stft.inverse(x)
        return x


def test(batch: int = 4):
    from time import time

    from soundfile import write

    from . import load_mdx23c
    from ...config import get_config
    from ...utils import load_audio

    global OPT_BATCH_SIZE
    OPT_BATCH_SIZE = batch

    config = get_config()

    mdx = load_mdx23c("mdx23c.ckpt", STFTConfig(), config.device, config.dtype)
    w, sr = load_audio("test.mp4", 44100)
    if len(w.shape) == 1:
        w = np.asfortranarray([w, w])
    t0 = time()
    sr, out = mdx(w)
    print(f"Time: {time() - t0}")

    write("test_vocal.wav", out["vocals"].T, 44100, "FLOAT")
    write("test_accom.wav", out["accompaniment"].T, 44100, "FLOAT")


if __name__ == "__main__":
    test()
