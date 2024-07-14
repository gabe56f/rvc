import torch
from librosa import pyin, to_mono
import numpy as np
from scipy import signal


class Pyin:
    def __init__(self, device="cuda", dtype=torch.bfloat16):
        self.device = device
        self.dtype = dtype

    def __call__(
        self,
        x: torch.Tensor,
        f0_min: float,
        f0_max: float,
        pred_len: int,
        hop_length: int = 160,
        batch_size: int = 512,
        sr: int = 44100,
        variant: str = "full",
    ) -> torch.Tensor:
        x_n: np.ndarray = x.cpu().float().numpy()
        x_n = to_mono(x_n)
        f0, _, _ = pyin(
            x_n,
            sr=sr,
            fmax=f0_max,
            fmin=f0_min,
        )
        f0 = signal.medfilt(f0, kernel_size=3)
        f0 = np.interp(
            np.arange(0, len(f0) * pred_len, len(f0)) / pred_len,
            np.arange(0, len(f0)),
            f0,
        )
        return torch.from_numpy(f0).to(self.device, self.dtype)
