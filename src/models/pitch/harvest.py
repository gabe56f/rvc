import torch
import pyworld
import numpy as np
from scipy import signal


class Harvest:
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
        x_n: np.ndarray = x.cpu().float().numpy().astype("double")
        f0, t = pyworld.harvest(
            x_n,
            fs=sr,
            f0_ceil=f0_max,
            f0_floor=f0_min,
            frame_period=10,
        )
        f0 = pyworld.stonemask(x_n, f0, t, sr)
        f0 = signal.medfilt(f0, kernel_size=3)
        return torch.from_numpy(f0[1:]).to(self.device, self.dtype)  # drop first frame
