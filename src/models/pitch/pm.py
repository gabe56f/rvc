import torch
import parselmouth
from scipy import signal
import numpy as np


class Parselmouth:
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
        timestep = hop_length / sr

        x_n: np.ndarray = x.cpu().float().numpy().astype("double")
        f0 = (
            parselmouth.Sound(x_n, sr)
            .to_pitch_ac(
                time_step=timestep,
                voicing_threshold=0.6,
                pitch_floor=f0_min,
                pitch_ceiling=f0_max,
            )
            .selected_array["frequency"]
        )
        pad_size = (pred_len - len(f0) + 1) // 2
        if pad_size > 0 or pred_len - len(f0) - pad_size > 0:
            f0 = np.pad(
                f0, ((pad_size, pred_len - len(f0) - pad_size)), mode="constant"
            )
        f0 = signal.medfilt(f0, kernel_size=3)
        return torch.from_numpy(f0).to(self.device, self.dtype)
