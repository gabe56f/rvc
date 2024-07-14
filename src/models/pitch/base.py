from typing import Protocol

import torch


class PitchExtractor(Protocol):
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
    ) -> torch.Tensor: ...
