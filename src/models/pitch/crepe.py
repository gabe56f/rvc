import torch
import torchcrepe
import numpy as np


class Crepe:
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
        with torch.autocast(torch.device(self.device).type, self.dtype):
            x = x[None].to(self.device)
            f0, pd = torchcrepe.predict(
                audio=x,
                sample_rate=sr,
                hop_length=hop_length,
                fmin=f0_min,
                fmax=f0_max,
                model=variant,
                batch_size=batch_size,
                device=self.device,
                return_periodicity=True,
                pad=True,
            )
        pd = torchcrepe.filter.median(pd, 3)
        f0 = torchcrepe.filter.mean(f0, 3)
        f0[pd < 0.1] = 0

        # resize to pred_len, have to do this in numpy since torch doesnt support 1D interp
        source = f0.squeeze(0).cpu().float().numpy()
        target = np.interp(
            np.arange(0, len(source) * pred_len, len(source)) / pred_len,
            np.arange(0, len(source)),
            source,
        )

        return torch.from_numpy(target).to(self.device, self.dtype)
