import torch


class FCPE:
    def __init__(self, device="cuda", dtype=torch.bfloat16) -> None:
        self.device = device
        self.dtype = dtype
        self.fcpe = None  # lazy init
        # self.fcpe = torchfcpe.spawn_bundled_infer_model(device).to(dtype)

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
        if self.fcpe is None:
            import torchfcpe

            self.fcpe = torchfcpe.spawn_bundled_infer_model(self.device).to(self.dtype)
        with torch.autocast(torch.device(self.device).type, self.dtype):
            x = x.clamp(-1, 1).unsqueeze(0)
            f0: torch.Tensor = self.fcpe.infer(x, sr=sr, f0_min=f0_min, f0_max=f0_max)
            f0 = f0.squeeze()
        return f0.to(self.device, self.dtype)
