import torch


class Melodia:
    def __init__(self, device="cuda", dtype=torch.bfloat16) -> None:
        self.device = device
        self.dtype = dtype
        self.extractor = None  # lazy init

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
        if self.extractor is None:
            from essentia.standard import PredominantPitchMelodia

            self.extractor = PredominantPitchMelodia(
                frameSize=2048,
                hopSize=hop_length,
                sampleRate=sr,
                minFrequency=f0_min,
                maxFrequency=f0_max,
                voicingTolerance=0.05,
                guessUnvoiced=True,
                voiceVibrato=True,
            )
        if x.ndim == 2:
            x = x.mean(0)
        f0, _ = self.extractor(x.cpu().float().numpy())
        f0: torch.Tensor = torch.from_numpy(f0)
        return f0.to(self.device, self.dtype)
