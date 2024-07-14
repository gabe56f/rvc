from typing import Dict, List, Union, Tuple, TYPE_CHECKING

import torch
import numpy as np
from scipy import signal
import librosa

from .base import PitchExtractor
from .crepe import Crepe
from .dio import Dio
from .fcpe import FCPE
from .harvest import Harvest
from .pm import Parselmouth
from .pyin import Pyin
from .rmvpe import RMVPE

if TYPE_CHECKING:
    from ..rvc import RVCConfig

_bh, _ah = signal.butter(N=5, Wn=48, btype="high", fs=16000)


_pitch_extractors: Dict[str, PitchExtractor] = {}


def _get_pitch_extractors(rvc_config: "RVCConfig"):
    global _pitch_extractors
    if len(_pitch_extractors) == 0:
        kwargs = {
            "device": rvc_config.device,
            "dtype": rvc_config.dtype,
        }
        _pitch_extractors = {
            "rmvpe": RMVPE(**kwargs),
            "fcpe": FCPE(**kwargs),
            "harvest": Harvest(**kwargs),
            "crepe": Crepe(**kwargs),
            "dio": Dio(**kwargs),
            "pyin": Pyin(**kwargs),
            "parselmouth": Parselmouth(**kwargs),
        }
    return _pitch_extractors


def compute_pitch_from_audio(
    audio: np.ndarray,
    transposition: int = 0,
    sr: int = 16000,
    hop_length: int = 160,
    rvc_config: "RVCConfig" = None,
    extractors: Union[List[str], str] = "rmvpe",
    skip_preprocess: bool = False,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], np.ndarray]:
    """Output is ((pitch, pitch_melodic), extended_audio)"""
    if rvc_config is None:
        from ..rvc import RVCConfig

        rvc_config = RVCConfig()
    if isinstance(extractors, str):
        if "," in extractors:
            extractors = extractors.split(",")
        else:
            extractors = [extractors]

    pitch_extractors = _get_pitch_extractors(rvc_config)
    extractors = list(
        filter(
            lambda x: x is not None,
            [pitch_extractors.get(e.lower(), None) for e in extractors],
        )
    )
    if len(extractors) == 0:
        raise ValueError("No valid pitch extractor found")
    # print(f"Using extractors: {extractors}")

    if skip_preprocess:
        audio_out = audio.copy()
        audio = torch.from_numpy(audio).to(rvc_config.device, rvc_config.dtype)
    else:
        t_pad = sr * rvc_config.x_pad
        audio = signal.filtfilt(_bh, _ah, audio)
        audio = librosa.to_mono(audio)
        audio_out = audio.copy()
        audio_pad = np.pad(audio, (t_pad, t_pad), mode="reflect")
        audio = torch.from_numpy(audio_pad).to(rvc_config.device, rvc_config.dtype)
    pred_len = audio.shape[0] // hop_length
    f0_min = 50
    f0_max = 1100
    f0_mel_min = 1127 * np.log(1 + f0_min / 700)
    f0_mel_max = 1127 * np.log(1 + f0_max / 700)

    kwargs = {
        "f0_min": 50,
        "f0_max": 1100,
        "pred_len": pred_len,
        "hop_length": hop_length,
        "batch_size": rvc_config.batch_size,
        "sr": sr,
        "variant": "full",
    }
    pitches: List[torch.Tensor] = []
    for extractor in extractors:
        pitches.append(extractor(audio, **kwargs))

    if len(pitches) == 1:
        f0: torch.Tensor = pitches[0]
    else:
        f0: torch.Tensor = torch.nanmedian(torch.stack(pitches), dim=0).values
    f0 *= pow(2, transposition / 12.0)  # 12 keys

    pitchf = f0.clone().to(rvc_config.device, rvc_config.dtype)
    f0_mel = 1127 * torch.log(1 + f0 / 700)
    f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (
        f0_mel_max - f0_mel_min
    ) + 1
    f0_mel[f0_mel <= 1] = 1
    f0_mel[f0_mel > 255] = 255
    pitch = torch.round(f0_mel).to(torch.long)
    return ((pitch.unsqueeze(0), pitchf.unsqueeze(0)), audio_out)
