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
from .melodia import Melodia
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
            "melodia": Melodia(**kwargs),
        }
    return _pitch_extractors


def _preprocess(audio: np.ndarray, sr: int, config: "RVCConfig", skip: bool = False):
    if skip:
        audio_out = audio.copy()
    else:
        t_pad = sr * config.x_pad
        audio = signal.filtfilt(_bh, _ah, audio)
        audio = librosa.to_mono(audio)
        audio_out = audio.copy()
        audio = np.pad(audio, (t_pad, t_pad), mode="reflect")
    audio = torch.from_numpy(audio).to(config.device, config.dtype)
    return audio, audio_out


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

    multiple = isinstance(audio, list)
    if multiple:
        audio_list = []
        audio_out_list = []

        for a in audio:
            au, au_o = _preprocess(a, sr, rvc_config, skip_preprocess)
            audio_list.append(au)
            audio_out_list.append(au_o)

        audio = torch.concatenate(audio_list, dim=0)
        audio_out = np.concatenate(audio_out_list, axis=0)
    else:
        audio, audio_out = _preprocess(audio, sr, rvc_config, skip_preprocess)
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

    # print(f"Pitch shape: {f0.shape}")

    pitchf = f0.clone().to(rvc_config.device, rvc_config.dtype)
    f0_mel = 1127 * torch.log(1 + f0 / 700)
    f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (
        f0_mel_max - f0_mel_min
    ) + 1
    f0_mel[f0_mel <= 1] = 1
    f0_mel[f0_mel > 255] = 255
    pitch = torch.round(f0_mel).to(torch.long)
    if multiple:
        return ((pitch, pitchf), audio_out)
    return ((pitch.unsqueeze(0), pitchf.unsqueeze(0)), audio_out)


def main():
    from time import time
    from ...utils import load_audio

    wav, _ = load_audio("test.mp4", 16000)

    t0 = time()
    compute_pitch_from_audio(wav, extractors="rmvpe")
    print(f"Init time: {time() - t0}s")

    t = []

    for _ in range(10):
        t0 = time()
        compute_pitch_from_audio(wav, extractors="rmvpe")
        t.append(time() - t0)

    print(f"Time: {np.mean(t)}s")


if __name__ == "__main__":
    main()
