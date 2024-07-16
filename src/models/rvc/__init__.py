from dataclasses import dataclass, field
from typing import Tuple, Union, Optional
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import librosa
import faiss
from faiss import Index

from .models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs768NSFsid,
    MultiPeriodDiscriminator,  # noqa
    MultiPeriodDiscriminatorV2,  # noqa
)

SynthesizerType = Union[
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs256NSFsid,
]


@dataclass
class RVCConfig:
    # optimized for 6G vram
    x_pad: int = 3
    x_query: int = 10
    x_center: int = 60
    x_max: int = 65
    batch_size: int = 512
    dtype: torch.dtype = field(default=torch.bfloat16)
    device: str = "cuda"

    def optimize_for_device(self, device: str = "cuda"):
        # TODO: Implement this
        pass


def _change_rms(
    data1: torch.Tensor, sr1: int, data2: torch.Tensor, sr2: int, rate: float
) -> torch.Tensor:  # 1是输入音频，2是输出音频,rate是2的占比
    dtype = data1.dtype
    device = data1.device
    data1 = data1.cpu().float().numpy()
    data2 = data2.cpu().float().numpy()
    rms1 = librosa.feature.rms(y=data1, frame_length=sr1 // 2 * 2, hop_length=sr1 // 2)
    rms2 = librosa.feature.rms(y=data2, frame_length=sr2 // 2 * 2, hop_length=sr2 // 2)
    rms1 = torch.from_numpy(rms1)
    rms1 = F.interpolate(
        rms1.unsqueeze(0), size=data2.shape[0], mode="linear"
    ).squeeze()
    rms2 = torch.from_numpy(rms2)
    rms2 = F.interpolate(
        rms2.unsqueeze(0), size=data2.shape[0], mode="linear"
    ).squeeze()
    rms2 = torch.max(rms2, torch.zeros_like(rms2) + 1e-6)
    data2 = torch.from_numpy(data2).to(dtype)
    data2 *= (
        torch.pow(rms1, torch.tensor(1 - rate))
        * torch.pow(rms2, torch.tensor(rate - 1))
    ).to(dtype)
    return data2.to(device)


def load_rvc_model(
    file: Path, index_file: Optional[Path], rvc_config: RVCConfig = RVCConfig()
) -> Tuple[SynthesizerType, Index, float, str, int, RVCConfig]:
    """Returns: (net_g, version, target_sr)"""
    if file.suffix == ".pth":
        ckpt = torch.load(file, map_location=rvc_config.device)
    else:
        raise ValueError(f"{file.suffix} not yet supported.")
    target_sr = ckpt["config"][-1]
    try:
        ckpt["config"][-3] = ckpt["weight"]["emb_g.weight"].shape[0]  # n_spk
    except:  # noqa
        pass
    f0 = ckpt.get("f0", 1)
    if f0 == 0:
        raise ValueError("f0-less models not supported!")
    version = ckpt.get("version", "v1")
    if version == "v1":
        net_g = SynthesizerTrnMs256NSFsid(*ckpt["config"])
    elif version == "v2":
        net_g = SynthesizerTrnMs768NSFsid(*ckpt["config"])
    else:
        raise ValueError(f"Unknown version: {version}")
    del net_g.enc_q  # useless for inference
    net_g.load_state_dict(ckpt["weight"], strict=False)
    net_g.eval()
    net_g = net_g.to(rvc_config.device, rvc_config.dtype)

    if index_file is None:
        index = None
        big_npy = None
    else:
        index: Index = faiss.read_index(index_file.as_posix())
        big_npy: float = index.reconstruct_n(0, index.ntotal)

    return net_g, index, big_npy, version, target_sr, rvc_config


def infer(
    f0_output: Tuple[Tuple[torch.Tensor, torch.Tensor], np.ndarray],
    hubert: torch.nn.Module,
    net_g: SynthesizerType,
    index: Index,
    big_npy: float,
    rvc_config: RVCConfig,
    target_sr: int,
    rms_mix_rate: float = 0.25,
    index_rate: float = 1.0,
    protect: float = 0.5,
    version: str = "v2",
    speaker_id: int = 0,
    sr: int = 16000,
    hop_length: int = 160,
) -> np.ndarray:
    (pitch, pitchf), audio = f0_output

    t_pad = sr * rvc_config.x_pad
    t_pad_target = target_sr * rvc_config.x_pad
    t_pad2 = t_pad * 2
    t_query = sr * rvc_config.x_query
    t_center = sr * rvc_config.x_center
    t_max = sr * rvc_config.x_max

    audio_pad = np.pad(audio.copy(), (hop_length // 2, hop_length // 2), mode="reflect")

    opt_ts = []
    if audio.shape[0] > t_max:
        audio_sum = np.zeros_like(audio)
        for i in range(hop_length):
            audio_sum += audio_pad[i : i - hop_length]
        for t in range(t_center, audio.shape[0], t_center):
            opt_ts.append(
                t
                - t_query
                + np.where(
                    np.abs(audio_sum[t - t_query : t + t_query])
                    == np.abs(audio_sum[t - t_query : t + t_query]).min()
                )[0][0]
            )
    s = 0
    audio_opt = []
    audio = np.pad(audio, (t_pad, t_pad), "reflect")
    audio = torch.from_numpy(audio).to(rvc_config.device, rvc_config.dtype)
    sid = torch.tensor(speaker_id, device=audio.device).unsqueeze(0).long()

    def infer(
        audio0: torch.Tensor,
        pitch: torch.Tensor,
        pitchf: torch.Tensor,
        sid: torch.Tensor,
    ):
        feats = audio0
        if feats.dim() == 2:
            feats = feats.mean(-1)
        feats = feats.view(1, -1)
        padding_mask = torch.BoolTensor(feats.shape).to(rvc_config.device).fill_(False)
        kwargs = {
            "source": feats,
            "padding_mask": padding_mask,
            "output_layer": 9 if version == "v1" else 12,
        }
        with torch.no_grad():
            logits = hubert.extract_features(**kwargs)
            logits = logits[0]
            feats: torch.Tensor = (
                hubert.final_proj(logits) if version == "v1" else logits
            )

        if protect < 0.5:
            feats0: torch.Tensor = feats.clone()

        if index is not None and big_npy is not None and index_rate != 0:
            npy = feats[0].cpu().float().numpy()
            score, ix = index.search(npy, k=8)
            weight: np.ndarray = np.square(1 / score)
            weight /= weight.sum(axis=1, keepdims=True)
            npy = np.sum(big_npy[ix] * np.expand_dims(weight, axis=2), axis=1)
            feats = (
                torch.from_numpy(npy)
                .unsqueeze(0)
                .to(rvc_config.device, rvc_config.dtype)
                * index_rate
                + (1 - index_rate) * feats
            )
        feats: torch.Tensor = F.interpolate(
            feats.permute(0, 2, 1), scale_factor=2
        ).permute(0, 2, 1)
        if protect < 0.5:
            feats0: torch.Tensor = F.interpolate(
                feats0.permute(0, 2, 1), scale_factor=2
            ).permute(0, 2, 1)
        pred_len = audio0.shape[0] // hop_length
        if feats.shape[1] < pred_len:
            pred_len = feats.shape[1]
            pitch = pitch[:, :pred_len]
            pitchf = pitchf[:, :pred_len]

        if protect < 0.5:
            pitchff = pitchf.clone()
            pitchff[pitchf > 0] = 1
            pitchff[pitchf < 1] = protect
            pitchff = pitchff.unsqueeze(-1)
            feats = feats * pitchff + feats0 * (1 - pitchff)
            feats = feats.to(feats0.device, feats0.dtype)
        pred_len = torch.tensor([pred_len], device=rvc_config.device).long()
        with torch.no_grad():
            audio1 = net_g.infer(feats, pred_len, pitch, pitchf, sid)[0][0, 0].data
        del feats, pred_len, padding_mask
        if "cuda" in rvc_config.device:
            torch.cuda.empty_cache()
        return audio1.to(rvc_config.device)

    t = None
    for t in opt_ts:
        t = t // hop_length * hop_length
        audio0 = audio[s : t + t_pad2 + hop_length].to(rvc_config.device)
        pitch0 = pitch[:, s // hop_length : (t + t_pad2) // hop_length]
        pitchf0 = pitchf[:, s // hop_length : (t + t_pad2) // hop_length]
        audio_opt.append(
            infer(audio0, pitch0, pitchf0, sid)[t_pad_target:-t_pad_target]
        )
        s = t
    pitch0 = pitch[:, t // hop_length :] if t is not None else pitch
    pitchf0 = pitchf[:, t // hop_length :] if t is not None else pitchf
    audio_opt.append(infer(audio[t:], pitch0, pitchf0, sid)[t_pad_target:-t_pad_target])
    audio_opt = torch.cat(audio_opt)

    if rms_mix_rate != 1:
        audio_opt = _change_rms(audio, sr, audio_opt, target_sr, rms_mix_rate)
    audio_max = torch.abs(audio_opt).max() / 0.99
    max_int16 = 32768
    if audio_max > 1:
        max_int16 /= audio_max.item()
    audio_opt = (audio_opt.cpu().float().numpy() * max_int16).astype(np.int16)
    del pitch, pitchf, sid
    if "cuda" in rvc_config.device:
        torch.cuda.empty_cache()
    return audio_opt, target_sr
