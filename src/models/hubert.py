from typing import TYPE_CHECKING

import torch
from fairseq import checkpoint_utils

if TYPE_CHECKING:
    from .rvc import RVCConfig


def load_hubert(
    hubert_file: str, rvc_config: "RVCConfig", return_cfg: bool = False
) -> torch.nn.Module:
    hubert, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        [hubert_file],
        suffix="",
    )
    hubert = hubert[0]
    if rvc_config is not None:
        hubert = hubert.to(rvc_config.device, rvc_config.dtype)
    hubert.eval()
    if return_cfg:
        return hubert, saved_cfg, task
    return hubert
