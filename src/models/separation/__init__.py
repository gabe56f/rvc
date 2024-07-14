from typing import Optional, Tuple
from pathlib import Path

import torch
import requests

from .mdx23c import Network as MDX23C, STFTConfig as MDX23CConfig
from .uvr import UVR, UVRParameters as UVRConfig  # noqa: F401


def load_mdx23c(
    file: str,
    mdx_config: Optional[MDX23CConfig] = None,
    device: str = "cuda",
    dtype=torch.bfloat16,
) -> MDX23C:
    default_mdx_config = MDX23CConfig()
    if mdx_config is None:
        mdx_config = default_mdx_config
    net = MDX23C(mdx_config)

    state_dict = torch.load(file)
    net.load_state_dict(state_dict)

    net = net.to(device, dtype)
    return net


def load_uvr(file: str, device: str = "cuda", dtype=torch.bfloat16) -> UVR:
    try:
        uvr = UVR(10, file, device, dtype, True)
    except Exception:
        uvr = UVR(10, file, device, dtype, True)
    return uvr


def load_preset(
    mdx_config: Optional[MDX23CConfig] = None,
    device: str = "cuda",
    dtype=torch.bfloat16,
) -> Tuple[MDX23C, UVR]:
    if not (mdx_file := Path("mdx23c.ckpt")).exists():
        url = "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.0/model_vocals_mdx23c_sdr_10.17.ckpt"
        response = requests.get(url, stream=True)
        with open(mdx_file, "wb") as handle:
            for data in response.iter_content():
                handle.write(data)
    if not (deecho_file := Path("VR-DeEchoNormal.pth")).exists():
        url = "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/uvr5_weights/VR-DeEchoNormal.pth"
        response = requests.get(url, stream=True)
        with open(deecho_file, "wb") as handle:
            for data in response.iter_content():
                handle.write(data)
    mdx = load_mdx23c(mdx_file.as_posix(), mdx_config, device, dtype)
    uvr = load_uvr(deecho_file.as_posix(), device, dtype)
    return mdx, uvr
