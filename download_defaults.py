from pathlib import Path

import requests
from rich.progress import Progress
from rich.prompt import Confirm

from src.logger import progress_bar


def download(url: str, name: str, progress: Progress):
    if Path(name).exists():
        return
    resp = requests.get(url)
    length = int(resp.headers.get("Content-Length", "0"))
    it = resp.iter_content(chunk_size=1024 * 1024)  # 1MB chunks
    spl = length // (1024 * 1024)
    with open(name, "wb") as f:
        for chunk in progress.track(it, total=spl, description=f"Downloading {name}"):
            f.write(chunk)


if __name__ == "__main__":
    download_train = Confirm.ask("Also download models for training?")
    total = 4 + (2 if download_train else 0)
    with progress_bar as progress:
        task = progress.add_task("Downloading models", total=total)
        download(
            "https://huggingface.co/lj1995/VoiceConversionWebUI/resikve/main/hubert_base.pt",
            "hubert_base.pt",
            progress,
        )
        progress.update(task, advance=1)
        download(
            "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt",
            "rmvpe.pt",
            progress,
        )
        progress.update(task, advance=1)
        download(
            "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.0/model_vocals_mdx23c_sdr_10.17.ckpt",
            "mdx23c.ckpt",
            progress,
        )
        progress.update(task, advance=1)
        download(
            "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/uvr5_weights/VR-DeEchoNormal.pth",
            "VR-DeEchoNormal.pth",
            progress,
        )
        progress.update(task, advance=1)
        if download_train:
            download(
                "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/f0D40k.pth",
                "weights/D_f0D40k.pth",
                progress,
            )
            progress.update(task, advance=1)
            download(
                "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/f0G40k.pth",
                "weights/G_f0G40k.pth",
                progress,
            )
            progress.update(task, advance=1)
