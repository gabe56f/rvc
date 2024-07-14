import torch


class Config:
    device: str = "cuda:0"
    dtype: torch.dtype = torch.bfloat16
    batch_size: int = 512


conf = None


def get_config() -> Config:
    global conf
    if conf is None:
        conf = Config()
    return conf
