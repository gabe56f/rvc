from pathlib import Path
from typing import Optional, Tuple
import datetime
from logging import Logger
from time import time
import glob
import os

import torch


class EpochRecorder:
    def __init__(self):
        self.last_time = time()

    def record(self) -> str:
        current_time = time()
        elapsed_time = current_time - self.last_time
        self.last_time = current_time
        elapsed_time_str = str(datetime.timedelta(seconds=elapsed_time))
        color = (
            "green" if elapsed_time < 5 else "yellow" if elapsed_time < 10 else "red"
        )
        return f"[{color}]{elapsed_time_str}[/]"


def load_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    logger: Logger,
    optimizer: Optional[torch.optim.Optimizer] = None,
    load_opt: bool = False,
) -> Tuple[torch.nn.Module, Optional[torch.optim.Optimizer], float, int]:
    if not os.path.exists(checkpoint_path):
        raise ValueError(f"Checkpoint {checkpoint_path} does not exist")

    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")

    saved_state_dict = checkpoint_dict["model"]
    if hasattr(model, "module"):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    new_state_dict = {}
    for k, v in state_dict.items():
        try:
            new_state_dict[k] = saved_state_dict[k]
            if saved_state_dict[k].shape != state_dict[k].shape:
                logger.warn(
                    f"Tensor {k} shape mismatch: {state_dict[k].shape} != {saved_state_dict[k].shape}"
                )  #
                raise KeyError
        except KeyError:
            logger.info(f"{k} is not in the checkpoint")
            new_state_dict[k] = v
    if hasattr(model, "module"):
        model.module.load_state_dict(new_state_dict, strict=False)
    else:
        model.load_state_dict(new_state_dict, strict=False)
    logger.info("Loaded model weights")

    iteration = checkpoint_dict["iteration"]
    learning_rate = checkpoint_dict["learning_rate"]
    if optimizer is not None and load_opt:
        optimizer.load_state_dict(checkpoint_dict["optimizer"])
    logger.info(f"Loaded checkpoint '{checkpoint_path}' (epoch {iteration})")
    return model, optimizer, learning_rate, iteration


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    logger: Logger,
    learning_rate: float,
    iteration: int,
    checkpoint_path: Path,
) -> Path:
    logger.info(
        f"Saving model and optimizer state at epoch {iteration} to {checkpoint_path}"
    )
    if hasattr(model, "module"):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    torch.save(
        {
            "model": state_dict,
            "iteration": iteration,
            "optimizer": optimizer.state_dict(),
            "learning_rate": learning_rate,
        },
        checkpoint_path,
    )
    return checkpoint_path


def latest_checkpoint_path(dir_path: Path, regex: str = "G_*.pth") -> str:
    f_list = glob.glob(os.path.join(dir_path, regex))
    f_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
    x = f_list[-1]
    # print(x)
    return x


def feature_loss(fmap_r: torch.Tensor, fmap_g: torch.Tensor) -> float:
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            rl = rl.float().detach()
            gl = gl.float()
            loss += torch.mean(torch.abs(rl - gl))
    return loss * 2


def discriminator_loss(
    disc_real_outputs: torch.Tensor, disc_generated_outputs: torch.Tensor
):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        dr = dr.float()
        dg = dg.float()
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg**2)
        loss += r_loss + g_loss
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(disc_outputs: torch.Tensor):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        dg = dg.float()
        gen_loss = torch.mean((1 - dg) ** 2)
        gen_losses.append(gen_loss)
        loss += gen_loss

    return loss, gen_losses


def kl_loss(
    z_p: torch.Tensor,
    logs_q: torch.Tensor,
    m_p: torch.Tensor,
    logs_p: torch.Tensor,
    z_mask: torch.Tensor,
):
    z_p = z_p.float()
    logs_q = logs_q.float()
    m_p = m_p.float()
    logs_p = logs_p.float()
    z_mask = z_mask.float()

    kl = logs_p - logs_q - 0.5
    kl += 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2.0 * logs_p)
    kl = torch.sum(kl * z_mask)
    kl_loss = kl / torch.sum(z_mask)
    return kl_loss
