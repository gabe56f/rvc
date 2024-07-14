from src.logger import progress_bar

import logging
from collections import OrderedDict
from dataclasses import dataclass, asdict, field
from typing import (
    Any,
    Dict,
    Protocol,
    Type,
    Tuple,
    Literal,
    List,
    Optional,
    TYPE_CHECKING,
    Union,
)
from pathlib import Path
from random import shuffle, randint
import os
from warnings import simplefilter


import cpuinfo
from rich.progress import Progress
from rich.prompt import Prompt, Confirm
import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torchaudio import functional as Fa
import numpy as np
from scipy import signal
from scipy.io import wavfile
from sklearn.cluster import MiniBatchKMeans
import faiss

from src.models.rvc import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs768NSFsid,
    MultiPeriodDiscriminator,
    MultiPeriodDiscriminatorV2,
    RVCConfig,
)
from src.models.hubert import load_hubert
from src.models.pitch import compute_pitch_from_audio
from src.utils import (
    load_checkpoint,
    save_checkpoint,
    latest_checkpoint_path,
    feature_loss,
    discriminator_loss,
    generator_loss,
    kl_loss,
    TextAudioLoaderMultiNSFsid,
    TextAudioCollateMultiNSFsid,
    DistributedBucketSampler,
    torch_spec_to_mel,
    torch_mel_spectrogram,
    load_audio,
    get_rms,
)

if TYPE_CHECKING:
    from lomo_optim import AdaLomo, Lomo
    from came_pytorch import CAME
    import lpmm
    from sophia import SophiaG
    from adan import Adan


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = str(randint(20000, 55555))

USE_WANDB = True

gt = "gtwaves/"
sixteenk = "16kwaves/"
features = "features/"
f0 = "f0/"
f0coarse = "f0nsf/"

hubert = "hubert_base.pt"
assert Path(
    hubert
).exists(), "Please download the pretrained Hubert model from Hugging Face"
hubert, saved_cfg, task = load_hubert(hubert, None, return_cfg=True)

simplefilter("ignore")

global_step = 0

if USE_WANDB:
    import wandb

    USE_WANDB = wandb.login()


OptimizerList = Literal[
    "adam",
    "adamw",
    "adamw_8bit",
    "adamw_4bit",
    "adamw_apex",
    "adan",
    "sophia",
    "lomo",
    "adalomo",
    "came",
]
OptimizerType = Union[
    torch.optim.Optimizer,
    "AdaLomo",
    "Lomo",
    "CAME",
    "lpmm.optim.AdamW",
    "SophiaG",
    "Adan",
]


@dataclass
class TrainParameters:
    log_interval: int = 1
    seed: int = 1234
    epochs: int = 3000
    learning_rate: float = 1e-4
    betas: List[float] = field(default_factory=lambda: [0.8, 0.99])
    eps: float = 1e-9
    batch_size: int = 36
    dtype: Literal["float32", "bfloat16", "float16"] = "bfloat16"
    lr_decay: float = 0.999875
    segment_size: int = 12800
    init_lr_ratio: float = 1
    warmup_epochs: int = 0
    c_mel: int = 45
    c_kl: float = 1.0
    optimizer: OptimizerList = "adamw"
    scheduler: Literal["exponential", "constant", "cosine"] = "exponential"


@dataclass
class DataParameters:
    max_wav_value: float = 32768
    sampling_rate: int = 40000
    filter_length: int = 2048
    hop_length: int = 400
    win_length: int = 2048
    n_mel_channels: int = 125
    mel_fmin: float = 0.0
    mel_fmax: Optional[float] = None


@dataclass
class ModelParameters:
    inter_channels: int = 192
    hidden_channels: int = 192
    filter_channels: int = 768
    n_heads: int = 2
    n_layers: int = 6
    kernel_size: int = 3
    p_dropout: int = 0
    resblock: str = "1"
    resblock_kernel_sizes: List[int] = field(default_factory=lambda: [3, 7, 11])
    resblock_dilation_sizes: List[List[int]] = field(
        default_factory=lambda: [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    )
    upsample_rates: List[int] = field(default_factory=lambda: [10, 10, 2, 2])
    upsample_initial_channel: int = 512
    upsample_kernel_sizes: List[int] = field(default_factory=lambda: [16, 16, 4, 4])
    use_spectral_norm: bool = False
    gin_channels: int = 256
    spk_embed_dim: int = 109


@dataclass
class TrainingParameters:
    train = TrainParameters()
    data = DataParameters()
    model = ModelParameters()


@dataclass
class InputParameters:
    training_files: Path
    model_files: Path
    used_device: torch.device
    save_every: int = 200
    latest_only: bool = False
    pitch_extractor: str = "rmvpe"
    cache_data: bool = True
    pretrain_g: str = ""
    pretrain_d: str = ""
    version: str = "v2"
    dataloader_workers: int = 4

    def get_rvc_config(self, hyperparameters: TrainingParameters) -> RVCConfig:
        dtype = getattr(torch, hyperparameters.train.dtype)

        return RVCConfig(
            device=self.used_device,
            dtype=dtype,
        )


class TrainingLogger(Protocol):
    def log(self, log: Dict[str, Any]) -> None: ...


class CLITrainingLogger:
    last_params = {}

    def log(self, log: Dict[str, Any]) -> None:
        out = ""
        color_params = {}
        for k, v in log.items():
            if k in self.last_params:
                if self.last_params[k] == v:
                    color_params[k] = "yellow"
                elif self.last_params[k] < v:
                    color_params[k] = "green"
                else:
                    color_params[k] = "red"
                continue
            if isinstance(v, (int, float)):
                self.last_params[k] = v
                color_params[k] = "green"

        log_len = len(log)
        for i, (k, v) in enumerate(log.items()):
            out += f"[bold]{k}[/]: [{color_params[k]}]{str(v)}[/]"
            if i < log_len - 1:
                out += " | "
        logger.info(out, extra={"markup": True})


class Slicer:
    def __init__(
        self,
        sr: int,
        silence_threshold: float = -40,
        min_length: int = 5000,
        min_interval: int = 300,
        hop_length: int = 20,
        max_sil_kept: int = 5000,
    ):
        min_interval = sr * min_interval / 1000
        self.threshold = 10 ** (silence_threshold / 20.0)
        self.hop_length = round(sr * hop_length / 1000)
        self.win_size = min(round(min_interval), 4 * self.hop_length)
        self.min_length = round(sr * min_length / 1000 / self.hop_length)
        self.min_interval = round(min_interval / self.hop_length)
        self.max_sil_kept = round(sr * max_sil_kept / 1000 / self.hop_length)

    def _apply_slice(self, waveform, begin, end):
        if len(waveform.shape) > 1:
            return waveform[
                :,
                begin * self.hop_length : min(waveform.shape[1], end * self.hop_length),
            ]
        else:
            return waveform[
                begin * self.hop_length : min(waveform.shape[0], end * self.hop_length)
            ]

    def __call__(self, waveform):
        if len(waveform.shape) > 1:
            samples = waveform.mean(axis=0)
        else:
            samples = waveform
        if samples.shape[0] <= self.min_length:
            return [waveform]

        rms_list = get_rms(
            y=samples, frame_length=self.win_size, hop_length=self.hop_length
        ).squeeze(0)
        sil_tags = []
        silence_start = None
        clip_start = 0
        for i, rms in enumerate(rms_list):
            if rms < self.threshold:
                if silence_start is None:
                    silence_start = i
                continue
            if silence_start is None:
                continue
            is_leading_silence = silence_start == 0 and i > self.max_sil_kept
            need_slice_middle = (
                i - silence_start >= self.min_interval
                and i - clip_start >= self.min_length
            )
            if not is_leading_silence and not need_slice_middle:
                silence_start = None
                continue
            if i - silence_start <= self.max_sil_kept:
                pos = torch.argmin(rms_list[silence_start : i + 1]) + silence_start
                if silence_start == 0:
                    sil_tags.append((0, pos))
                else:
                    sil_tags.append((pos, pos))
                clip_start = pos
            elif i - silence_start <= self.max_sil_kept * 2:
                pos = (
                    torch.argmin(
                        rms_list[
                            i
                            - self.max_sil_kept : silence_start
                            + self.max_sil_kept
                            + 1
                        ]
                    )
                    + i
                    - self.max_sil_kept
                )
                pos_l = (
                    torch.argmin(
                        rms_list[silence_start : silence_start + self.max_sil_kept + 1]
                    )
                    + silence_start
                )
                pos_r = (
                    torch.argmin(rms_list[i - self.max_sil_kept : i + 1])
                    + i
                    - self.max_sil_kept
                )
                if silence_start == 0:
                    sil_tags.append((0, pos_r))
                    clip_start = pos_r
                else:
                    sil_tags.append((min(pos_l, pos), max(pos_r, pos)))
                    clip_start = max(pos_r, pos)
            else:
                pos_l = (
                    torch.argmin(
                        rms_list[silence_start : silence_start + self.max_sil_kept + 1]
                    )
                    + silence_start
                )
                pos_r = (
                    torch.argmin(rms_list[i - self.max_sil_kept : i + 1])
                    + i
                    - self.max_sil_kept
                )
                if silence_start == 0:
                    sil_tags.append((0, pos_r))
                else:
                    sil_tags.append((pos_l, pos_r))
                clip_start = pos_r
            silence_start = None
        total_frames = rms_list.shape[0]
        if (
            silence_start is not None
            and total_frames - silence_start >= self.min_interval
        ):
            silence_end = min(total_frames, silence_start + self.max_sil_kept)
            pos = (
                torch.argmin(rms_list[silence_start : silence_end + 1]) + silence_start
            )
            sil_tags.append((pos, total_frames))
        if len(sil_tags) == 0:
            return [waveform]
        else:
            chunks = []
            if sil_tags[0][0] > 0:
                chunks.append(self._apply_slice(waveform, 0, sil_tags[0][0]))
            for i in range(len(sil_tags) - 1):
                chunks.append(
                    self._apply_slice(waveform, sil_tags[i][1], sil_tags[i + 1][0])
                )
            if sil_tags[-1][1] < total_frames:
                chunks.append(
                    self._apply_slice(waveform, sil_tags[-1][1], total_frames)
                )
            return chunks


def train_index(output_dir: Path, input_parameters: InputParameters):
    out = input_parameters.model_files / "checkpoint.index"
    feature_dir = output_dir / features
    listdir_res = list(os.listdir(feature_dir))
    npys = []
    for name in sorted(listdir_res):
        phone = np.load((feature_dir / name).as_posix())
        npys.append(phone)
    big_npy = np.concatenate(npys, 0)
    big_npy_idx = np.arange(big_npy.shape[0])
    np.random.shuffle(big_npy_idx)
    big_npy = big_npy[big_npy_idx]
    if big_npy.shape[0] > 2e5:
        big_npy = (
            MiniBatchKMeans(
                n_clusters=10000,
                verbose=True,
                batch_size=256 * 1,
                compute_labels=False,
                init="random",
            )
            .fit(big_npy)
            .cluster_centers_
        )

    np.save((output_dir / "total_fea.npy").as_posix(), big_npy)
    n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
    index = faiss.index_factory(
        256 if input_parameters.version == "v1" else 768, f"IVF{n_ivf},Flat"
    )
    index_ivf = faiss.extract_index_ivf(index)
    index_ivf.nprobe = 1
    index.train(big_npy)
    faiss.write_index(
        index,
        (
            output_dir / f"trained_IVF{n_ivf}_Flat_nprobe_{index_ivf.nprobe}.index"
        ).as_posix(),
    )

    batch_size_add = 8192
    for i in range(0, big_npy.shape[0], batch_size_add):
        index.add(big_npy[i : i + batch_size_add])
    faiss.write_index(
        index,
        out.as_posix(),
    )


def preprocess(
    input_dir: Path, output_dir: Path, progress: Progress, sr: int = 40000
) -> Path:
    assert input_dir.exists(), f"{input_dir} does not exist"
    output_dir.mkdir(exist_ok=True)

    slicer = Slicer(
        sr,
        silence_threshold=-42,
        min_length=1500,
        min_interval=400,
        hop_length=15,
        max_sil_kept=500,
    )

    bh, ah = signal.butter(N=5, Wn=48, btype="high", fs=sr)

    per = 3.0
    overlap = 0.3
    tail = per + overlap

    max = 0.9
    alpha = 0.75

    gt_wavs = output_dir / gt
    wavs16k = output_dir / sixteenk

    gt_wavs.mkdir(exist_ok=True)
    wavs16k.mkdir(exist_ok=True)

    def norm_write(tmp_audio: torch.Tensor, idx0, idx1):
        tmp_max = torch.abs(tmp_audio).max()
        if tmp_max > 2.5:
            logger.debug(f"{idx0}-{idx1} is filtered")
            return
        tmp_audio = (tmp_audio / tmp_max * (max * alpha)) + (1 - alpha) * tmp_audio
        # tmp_audio = ((tmp_audio / tmp_max * (max * alpha)) + (1 - alpha)) * tmp_audio
        wavfile.write(gt_wavs / f"{idx0}-{idx1}.wav", sr, tmp_audio.float().numpy())
        tmp_audio = Fa.resample(tmp_audio, sr, 16000)
        wavfile.write(wavs16k / f"{idx0}-{idx1}.wav", 16000, tmp_audio.float().numpy())

    for idx0, f in enumerate(os.listdir(input_dir)):
        input = input_dir / f
        try:
            audio, _ = load_audio(input, sr)
            audio = signal.lfilter(bh, ah, audio)
            audio = torch.from_numpy(audio.astype(np.float32))

            idx1 = 0
            slices = slicer(audio)
            for audio in progress.track(
                slices, total=len(slices), description="Splitting files..."
            ):
                i = 0
                while True:
                    start = int(sr * (per - overlap) * i)
                    i += 1
                    if len(audio[start:]) > tail * sr:
                        tmp_audio = audio[start : start + int(per * sr)]
                        norm_write(tmp_audio, idx0, idx1)
                        idx1 += 1
                    else:
                        tmp_audio = audio[start:]
                        idx1 += 1
                        break
                norm_write(tmp_audio, idx0, idx1)
        except Exception as e:
            logger.error(f"Failed to process {input.name}:", e)

    return output_dir


def extract_features(
    output_dir: Path,
    input_parameters: InputParameters,
    hyperparameters: TrainingParameters,
    progress: Progress,
) -> Path:
    gt_wavs = output_dir / gt
    features_dir = output_dir / features
    features_dir.mkdir(exist_ok=True)

    dtype = getattr(torch, hyperparameters.train.dtype)
    hubert.to(input_parameters.used_device, dtype)
    hubert.eval()

    files = os.listdir(gt_wavs)
    files = list(map(lambda x: gt_wavs / x, files))
    for f in progress.track(
        files, total=len(files), description="Extracting features..."
    ):
        out_file = features_dir / (f.stem + ".npy")
        feats = read_wave(f, normalize=saved_cfg.task.normalize)
        padding_mask = torch.BoolTensor(feats.shape).fill_(False)
        inputs = {
            "source": feats.to(input_parameters.used_device, dtype),
            "padding_mask": padding_mask.to(input_parameters.used_device),
            "output_layer": 9 if input.version == "v1" else 12,
        }
        with torch.no_grad():
            logits = hubert.extract_features(**inputs)
            feats = hubert.final_proj(logits[0]) if input.version == "v1" else logits[0]

        feats = feats.squeeze(0).float().cpu().numpy()
        if np.isnan(feats).sum() == 0:
            np.save(out_file, feats, allow_pickle=False)
        else:
            logger.warn(f"NaNs found in {f.name}, skipping")
    return features_dir


def extract_f0(
    output_dir: Path,
    input_parameters: InputParameters,
    hyperparameters: TrainingParameters,
    progress: Progress,
) -> Path:
    f0_dir = output_dir / f0
    f0_coarse_dir = output_dir / f0coarse
    f0_dir.mkdir(exist_ok=True)
    f0_coarse_dir.mkdir(exist_ok=True)

    files = os.listdir(output_dir / sixteenk)
    files = filter(lambda x: "spec" not in x, files)
    files = list(map(lambda x: output_dir / sixteenk / x, files))
    total = len(files)
    rvc_config = input_parameters.get_rvc_config(hyperparameters)
    for file in progress.track(files, total=total, description="Extracting F0..."):
        wav, _ = load_audio(file, 16000)
        (f0_coarse, f0_), wav = compute_pitch_from_audio(
            wav,
            rvc_config=rvc_config,
            extractors=input_parameters.pitch_extractor,
            skip_preprocess=True,
        )
        f0_.squeeze_()
        f0_coarse.squeeze_()

        np.save(
            f0_coarse_dir / f"{file.stem}.npy",
            f0_coarse.cpu().int().numpy(),
            allow_pickle=False,
        )
        np.save(
            f0_dir / f"{file.stem}.npy", f0_.cpu().float().numpy(), allow_pickle=False
        )


def write_filelist(output_dir: Path, input_parameters: InputParameters) -> None:
    gt_wavs = output_dir / gt

    opt = []
    for file in os.listdir(gt_wavs):
        file = gt_wavs / file
        gt_wav = file.as_posix()
        feature = (output_dir / features / f"{file.stem}.npy").as_posix()
        f0_ = (output_dir / f0 / f"{file.stem}.npy").as_posix()
        f0_coarse = (output_dir / f0coarse / f"{file.stem}.npy").as_posix()
        spkid = "0"
        opt.append(f"{gt_wav}|{feature}|{f0_coarse}|{f0_}|{spkid}")
    opt.append(
        "train_logs/mute/gtwaves/mute40k.wav|train_logs/mute/features/mute.npy|train_logs/mute/f0nsf/mute.wav.npy|train_logs/mute/f0/mute.wav.npy|0"
    )
    with open(input_parameters.model_files / "filelist.txt", "w+") as f:
        f.write("\n".join(opt))


def read_wave(wav_path: Path, normalize: bool = False) -> torch.Tensor:
    wav, sr = load_audio(wav_path.as_posix(), 16000)
    assert sr == 16000, "no idea how this could happen, but best to check just in case"
    feats = torch.from_numpy(wav).float()
    if feats.dim() == 2:
        feats = feats.mean(-1)
    assert feats.dim() == 1, feats.dim()
    if normalize:
        with torch.no_grad():
            feats = F.layer_norm(feats, feats.shape)
    feats = feats.view(1, -1)
    return feats


def setup_logging(training_parameters: TrainingParameters) -> TrainingLogger:
    if not USE_WANDB:
        return CLITrainingLogger()
    return wandb.init(
        project="rvc",
        config=asdict(training_parameters.train),
    )


def choose_models(version: str) -> Tuple[Type, Type]:
    """ret: [G, D]"""
    if version.lower() == "v2":
        return SynthesizerTrnMs768NSFsid, MultiPeriodDiscriminatorV2
    return SynthesizerTrnMs256NSFsid, MultiPeriodDiscriminator


def choose_optimizer(
    optimizer: OptimizerList,
    model: torch.nn.Module,
    lr: float,
    betas: List[float],
    eps: float,
) -> OptimizerType:
    if optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, betas=betas, eps=eps)
    elif optimizer == "adamw_8bit":
        from bitsandbytes.optim import AdamW8bit

        return AdamW8bit(model.parameters(), lr=lr, betas=betas, eps=eps)
    elif optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, betas=betas, eps=eps)
    elif optimizer == "adalomo":
        from lomo_optim import AdaLomo

        return AdaLomo(model, lr=lr, eps=eps, clip_grad_norm=1.0)
    elif optimizer == "lomo":
        from lomo_optim import Lomo

        return Lomo(model, lr=lr, clip_grad_norm=1.0)
    elif optimizer == "adamw_apex":
        raise NotImplementedError("Apex is not supported yet.")
    elif optimizer == "came":
        from came_pytorch import CAME

        return CAME(model.parameters(), lr=lr, betas=[0.8, 0.99, 0.9999])
    elif optimizer == "adamw_4bit":
        import lpmm

        return lpmm.optim.AdamW(
            model.parameters(), lr=lr, betas=betas, eps=eps, fused=True
        )
    elif optimizer == "adan":
        from adan import Adan

        return Adan(model.parameters(), lr=lr, eps=eps, fused=True)


def choose_scheduler(
    sched: Literal["exponential", "constant", "cosine"],
    optimizer: torch.optim.Optimizer,
    gamma: float,
    last_epoch: int,
    max_epochs: int,
):
    if sched == "exponential":
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer=optimizer, gamma=gamma, last_epoch=last_epoch
        )
    elif sched == "constant":
        return torch.optim.lr_scheduler.ConstantLR(
            optimizer=optimizer, last_epoch=last_epoch
        )
    elif sched == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=max_epochs, last_epoch=last_epoch
        )
    return None


def slice_segments(
    x: torch.Tensor, ids_str: str, segment_size: int = 4
) -> torch.Tensor:
    ret = torch.zeros_like(x[:, :, :segment_size])
    for i in range(x.size(0)):
        idx_str = ids_str[i]
        idx_end = idx_str + segment_size
        ret[i] = x[i, :, idx_str:idx_end]
    return ret


def save_small_model(
    ckpt: torch.nn.Module,
    name: str,
    epoch: int,
    hps: TrainingParameters,
    version: str = "v2",
    sr: int = 40000,
):
    opt = OrderedDict()
    opt["weight"] = {}
    dtype = getattr(torch, hps.train.dtype)
    for key in ckpt.keys():
        if "enc_q" in key:
            continue
        opt["weight"][key] = ckpt[key].to(dtype=dtype)
    opt["config"] = [
        hps.data.filter_length // 2 + 1,
        32,
        hps.model.inter_channels,
        hps.model.hidden_channels,
        hps.model.filter_channels,
        hps.model.n_heads,
        hps.model.n_layers,
        hps.model.kernel_size,
        hps.model.p_dropout,
        hps.model.resblock,
        hps.model.resblock_kernel_sizes,
        hps.model.resblock_dilation_sizes,
        hps.model.upsample_rates,
        hps.model.upsample_initial_channel,
        hps.model.upsample_kernel_sizes,
        hps.model.spk_embed_dim,
        hps.model.gin_channels,
        hps.data.sampling_rate,
    ]
    opt["info"] = f"{epoch}epoch"
    opt["sr"] = sr
    opt["f0"] = 1
    opt["version"] = version
    (Path("models/") / name).mkdir(exist_ok=True, parents=True)
    torch.save(opt, f"models/{name}/checkpoint.pth")
    return "Success."


def run(
    rank: int, n_gpus: int, input: InputParameters, hyperparameters: TrainingParameters
):
    global global_step
    if n_gpus > 1:
        raise ValueError("Only single-gpu training is supported for now")

    dist.init_process_group(
        backend="gloo", init_method="env://", world_size=n_gpus, rank=rank
    )

    # set seed
    torch.manual_seed(hyperparameters.train.seed)
    if input.used_device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed(hyperparameters.train.seed)
        torch.cuda.set_device(rank)

    dtype = getattr(torch, hyperparameters.train.dtype)

    train_dataset = TextAudioLoaderMultiNSFsid(
        input.model_files / "filelist.txt", hyperparameters.data
    )
    train_sampler = DistributedBucketSampler(
        train_dataset,
        batch_size=hyperparameters.train.batch_size * n_gpus,
        boundaries=list(map(lambda x: x * 100, range(1, 10))),
        num_replicas=n_gpus,
        rank=rank,
        shuffle=True,
    )

    collate_fn = TextAudioCollateMultiNSFsid()
    train_loader = DataLoader(
        train_dataset,
        num_workers=input.dataloader_workers,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn,
        batch_sampler=train_sampler,
        persistent_workers=True,
        prefetch_factor=8,
    )
    G, D = choose_models(input.version)
    net_g = G(
        hyperparameters.data.filter_length // 2 + 1,
        hyperparameters.train.segment_size // hyperparameters.data.hop_length,
        sr=hyperparameters.data.sampling_rate,
        **asdict(hyperparameters.model),
    ).to(input.used_device, dtype)
    net_d = D(hyperparameters.model.use_spectral_norm).to(input.used_device, dtype)

    optim = hyperparameters.train.optimizer
    lr = hyperparameters.train.learning_rate
    betas = hyperparameters.train.betas
    eps = hyperparameters.train.eps
    optimizer_g = choose_optimizer(optim, net_g, lr, betas, eps)
    optimizer_d = choose_optimizer(optim, net_d, lr, betas, eps)

    if input.used_device.type == "cuda" and torch.cuda.is_available():
        net_g = DDP(net_g, device_ids=[rank])
        net_d = DDP(net_d, device_ids=[rank])
    else:
        net_g = DDP(net_g)
        net_d = DDP(net_d)

    try:
        _, _, _, epoch_str = load_checkpoint(
            latest_checkpoint_path(input.model_files, "D_*.pth"),
            net_d,
            logger,
            optimizer_d,
            load_opt=True,
        )
        logger.info("Loaded net_d")
        _, _, _, epoch_str = load_checkpoint(
            latest_checkpoint_path(input.model_files, "G_*.pth"),
            net_g,
            logger,
            optimizer_g,
            load_opt=True,
        )
        global_step = (epoch_str - 1) * len(train_loader)
    except:  # noqa
        epoch_str = 1
        global_step = 0
        if input.pretrain_g != "":
            logger.info(f"Loading pretrain_g from {input.pretrain_g}")
            net_g.module.load_state_dict(
                torch.load(input.pretrain_g, map_location="cpu")["model"]
            )
        if input.pretrain_d != "":
            logger.info(f"Loading pretrain_d from {input.pretrain_d}")
            net_d.module.load_state_dict(
                torch.load(input.pretrain_d, map_location="cpu")["model"]
            )
    sched = hyperparameters.train.scheduler
    scheduler_g = choose_scheduler(
        sched,
        optimizer_g,
        hyperparameters.train.lr_decay,
        epoch_str - 2,
        hyperparameters.train.epochs,
    )
    scheduler_d = choose_scheduler(
        sched,
        optimizer_d,
        hyperparameters.train.lr_decay,
        epoch_str - 2,
        hyperparameters.train.epochs,
    )

    if (
        input.used_device.type == "cuda"
        and torch.cuda.is_available()
        and hyperparameters.train.dtype == "float16"
    ):
        # from torch.cuda.amp import GradScaler

        # scaler = GradScaler(enabled=True)
        scaler = None
    else:
        scaler = None

    train_logger = setup_logging(hyperparameters)

    cache = []
    with progress_bar as progress:
        task = progress.add_task("Training...", total=hyperparameters.train.epochs + 1)
        progress.update(task, advance=epoch_str)
        for epoch in progress.track(
            range(epoch_str, hyperparameters.train.epochs + 1), task_id=task
        ):
            train_and_evaluate(
                epoch,
                dtype,
                hyperparameters,
                input,
                [net_g, net_d],
                [optimizer_g, optimizer_d],
                scaler,
                [train_loader, None],
                train_logger,
                cache,
            )
            scheduler_g.step()
            scheduler_d.step()
    if not input.latest_only:
        save_checkpoint(
            net_g,
            optimizer_g,
            logger,
            hyperparameters.train.learning_rate,
            epoch,
            os.path.join(input.model_files, f"G_{epoch}.pth"),
        )
        save_checkpoint(
            net_d,
            optimizer_d,
            logger,
            hyperparameters.train.learning_rate,
            epoch,
            os.path.join(input.model_files, f"D_{epoch}.pth"),
        )
    else:
        save_checkpoint(
            net_g,
            optimizer_g,
            logger,
            hyperparameters.train.learning_rate,
            epoch,
            os.path.join(input.model_files, "G_2333333.pth"),
        )
        save_checkpoint(
            net_d,
            optimizer_d,
            logger,
            hyperparameters.train.learning_rate,
            epoch,
            os.path.join(input.model_files, "D_2333333.pth"),
        )


def train_and_evaluate(
    epoch: int,
    dtype: torch.dtype,
    hyperparameters: TrainingParameters,
    input: InputParameters,
    nets: List[torch.nn.Module],
    optimizers: List[torch.optim.Optimizer],
    scaler: Optional[torch.cuda.amp.GradScaler],
    loaders: List[DataLoader],
    train_logger: TrainingLogger,
    cache: List[Dict[str, Any]],
):
    global global_step

    net_g, net_d = nets
    optimizer_g, optimizer_d = optimizers
    train_loader, _ = loaders

    train_loader.batch_sampler.set_epoch(epoch)
    net_g.train()
    net_d.train()

    if input.cache_data:
        data_iterator = cache
        if len(cache) == 0:
            for batch_idx, info in enumerate(train_loader):
                (
                    phone,
                    phone_lengths,
                    pitch,
                    pitchf,
                    spec,
                    spec_lengths,
                    wave,
                    wave_lengths,
                    sid,
                ) = info
                if input.used_device.type == "cuda" and torch.cuda.is_available():
                    phone = phone.cuda(non_blocking=True)
                    phone_lengths = phone_lengths.cuda(non_blocking=True)
                    pitch = pitch.cuda(non_blocking=True)
                    pitchf = pitchf.cuda(non_blocking=True)
                    spec = spec.cuda(non_blocking=True)
                    spec_lengths = spec_lengths.cuda(non_blocking=True)
                    wave = wave.cuda(non_blocking=True)
                    wave_lengths = wave_lengths.cuda(non_blocking=True)
                    sid = sid.cuda(non_blocking=True)
                cache.append(
                    (
                        batch_idx,
                        (
                            phone,
                            phone_lengths,
                            pitch,
                            pitchf,
                            spec,
                            spec_lengths,
                            wave,
                            wave_lengths,
                            sid,
                        ),
                    )
                )
        else:
            shuffle(cache)
    else:
        data_iterator = enumerate(train_loader)

    for batch_idx, info in data_iterator:
        (
            phone,
            phone_lengths,
            pitch,
            pitchf,
            spec,
            spec_lengths,
            wave,
            wave_lengths,
            sid,
        ) = info
        if (
            not input.cache_data
            and input.used_device.type == "cuda"
            and torch.cuda.is_available()
        ):
            phone = phone.cuda(non_blocking=True)
            phone_lengths = phone_lengths.cuda(non_blocking=True)
            pitch = pitch.cuda(non_blocking=True)
            pitchf = pitchf.cuda(non_blocking=True)
            spec = spec.cuda(non_blocking=True)
            spec_lengths = spec_lengths.cuda(non_blocking=True)
            wave = wave.cuda(non_blocking=True)
            wave_lengths = wave_lengths.cuda(non_blocking=True)
            sid = sid.cuda(non_blocking=True)

        with torch.autocast(
            device_type=input.used_device.type,
            dtype=dtype,
            enabled=dtype != torch.float32,
        ):
            y_hat, ids_slice, x_mask, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q) = (
                net_g(phone, phone_lengths, pitch, pitchf, spec, spec_lengths, sid)
            )
            mel = torch_spec_to_mel(
                spec,
                hyperparameters.data.filter_length,
                hyperparameters.data.n_mel_channels,
                hyperparameters.data.sampling_rate,
                hyperparameters.data.mel_fmin,
                hyperparameters.data.mel_fmax,
            )
            y_mel = slice_segments(
                mel,
                ids_slice,
                hyperparameters.train.segment_size // hyperparameters.data.hop_length,
            )
            with torch.autocast(device_type=input.used_device.type, enabled=False):
                y_hat_mel = torch_mel_spectrogram(
                    y_hat.float().squeeze(1),
                    hyperparameters.data.filter_length,
                    hyperparameters.data.n_mel_channels,
                    hyperparameters.data.sampling_rate,
                    hyperparameters.data.hop_length,
                    hyperparameters.data.win_length,
                    hyperparameters.data.mel_fmin,
                    hyperparameters.data.mel_fmax,
                )
            y_hat_mel = y_hat_mel.to(dtype)
            wave = slice_segments(
                wave,
                ids_slice * hyperparameters.data.hop_length,
                hyperparameters.train.segment_size,
            )

            y_d_hat_r, y_d_hat_g, _, _ = net_d(wave, y_hat.detach())
            with torch.autocast(device_type=input.used_device.type, enabled=False):
                loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
                    y_d_hat_r, y_d_hat_g
                )
        optimizer_d.zero_grad()
        if scaler is not None:
            scaler.scale(loss_disc).backward()
            scaler.unscale_(optimizer_d)
            scaler.step(optimizer_d)
        else:
            if "lomo" in hyperparameters.train.optimizer:
                optimizer_d: Lomo
                optimizer_d.grad_norm(loss_disc)
                lr = optimizer_d.param_groups[0]["lr"]
                optimizer_d.fused_backward(loss_disc, lr)
            else:
                loss_disc.backward()
                optimizer_d.step()

        with torch.autocast(
            device_type=input.used_device.type,
            dtype=dtype,
            enabled=dtype != torch.float32,
        ):
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(wave, y_hat)
            with torch.autocast(device_type=input.used_device.type, enabled=False):
                loss_mel = F.l1_loss(y_mel, y_hat_mel) * hyperparameters.train.c_mel
                loss_kl = (
                    kl_loss(z_p, logs_q, m_p, logs_p, z_mask)
                    * hyperparameters.train.c_kl
                )
                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen, losses_gen = generator_loss(y_d_hat_g)
                loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl
        optimizer_g.zero_grad()
        if scaler is not None:
            scaler.scale(loss_gen_all).backward()
            scaler.unscale_(optimizer_g)
            scaler.step(optimizer_g)
            scaler.update()
        else:
            if "lomo" in hyperparameters.train.optimizer:
                optimizer_g.grad_norm(loss_gen_all)
                lr = optimizer_g.param_groups[0]["lr"]
                optimizer_g.fused_backward(loss_gen_all, lr)
            else:
                loss_gen_all.backward()
                optimizer_g.step()
        if global_step % hyperparameters.train.log_interval == 0:
            lr = optimizer_g.param_groups[0]["lr"]
            train_logger.log(
                {
                    "epoch": epoch,
                    "step": global_step,
                    "learning_rate": lr,
                    "loss/d/total": loss_disc.item(),
                    "loss/g/total": loss_gen_all.item(),
                    "loss/g/mel": loss_mel.item(),
                    "loss/g/kl": loss_kl.item(),
                    "loss/g/fm": loss_fm.item(),
                }
            )
        global_step += 1
    if epoch % input.save_every == 0:
        if not input.latest_only:
            save_checkpoint(
                net_g,
                optimizer_g,
                logger,
                hyperparameters.train.learning_rate,
                epoch,
                os.path.join(input.model_files, f"G_{epoch}.pth"),
            )
            save_checkpoint(
                net_d,
                optimizer_d,
                logger,
                hyperparameters.train.learning_rate,
                epoch,
                os.path.join(input.model_files, f"D_{epoch}.pth"),
            )
        else:
            save_checkpoint(
                net_g,
                optimizer_g,
                logger,
                hyperparameters.train.learning_rate,
                epoch,
                os.path.join(input.model_files, "G_2333333.pth"),
            )
            save_checkpoint(
                net_d,
                optimizer_d,
                logger,
                hyperparameters.train.learning_rate,
                epoch,
                os.path.join(input.model_files, "D_2333333.pth"),
            )


if __name__ == "__main__":
    while True:
        choice = Prompt.ask(
            "Do you want to extract a model or train a new one?",
            choices=["extract", "train", "index"],
            default="train",
        )

        if choice == "extract":
            model_name = Prompt.ask("Model name?", default="rvc_model")
            model_path = Path("models") / model_name
            model_path.mkdir(exist_ok=True, parents=True)

            log_dir = Path("train_logs") / model_name

            ckpt = torch.load(
                latest_checkpoint_path(log_dir, "G_*.pth"), map_location="cpu"
            )["model"]
            hps = TrainingParameters()
            hps.train.dtype = Prompt.ask(
                "What dtype should it be saved in?",
                choices=["float32", "float16", "bfloat16"],
                default="bfloat16",
            )

            save_small_model(ckpt, model_name, 0, hps, sr=40000)
        elif choice == "index":
            model_name = Prompt.ask("Model name?", default="rvc_model")
            model_path = Path("models") / model_name
            model_path.mkdir(exist_ok=True, parents=True)

            log_dir = Path("train_logs") / model_name
            input = InputParameters(
                training_files=None,
                model_files=model_path,
                used_device=None,
            )
            train_index(log_dir, input)
        elif choice == "train":
            input = InputParameters(
                training_files=Path("input_files/"),
                model_files=Path("output/"),
                used_device=torch.device("cuda"),
                cache_data=True,
                pretrain_g="weights/G_RIN_E.pth",
                pretrain_d="weights/D_RIN_E.pth",
            )
            hyperparameters = TrainingParameters()
            hyperparameters.train.dtype = "bfloat16"
            hyperparameters.train.optimizer = "lomo"
            hyperparameters.train.scheduler = "exponential"

            G_list = list(os.listdir("weights/"))
            G_list = list(
                filter(lambda x: "g_" in x.lower() or "_g" in x.lower(), G_list)
            )

            D_list = list(os.listdir("weights/"))
            D_list = list(
                filter(lambda x: "d_" in x.lower() or "_d" in x.lower(), D_list)
            )

            device_list = []
            device_list_ = []
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    device_list.append("(CUDA) " + torch.cuda.get_device_name(i))
                    device_list_.append(f"cuda:{i}")

            if torch.backends.mps.is_available():
                device_list.append("MPS")
                device_list_.append("mps")

            device_list.append(cpuinfo.get_cpu_info()["brand_raw"])
            device_list_.append("CPU")

            logger.info("Available devices:")
            for i, device in enumerate(device_list):
                logger.info(f" - {i}: {device}")

            device = device_list_[
                int(
                    Prompt.ask(
                        "Which device do you want to use?",
                        choices=list(map(str, range(len(device_list)))),
                        default="0",
                    )
                )
            ]

            dtypes = ["float32", "float16", "bfloat16"]
            supported = []
            for dt in dtypes:
                try:
                    dtype = getattr(torch, dt)
                    a = torch.tensor([1.0], device=device, dtype=dtype)
                    b = torch.tensor([2.0], device=device, dtype=dtype)
                    torch.matmul(a, b)
                    supported.append(dt)
                except RuntimeError:
                    pass
                except AssertionError:
                    pass
            dtype = Prompt.ask(
                "Which dtype do you want to use?",
                choices=supported,
                default=supported[-1],
            )
            input.used_device = torch.device(device)
            hyperparameters.train.dtype = dtype

            input_files = Prompt.ask("Input audio files?", default="./train_input/")
            input.training_files = Path(input_files)

            model_name = Prompt.ask(
                "What should the model be named?", default="rvc_model"
            )
            input.model_files = output_dir = Path("train_logs") / model_name

            hyperparameters.train.batch_size = int(
                Prompt.ask("Batch size?", default="12")
            )
            hyperparameters.train.learning_rate = float(
                Prompt.ask("Learning rate?", default="1e-4")
            )
            hyperparameters.train.epochs = int(
                Prompt.ask("Number of epochs?", default="1000")
            )
            hyperparameters.train.log_interval = int(
                Prompt.ask("Log interval?", default="5")
            )
            input.save_every = int(Prompt.ask("Save interval?", default="100"))
            input.latest_only = Confirm.ask("Save only the latest model?", default="y")
            input.cache_data = Confirm.ask("Cache all data to GPU memory?", default="y")
            hyperparameters.train.seed = int(Prompt.ask("Seed?", default="1337"))
            hyperparameters.train.optimizer = Prompt.ask(
                "Optimizer?",
                choices=[
                    "adamw",
                    "adamw_8bit",
                    "adamw_4bit",
                    "adam",
                    "adan",
                    "sophia",
                    "adalomo",
                    "lomo",
                    "adamw_apex",
                    "came",
                ],
                default="adan",
            )
            hyperparameters.train.scheduler = Prompt.ask(
                "Scheduler?",
                choices=["exponential", "constant", "cosine"],
                default="exponential",
            )

            input.pretrain_d = Path("weights/") / Prompt.ask(
                "Pretrained D?", choices=D_list, default=D_list[0]
            )
            input.pretrain_g = Path("weights/") / Prompt.ask(
                "Pretrained G?", choices=G_list, default=G_list[0]
            )

            with progress_bar as progress:
                (output_dir / gt).mkdir(exist_ok=True, parents=True)
                if len(os.listdir(output_dir / gt)) == 0:
                    preprocess(input.training_files, output_dir, progress)

                (output_dir / f0).mkdir(exist_ok=True)
                if len(os.listdir(output_dir / f0)) == 0:
                    extract_f0(output_dir, input, hyperparameters, progress)

                (output_dir / features).mkdir(exist_ok=True)
                if len(os.listdir(output_dir / features)) == 0:
                    extract_features(output_dir, input, hyperparameters, progress)

                if not (output_dir / "filelist.txt").exists():
                    write_filelist(output_dir, input)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            run(0, 1, input, hyperparameters)
