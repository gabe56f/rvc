from pathlib import Path
from typing import List, AsyncGenerator
import logging
import os

from strawberry import type, field, mutation, subscription
import torch

from .devices import Device
from .config import Config, ConfigInput
from .generations import Generation, GenerationInput
from .generic import Entry
from .model import Model
from ..config import get_config
from ..tagging import get_tags
from ..utils import is_available, is_valid_model


logger = logging.getLogger(__name__)


@type
class Query:
    @field
    def devices(self) -> List[Device]:
        devices = []
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                devices.append(Device(type="cuda", index=i))
        if torch.backends.mps.is_available():
            devices.append(Device(type="mps", index=0))
        devices.append(Device(type="cpu", index=0))
        try:
            import torch_directml  # type: ignore

            for i in range(torch_directml.device_count()):
                devices.append(Device(type="privateuseone", index=i))
        except ImportError:
            pass
        return devices

    @field
    def pitchExtractors(self) -> List[Entry]:
        out = []
        if Path("rmvpe.pt").exists():
            out.append(Entry(label="RMVPE", value="rmvpe"))
        if is_available("torchfcpe"):
            out.append(Entry(label="FCPE", value="fcpe"))
        if is_available("torchcrepe"):
            out.append(Entry(label="Crepe", value="crepe"))
        if is_available("essentia"):
            out.append(Entry(label="Melodia", value="melodia"))
        if is_available("pyworld"):
            out.append(Entry(label="Harvest", value="harvest"))
            out.append(Entry(label="Dio", value="dio"))
        if is_available("parselmouth"):
            out.append(Entry(label="Parselmouth", value="pm"))
        out.append(Entry(label="Pyin", value="pyin"))
        return out

    @field
    def uvrModels(self) -> List[Entry]:
        out = []
        for f in os.listdir("./"):
            if f.endswith(".pth"):
                out.append(Entry(label=f, value=f))
        for f in os.listdir("models/"):
            if f.endswith(".pth"):
                out.append(Entry(label=f, value=f"models/{f}"))
        return out

    @field
    def config(self) -> Config:
        config = get_config()
        t = config.device.split(":")
        if len(t) == 1:
            t, i = t[0], "0"
        else:
            t, i = t

        return Config(
            device=Device(type=t, index=int(i)),
            dtype=str(config.dtype).split(".")[1],
            batch_size=config.batch_size,
        )

    @field
    def models(self) -> List[Model]:
        return list(
            map(
                lambda x: Model(
                    label=(x[0].upper() + x[1:]).replace("_", " ").replace("-", " "),
                    value=x,
                ),
                filter(
                    lambda x: is_valid_model(Path("models/") / x), os.listdir("models/")
                ),
            )
        )

    @field
    def past_generations(self, count: int = 10) -> List[Generation]:
        from ..pipeline import output_folder

        generations: List[Generation] = []
        for f in os.listdir(output_folder):
            if not f.endswith(".flac"):
                continue
            tags = dict(get_tags(output_folder / f))
            gen = Generation(
                id=int(tags["id"][0]),
                input_name=tags["input_name"][0],
                model_used=tags["model_used"][0],
                created_at=tags["created_at"][0],
                time_taken=list(
                    map(lambda x: float(x), tags["time_taken"][0].split("|"))
                ),
                status=tags["status"][0],
                output=tags["output"][0].split("|"),
                output_count=int(tags["output_count"][0]),
            )
            generations.append(gen)
        generations = list(
            sorted(generations, key=lambda x: x.created_at, reverse=True)
        )[: min(count, len(generations))]
        return generations


@type
class Mutations:
    @mutation
    def set_config(self, input: ConfigInput) -> Config:
        conf = get_config()

        if input.device is not None:
            if input.device.index is None:
                conf.device = input.device.type
            else:
                conf.device = f"{input.device.type}:{input.device.index}"

        if input.dtype is not None:
            conf.dtype = getattr(torch, input.dtype)

        if input.batch_size is not None:
            conf.batch_size = input.batch_size

        return Config(
            device=Device(type=input.device.type, index=input.device.index),
            dtype=input.dtype,
            batch_size=input.batch_size,
        )

    @mutation
    async def create_and_await_result(self, input: GenerationInput) -> Generation:
        from ..pipeline import PIPELINE

        async for output in PIPELINE.infer(input):
            pass

        return output


@type
class Subscriptions:
    @subscription
    async def create_inference_request(
        self, input: GenerationInput
    ) -> AsyncGenerator[Generation, None]:
        from ..pipeline import PIPELINE

        return PIPELINE.infer(input)
