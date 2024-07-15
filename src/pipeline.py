import base64
from datetime import datetime
from typing import TYPE_CHECKING, Tuple, Dict, AsyncGenerator
from pathlib import Path
from time import time as time
import logging
import os

import ffmpeg
import numpy as np
import soundfile as sf
import torch
from torchaudio import functional as Fa

from .schema.generations import Generation, GenerationInput
from .config import get_config
from .models.separation import load_mdx23c, load_uvr, MDX23CConfig, UVR
from .models.hubert import load_hubert
from .models.rvc import load_rvc_model, SynthesizerType, infer, RVCConfig
from .models.pitch import compute_pitch_from_audio
from .logger import progress_bar
from .tagging import write_tags_to_flac
from .utils import load_audio

if TYPE_CHECKING:
    from faiss import Index

MODEL_FOLDER = "models/"
base_folder = Path(MODEL_FOLDER)
base_folder.mkdir(exist_ok=True)

TEMP_FOLDER = "temp/"
temp_folder = Path(TEMP_FOLDER)
temp_folder.mkdir(exist_ok=True)

ACCOMPANIMENT_FOLDER = "accompaniment/"
accompaniment_folder = Path(ACCOMPANIMENT_FOLDER)
accompaniment_folder.mkdir(exist_ok=True)

VOCALS_FOLDER = "vocals/"
vocals_folder = Path(VOCALS_FOLDER)
vocals_folder.mkdir(exist_ok=True)

OUTPUT_FOLDER = "output/"
output_folder = Path(OUTPUT_FOLDER)
output_folder.mkdir(exist_ok=True)

logger = logging.getLogger(__name__)


class VocalPipeline:
    loaded_models: Dict[str, Tuple[SynthesizerType, ...]] = {}
    loaded_uvr_models: Dict[str, UVR] = {}
    df, df_state = None, None
    hubert = None
    rvc_config = None
    mdx = None

    def to(self, device=None, dtype=None) -> "VocalPipeline":
        for v in self.loaded_models.values():
            v[0].to(device, dtype)
        for v in self.loaded_uvr_models.values():
            v.to(device, dtype)
        if self.hubert is not None:
            self.hubert.to(device, dtype)
        if self.mdx is not None:
            self.mdx.to(device, dtype)
        if self.rvc_config is not None:
            self.rvc_config.device = device
            self.rvc_config.dtype = dtype

        return self

    def _get_df(self):
        from df import enhance

        config = get_config()
        if self.df is None:
            from df import init_df

            self.df, self.df_state, _ = init_df(
                post_filter=True, log_level="none", log_file=None
            )
            self.df.to(config.device)
        return self.df, self.df_state, enhance

    def _get_mdx(self):
        if self.mdx is None:
            config = get_config()

            t0 = time()
            self.mdx = load_mdx23c(
                "mdx23c.ckpt", MDX23CConfig(), config.device, config.dtype
            )
            logger.debug(f"MDX23C loaded in {(time() - t0):3.0f} seconds.")
        return self.mdx

    def _get_hubert(self):
        if self.hubert is None:
            if self.rvc_config is None:
                raise ValueError(
                    "Illegal state, no rvc_config available, but trying to load hubert"
                )
            t0 = time()
            self.hubert = load_hubert("hubert_base.pt", self.rvc_config)
            logger.debug(f"Hubert loaded in {(time() - t0):3.0f} seconds.")
        return self.hubert

    def _get_uvr(self, model: str):
        if model in self.loaded_uvr_models:
            return self.loaded_uvr_models[model]
        config = get_config()

        t0 = time()
        uvr = load_uvr(model, config.device, config.dtype)
        self.loaded_uvr_models[model] = uvr
        logger.debug(f"{model} loaded in {(time() - t0):3.0f} seconds.")
        return uvr

    def load_model(
        self, model_name: str
    ) -> Tuple[SynthesizerType, "Index", float, str, int]:
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]

        if self.rvc_config is None:
            config = get_config()

            self.rvc_config = RVCConfig()
            self.rvc_config.device = config.device
            self.rvc_config.dtype = config.dtype
            self.rvc_config.batch_size = config.batch_size

        folder = base_folder / model_name
        t0 = time()
        model, index, big_npy, version, target_sr, self.rvc_config = load_rvc_model(
            folder / "checkpoint.pth", folder / "checkpoint.index", self.rvc_config
        )
        self.loaded_models[model_name] = (model, index, big_npy, version, target_sr)
        logger.debug(f"{model_name} loaded in {(time() - t0):3.0f} seconds.")
        return self.loaded_models[model_name]

    async def infer(
        self, generation: GenerationInput
    ) -> AsyncGenerator[Generation, None]:
        id = len(os.listdir(output_folder))
        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        time_taken = []
        output_count = 1 + sum(
            [
                int(p.save_accompaniment) + int(p.save_vocals)
                for p in generation.preprocess
            ]
        )
        outputs = []

        with progress_bar as main_progress:
            main_task = main_progress.add_task("Loading audio", total=6)
            config = get_config()

            t0 = time()

            (file := (temp_folder / f"input.{generation.filetype}")).write_bytes(
                base64.b64decode(generation.input.split(";base64,")[1])
            )
            audio, sr = load_audio(file, 44100)
            file.unlink()

            main_progress.update(main_task, advance=1)
            t1 = time()
            time_taken.append(t1 - t0)

            yield Generation(
                id=id,
                input_name=generation.filename,
                model_used=generation.model,
                created_at=created_at,
                time_taken=time_taken,
                status="preprocess_start",
                output=[],
                output_count=output_count,
            )

            if len(generation.preprocess) != 0:
                main_progress.update(main_task, description="Preprocessing audio")
                output_list = list(
                    map(lambda x: x.input, generation.preprocess[1:])
                ) + [generation.preprocess_output]
                for i, preprocess, next_preprocess in main_progress.track(
                    zip(
                        range(len(generation.preprocess)),
                        generation.preprocess,
                        output_list,
                    ),
                    total=len(generation.preprocess),
                ):
                    if preprocess.type == "MDX23C":
                        mdx = self._get_mdx()
                        if mdx.config.sample_rate != sr:
                            audio = (
                                Fa.resample(audio, sr, mdx.config.sample_rate)
                                .cpu()
                                .float()
                                .numpy()
                            )
                            sr = mdx.config.sample_rate
                        if len(audio.shape) == 1:
                            audio = np.asfortranarray([audio, audio])
                        sr, audio = mdx(audio)
                        vocals = audio["vocals"]
                        accompaniment = audio["accompaniment"]
                        if preprocess.save_accompaniment:
                            accompaniment_dir = (
                                Path(preprocess.accompaniment_directory_override)
                                or accompaniment_folder
                            )
                            accompaniment_dir.mkdir(exist_ok=True)
                            filename = f"accompaniment_{generation.filename}_{i}.wav"
                            sf.write(
                                accompaniment_dir / filename,
                                accompaniment.T,
                                sr,
                                "FLOAT",
                            )
                            outputs.append(
                                "http://localhost:8000/accompaniment/" + filename
                            )

                        if preprocess.save_vocals:
                            vocals_dir = (
                                Path(preprocess.vocals_directory_override)
                                or vocals_folder
                            )
                            vocals_dir.mkdir(exist_ok=True)
                            filename = f"vocals_{generation.filename}_{i}.wav"
                            sf.write(
                                vocals_dir / filename,
                                vocals.T,
                                sr,
                                "FLOAT",
                            )
                            outputs.append("http://localhost:8000/vocals/" + filename)
                        if next_preprocess == "file":
                            raise ValueError(
                                "Illegal state, preprocess_output cannot be file after processing already happened."
                            )
                        audio = audio[next_preprocess]
                    elif preprocess.type == "UVR":
                        uvr = self._get_uvr(preprocess.file)
                        if sr != 44100:
                            audio = (
                                Fa.resample(
                                    torch.from_numpy(audio).to(
                                        config.device, config.dtype
                                    ),
                                    sr,
                                    44100,
                                )
                                .cpu()
                                .float()
                                .numpy()
                            )
                            sr = 44100
                        sr, audio = uvr(audio)
                        vocals = audio["vocals"]
                        accompaniment = audio["accompaniment"]
                        if preprocess.save_accompaniment:
                            accompaniment_dir = (
                                Path(preprocess.accompaniment_directory_override)
                                or accompaniment_folder
                            )
                            accompaniment_dir.mkdir(exist_ok=True)
                            filename = f"accompaniment_{generation.filename}_{i}.wav"
                            sf.write(
                                accompaniment_dir / filename,
                                accompaniment,
                                sr,
                            )
                            outputs.append(
                                "http://localhost:8000/accompaniment/" + filename
                            )

                        if preprocess.save_vocals:
                            vocals_dir = (
                                Path(preprocess.vocals_directory_override)
                                or vocals_folder
                            )
                            vocals_dir.mkdir(exist_ok=True)
                            filename = f"vocals_{generation.filename}_{i}.wav"
                            sf.write(
                                vocals_dir / filename,
                                vocals,
                                sr,
                            )
                            outputs.append("http://localhost:8000/vocals/" + filename)
                        if next_preprocess == "file":
                            raise ValueError(
                                "Illegal state, preprocess_output cannot be file after processing already happened."
                            )
                        audio = audio[next_preprocess].T
                    elif preprocess.type == "DeepFilterNet":
                        df, df_state, enhance = self._get_df()
                        if sr != df_state.sr():
                            audio = (
                                Fa.resample(
                                    torch.from_numpy(audio).to(
                                        config.device, config.dtype
                                    ),
                                    sr,
                                    df_state.sr(),
                                )
                                .cpu()
                                .float()
                            )
                            sr = df_state.sr()
                            logger.debug(sr)
                        vocals = enhance(df, df_state, audio)
                        if preprocess.save_vocals:
                            vocals_dir = (
                                Path(preprocess.vocals_directory_override)
                                or vocals_folder
                            )
                            vocals_dir.mkdir(exist_ok=True)
                            filename = f"vocals_{generation.filename}_{i}.wav"
                            sf.write(
                                vocals_dir / filename,
                                vocals.cpu().float().numpy().T,
                                sr,
                            )
                            outputs.append("http://localhost:8000/vocals/" + filename)
                        if next_preprocess == "file":
                            raise ValueError(
                                "Illegal state, preprocess_output cannot be file after processing already happened."
                            )
                        audio: np.ndarray = vocals.cpu().float().numpy()
                    logger.debug(
                        f"audio shape post preprocess {i}-{preprocess.type}: {audio.shape}"
                    )

            t2 = time()
            main_progress.update(main_task, description="Loading models", advance=1)

            time_taken.append(t2 - t1)
            yield Generation(
                id=id,
                input_name=generation.filename,
                model_used=generation.model,
                created_at=created_at,
                time_taken=time_taken,
                status="model_load_start",
                output=[],
                output_count=output_count,
            )

            net_g, index, big_npy, version, target_sr = self.load_model(
                generation.model
            )
            hubert = self._get_hubert()
            t3 = time()

            time_taken.append(t3 - t2)
            yield Generation(
                id=id,
                input_name=generation.filename,
                model_used=generation.model,
                created_at=created_at,
                time_taken=time_taken,
                status="pitch_start",
                output=[],
                output_count=output_count,
            )

            main_progress.update(main_task, description="Calculating pitch", advance=1)
            audio = (
                Fa.resample(
                    torch.from_numpy(audio).to(config.device, config.dtype), sr, 16000
                )
                .cpu()
                .float()
                .numpy()
            )
            logger.debug(f"audio shape post-16000: {audio.shape}")
            f0_output = compute_pitch_from_audio(
                audio,
                generation.transpose,
                rvc_config=self.rvc_config,
                extractors=generation.pitch_extraction,
            )
            t4 = time()

            time_taken.append(t4 - t3)
            yield Generation(
                id=id,
                input_name=generation.filename,
                model_used=generation.model,
                created_at=created_at,
                time_taken=time_taken,
                status="infer_start",
                output=[],
                output_count=output_count,
            )

            main_progress.update(main_task, description="Inferring f0", advance=1)
            try:
                output, target_sr = infer(
                    f0_output,
                    hubert,
                    net_g,
                    index,
                    big_npy,
                    self.rvc_config,
                    target_sr,
                    rms_mix_rate=generation.rms_mix_rate,
                    version=version,
                )
            except:  # noqa
                (pitch, pitchf), audio = f0_output
                pitch = pitch[:, 1:]
                pitchf = pitchf[:, 1:]
                output, target_sr = infer(
                    ((pitch, pitchf), audio),
                    hubert,
                    net_g,
                    index,
                    big_npy,
                    self.rvc_config,
                    target_sr,
                    rms_mix_rate=generation.rms_mix_rate,
                    version=version,
                )

            sf.write(
                (input_file := (temp_folder / "output.wav")).as_posix(),
                output,
                target_sr,
            )
            t5 = time()
            main_progress.update(main_task, description="Writing audio", advance=1)

            time_taken.append(t5 - t4)
            yield Generation(
                id=id,
                input_name=generation.filename,
                model_used=generation.model,
                created_at=created_at,
                time_taken=time_taken,
                status="convert_output_start",
                output=[],
                output_count=output_count,
            )

            ffmpeg.input(input_file).output(
                (
                    output_file := (
                        output_folder / f"{generation.filename}_{generation.model}.flac"
                    )
                ).as_posix()
            ).run(overwrite_output=True, quiet=True)
            outputs.append(f"http://localhost:8000/audio/{output_file.name}")
            input_file.unlink()

            generation_data = {
                "id": str(id),
                "input_name": generation.filename,
                "model_used": generation.model,
                "created_at": created_at,
                "time_taken": "|".join(map(str, time_taken)),
                "status": "done",
                "output": "|".join(outputs),
                "output_count": str(output_count),
            }
            write_tags_to_flac(output_file, generation_data)

            t6 = time()
            main_progress.update(main_task, description="Done", advance=1)

            time_taken.append(t6 - t5)
            logger.info(
                f"[bold green]Inference done in: {(time_taken[-1] - time_taken[0]):3.1f}[/]"
            )

            yield Generation(
                id=id,
                input_name=generation.filename,
                model_used=generation.model,
                created_at=created_at,
                time_taken=time_taken,
                status="done",
                output=outputs,
                output_count=output_count,
            )


PIPELINE = VocalPipeline()
