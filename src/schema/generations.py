from typing import Optional, List

from strawberry import type, field, input
from strawberry.file_uploads import Upload


@type
class Generation:
    id: int = field(description="The ID of the generation.")
    input_name: str = field(description="The name of the generation.")
    model_used: str = field(description="The model used for the generation.")
    created_at: str = field(description="The time the generation started.")
    time_taken: List[float] = field(
        description="The time taken for the generation in seconds, per stage."
    )
    status: str = field(description="The status of the generation.")
    output: List[str] = field(description="The output of the generation.")
    output_count: int = field(description="The number of outputs.")


@input
class Preprocess:
    input: str = field(description="The input for the preprocess. Start with 'file.'")
    type: str = field(description="The type of the preprocess -- 'UVR' or 'MDX23C'.")
    file: str = field(
        description="The file to use for the preprocess, only used with UVR."
    )
    save_accompaniment: Optional[bool] = field(
        description="Whether to save the accompaniment to a file. Defaults to true."
    )
    save_vocals: Optional[bool] = field(
        description="Whether to save the vocals to a file. Defaults to false."
    )
    accompaniment_directory_override: Optional[str] = field(
        description="The directory to save the accompaniment to. (Defaults to './accompaniment/')"
    )
    vocals_directory_override: Optional[str] = field(
        description="The directory to save the vocals to. (Defaults to './vocals/')"
    )


@input
class GenerationInput:
    filename: str
    filetype: str
    input: Upload
    model: str
    transpose: int
    pitch_extraction: str
    index_rate: float = 1.0
    rms_mix_rate: float = 1
    preprocess: List[Preprocess]
    preprocess_output: str = field(
        description="The output of the preprocess/the one we want to use for vocal transformation."
    )
