from typing import Tuple, Optional
from pathlib import Path
import os


def is_valid_model(path: Path) -> bool:
    if not isinstance(path, Path):
        path = Path(path)
    return path.is_dir() and any([x.endswith(".pth") for x in os.listdir(path)])


def find_models(model_dir: Path) -> Tuple[str, Optional[str]]:
    files = os.listdir(model_dir)
    model = model_dir / list(filter(lambda x: x.endswith(".pth"), files))[0]
    if any([x.endswith(".index") for x in files]):
        index = model_dir / list(filter(lambda x: x.endswith(".index"), files))[0]
    else:
        index = None
    return model, index
