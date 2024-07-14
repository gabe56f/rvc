import logging

# from datetime import datetime

from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    Progress,
    TimeRemainingColumn,
    TextColumn,
    MofNCompleteColumn,
    TimeElapsedColumn,
)

progress_bar = Progress(
    TextColumn("[progress.percentage]{task.description}"),
    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    BarColumn(),
    MofNCompleteColumn(),
    TextColumn("•"),
    TimeElapsedColumn(),
    TextColumn("•"),
    TimeRemainingColumn(),
)

logging.basicConfig(
    level="INFO",
    format="%(asctime)s | %(name)s » %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        RichHandler(
            rich_tracebacks=True, show_time=False, show_path=False, markup=True
        ),
        # logging.FileHandler(
        #     f"data/logs/{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}.log",
        #     mode="w",
        #     encoding="utf-8",
        # ),
    ],
)
logging.getLogger("fairseq.tasks").setLevel(logging.DEBUG)
logging.getLogger("faiss.loader").setLevel(logging.DEBUG)
logging.getLogger("fairseq.models.hubert.hubert").setLevel(logging.DEBUG)
