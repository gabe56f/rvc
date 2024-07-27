import logging
from typing import Any

# from datetime import datetime

from rich.logging import RichHandler
from rich.prompt import Prompt, InvalidResponse
from rich.text import Text
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


class Question(Prompt):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def make_prompt(self, default: str) -> Text:
        """Make prompt text.

        Args:
            default (DefaultType): Default value.

        Returns:
            Text: Text to display in prompt.
        """
        prompt = self.prompt.copy()
        prompt.stylize("bold")
        prompt.end = ""

        default = (
            (self.choices[0] if len(self.choices) != 0 else None)
            if default is ...
            else default
        )

        if self.show_choices and self.choices:
            prompt.append("\r\n")
            for i, choice in enumerate(self.choices):
                if choice == default:
                    prompt.append(f"\r\n{i + 1}.  {choice}", "prompt.default")
                else:
                    prompt.append(f"\r\n{i + 1}.  {choice}", "prompt.choices")

        prompt.append("\r\n> ")

        return prompt

    def check_choice(self, value: str) -> bool:
        """Check value is in the list of valid choices.

        Args:
            value (str): Value entered by user.

        Returns:
            bool: True if choice was valid, otherwise False.
        """
        assert self.choices is not None
        try:
            v = int(value)
            return (v < len(self.choices) + 1) and v > 0
        except:  # noqa
            return value.strip().lower() in [choice.lower() for choice in self.choices]

    def process_response(self, value: str) -> Any:
        """Process response from user, convert to prompt type.

        Args:
            value (str): String typed by user.

        Raises:
            InvalidResponse: If ``value`` is invalid.

        Returns:
            PromptType: The value to be returned from ask method.
        """
        value = value.strip()
        try:
            return_value = self.response_type(value)
        except ValueError:
            raise InvalidResponse(self.validate_error_message)

        if self.choices is not None:
            if not self.check_choice(value):
                raise InvalidResponse(self.illegal_choice_message)

            # return the original choice, not the lower case version
            try:
                i = int(value)
                return_value = self.response_type(self.choices[i - 1])
            except:  # noqa
                return_value = self.response_type(
                    self.choices[
                        [choice.lower() for choice in self.choices].index(value.lower())
                    ]
                )
        return return_value
