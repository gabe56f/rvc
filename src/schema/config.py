from typing import Optional

from strawberry import input, field, type

from .devices import DeviceInput, Device


@type
class Config:
    device: Device = field(description="The device to use.")
    dtype: str = field(description="The data type to use. (e.g. 'float32', 'float16')")
    batch_size: int = field(description="The batch size to use.")


@input
class ConfigInput:
    device: Optional[DeviceInput] = None
    dtype: Optional[str] = None
    batch_size: Optional[int] = None
