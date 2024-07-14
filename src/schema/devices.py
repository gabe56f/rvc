from typing import List, Optional
import psutil

from strawberry import type, field, input
from cpuinfo import get_cpu_info
import torch


def name_resolver(root: "Device"):
    if root.type == "cpu":
        return get_cpu_info()["brand_raw"]
    if root.type == "mps":
        return "MPS"
    if root.type == "cuda":
        return "(CUDA) " + torch.cuda.get_device_name(root.index)
    if root.type == "privateuseone":
        import torch_directml  # type: ignore

        return "(DML)" + torch_directml.device_name(root.index)
    if root.type == "xpu":
        return "(XPU) Intel GPU/ (generic) CPU device"
    return "(Unknown) Generic computing device"


def memory_resolver(root: "Device"):
    if root.type == "cpu" or root.type == "mps":
        return psutil.virtual_memory().total / 1024 / 1024
    if root.type == "cuda":
        return torch.cuda.mem_get_info(root.index)[0] / 1024 / 1024
    return "Generic computing device"


def datatype_resolver(root: "Device"):
    datatypes = ["float8_e4m3fn", "float8_e5m2", "bfloat16", "float16", "float32"]
    supported = []
    if root.type == "cpu" or root.type == "mps":
        device = torch.device(root.type)
    else:
        device = torch.device(root.type, root.index)
    for dt in datatypes:
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
    return supported


@type
class Device:
    type: str = field(description="The type of the device. (e.g. 'cpu', 'cuda')")
    index: int = field(description="The index of the device.")

    name: str = field(resolver=name_resolver, description="The name of the device.")
    memory: float = field(
        resolver=memory_resolver, description="The memory of the device in megabytes."
    )
    supported_datatypes: List[str] = field(
        resolver=datatype_resolver, description="The supported data types."
    )


@input
class DeviceInput:
    type: str
    index: Optional[int]
