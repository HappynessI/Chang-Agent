"""Adapters for the independently installed upstream model environments."""

from .omniovcd_adapter import MaskPairProcessor, OmniOVCDAdapter
from .qwen3vl_adapter import GroundingModelQwen3VL
from .sam3_adapter import SAM3ProcessorAdapter
from .segagent_adapter import SimpleClickAdapter
from .subprocess_adapters import SubprocessBoxBackend, SubprocessPointBackend

__all__ = [
    "GroundingModelQwen3VL",
    "MaskPairProcessor",
    "OmniOVCDAdapter",
    "SAM3ProcessorAdapter",
    "SimpleClickAdapter",
    "SubprocessBoxBackend",
    "SubprocessPointBackend",
]
