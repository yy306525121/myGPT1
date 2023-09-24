from abc import ABCMeta
from typing import List

from langchain.callbacks.base import BaseCallbackHandler


class BaseModel(metaclass=ABCMeta):
    def __init__(self, model_path: str = None, model_n_ctx: int = 1000, model_n_batch: int = 8,
                 callback: List[BaseCallbackHandler] = None, verbose: bool = False, gpu_layers: int = 0):
        pass