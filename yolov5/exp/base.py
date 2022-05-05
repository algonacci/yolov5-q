import ast
import pprint
from abc import ABCMeta, abstractmethod
from typing import Dict
from tabulate import tabulate

import torch
from torch.nn import Module


class BaseExp(metaclass=ABCMeta):
    """Basic class for any experiment."""

    def __init__(self):
        self.seed = None
        self.output_dir = "./runs"

    # @abstractmethod
    # def get_model(self) -> Module:
    #     pass

    @abstractmethod
    def get_data_loader(
        self, batch_size: int, rank: int
    ) -> Dict[str, torch.utils.data.DataLoader]:
        pass

    @abstractmethod
    def set_logger(self, rank: int):
        pass

    @abstractmethod
    def get_loss(self, batch_size: int):
        pass

    @abstractmethod
    def get_evaluator(self):
        pass

    @abstractmethod
    def eval(self, model, evaluator, weights):
        pass

    def __repr__(self):
        table_header = ["keys", "values"]
        exp_table = [
            (str(k), pprint.pformat(v))
            for k, v in vars(self).items()
            if not k.startswith("_")
        ]
        return tabulate(exp_table, headers=table_header, tablefmt="fancy_grid")

    def merge(self, cfg_list):
        assert len(cfg_list) % 2 == 0
        for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
            # only update value with same key
            if hasattr(self, k):
                src_value = getattr(self, k)
                src_type = type(src_value)
                if src_value is not None and src_type != type(v):
                    try:
                        v = src_type(v)
                    except Exception:
                        v = ast.literal_eval(v)
                setattr(self, k, v)
