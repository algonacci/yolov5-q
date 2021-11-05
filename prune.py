from loguru import logger
from torch import nn
import torch_pruning as tp
import torch
import tabulate
import numpy as np
import os
import matplotlib.pyplot as plt
import copy
from models.yolo import Model
from collections import defaultdict
from typing import List, Tuple
from utils.general import check_yaml
from val_sparse import run
import yaml

print = logger.info

ignore_modules = [
    "model.2.m.0.cv2.bn",
    "model.13.m.0.cv2.conv",
    "model.2.cv1.bn",
    "model.6.m.0.cv1.bn",
    "model.20.m.0.cv1.conv",
    "model.23.m.0.cv2.conv",
    "model.6.m.2.cv1.conv",
    "model.20.m.0.cv2.conv",
    "model.2.cv1.conv",
    "model.8.cv1.bn",
    "model.23.cv3.conv",
    "model.4.m.1.cv2.conv",
    "model.17.cv3.conv",
    "model.4.cv1.conv",
    "model.8.m.0.cv2.bn",
    "model.17.m.0.cv1.bn",
    "model.20.m.0.cv1.bn",
    "model.8.cv3.conv",
    "model.2.m.0.cv1.conv",
    "model.17.m.0.cv1.conv",
    "model.8.m.0.cv1.conv",
    "model.4.m.0.cv1.conv",
    "model.13.m.0.cv1.conv",
    "model.2.m.0.cv2.conv",
    "model.8.m.0.cv1.bn",
    "model.13.cv3.conv",
    "model.4.m.1.cv1.conv",
    "model.8.m.0.cv2.conv",
    "model.23.m.0.cv1.bn",
    "model.4.m.1.cv2.bn",
    "model.6.m.0.cv2.conv",
    "model.6.cv1.bn",
    "model.4.m.1.cv1.bn",
    "model.6.cv1.conv",
    "model.4.m.0.cv2.conv",
    "model.6.cv3.conv",
    "model.4.m.0.cv1.bn",
    "model.17.m.0.cv2.conv",
    "model.4.cv3.conv",
    "model.6.m.2.cv2.bn",
    "model.13.m.0.cv2.bn",
    "model.6.m.1.cv1.conv",
    "model.20.cv3.conv",
    "model.2.m.0.cv1.bn",
    "model.23.m.0.cv2.bn",
    "model.6.m.1.cv2.conv",
    "model.4.cv1.bn",
    "model.6.m.0.cv1.conv",
    "model.6.m.1.cv2.bn",
    "model.4.m.0.cv2.bn",
    "model.20.m.0.cv2.bn",
    "model.23.m.0.cv1.conv",
    "model.6.m.1.cv1.bn",
    "model.6.m.2.cv1.bn",
    "model.6.m.2.cv2.conv",
    "model.13.m.0.cv1.bn",
    "model.8.cv1.conv",
    "model.17.m.0.cv2.bn",
    "model.6.m.0.cv2.bn",
    "model.2.cv3.conv",
]

def load_model(cfg="models/mobile-yolo5l_voc.yaml", weights="./outputs/mvoc/weights/best_mvoc.pt"):
    model = Model(cfg).to(device)
    ckpt = torch.load(weights, map_location=device)['model']  # load checkpoint
    model.load_state_dict(ckpt, strict=False)
    del ckpt

    model.float()
    model.model[-1].export = True
    return model


def _format_size(x: int, sig_figs: int = 3, hide_zero: bool = False) -> str:
    """
    Formats an integer for printing in a table or model representation.
    Expresses the number in terms of 'kilo', 'mega', etc., using
    'K', 'M', etc. as a suffix.

    Args:
        x (int) : The integer to format.
        sig_figs (int) : The number of significant figures to keep
        hide_zero (bool) : If True, x=0 is replaced with an empty string
            instead of '0'.

    Returns:
        str : The formatted string.
    """
    if hide_zero and x == 0:
        return str("")

    def fmt(x: float) -> str:
        # use fixed point to avoid scientific notation
        return "{{:.{}f}}".format(sig_figs).format(x).rstrip("0").rstrip(".")

    if abs(x) > 1e14:
        return fmt(x / 1e15) + "P"
    if abs(x) > 1e11:
        return fmt(x / 1e12) + "T"
    if abs(x) > 1e8:
        return fmt(x / 1e9) + "G"
    if abs(x) > 1e5:
        return fmt(x / 1e6) + "M"
    if abs(x) > 1e2:
        return fmt(x / 1e3) + "K"
    return str(x)


class Pruning:
    def __init__(self, model, prunable_module_type, ignore_modules):
        self.model = copy.deepcopy(model)
        self.model.cpu().eval()

        self.ori_model = model
        self.ori_model.cpu().eval()

        self.prunable_module_type = prunable_module_type

        self.ignore_modules = ignore_modules
        self.prunable_modules = self.get_prunable_modules()

    def bn_analyze(self, save_path=None):
        bn_val = []
        max_val = []
        for n, layer_to_prune in self.prunable_modules.items():
            # select a layer
            weight = layer_to_prune.weight.data.detach().cpu().numpy()
            # weight = layer_to_prune.bias.data.detach().cpu().numpy()
            max_val.append(max(weight))
            bn_val.extend(weight)
        bn_val = np.abs(bn_val)
        max_val = np.abs(max_val)
        bn_val = sorted(bn_val)
        max_val = sorted(max_val)

        plt.hist(bn_val, bins=101, align="mid", log=True, range=(0, 1.0))
        if save_path is not None:
            if os.path.isfile(save_path):
                os.remove(save_path)
            plt.savefig(save_path)
        return bn_val, max_val

    def get_prunable_modules(self):
        prunable_modules = {}

        for k, v in self.model.named_modules():
            if k in self.ignore_modules:
                continue
            if isinstance(v, self.prunable_module_type):
                prunable_modules[k] = v
        return prunable_modules

    def pruning(self, thres=None, pruned_prob=0.3, check_only=False, weight_only=False):
        """
        Args:
            check_only (bool): Set BN value to zero for test.
            weight_only (bool): Whether to prune according to the index of weight only,
                Note:
                    In this way, a greater pruning rate can be obtained, but may get wrose results
                        on small model.

        """
        ori_size = tp.utils.count_params(self.model)

        DG = tp.DependencyGraph().build_dependency(
            self.model, example_inputs=example_inputs, output_transform=output_transform
        )

        fig, ax = plt.subplots(1, 1, figsize=(18, 6))
        bn_val, max_val = self.bn_analyze(save_path="./before_pruning.jpg")

        if thres is None:
            thres_pos = int(pruned_prob * len(bn_val))
            thres_pos = min(thres_pos, len(bn_val) - 1)
            thres_pos = max(thres_pos, 0)
            thres = bn_val[thres_pos]
        print(
            "Min val is %f, Max val is %f, Thres is %f" % (bn_val[0], bn_val[-1], thres)
        )

        num_pruned_channels = defaultdict(int)
        ignore_modules = []
        for name, layer_to_prune in self.prunable_modules.items():
            weight = layer_to_prune.weight.data.detach().cpu().numpy()
            bias = layer_to_prune.bias.data.detach().cpu().numpy()
            if isinstance(layer_to_prune, nn.Conv2d):
                if layer_to_prune.groups > 1:
                    prune_fn = tp.prune_group_conv
                else:
                    prune_fn = tp.prune_conv
                L1_norm = np.sum(np.abs(weight), axis=(1, 2, 3))
            elif isinstance(layer_to_prune, nn.BatchNorm2d):
                prune_fn = tp.prune_batchnorm
                weight_norm = np.abs(weight)
                bias_norm = np.abs(bias)

            pruning_idx = (
                (weight_norm < thres)
                if weight_only
                else (weight_norm < thres) * (bias_norm < thres)
            )
            if check_only:
                pruned_idx_mask = np.array(~(pruning_idx), dtype=np.float32)
                layer_to_prune.weight = torch.nn.Parameter(
                    torch.tensor(weight * pruned_idx_mask), requires_grad=False
                )
                layer_to_prune.bias = torch.nn.Parameter(
                    torch.tensor(bias * pruned_idx_mask), requires_grad=False
                )
            else:
                pos = np.array([i for i in range(len(weight_norm))])
                pruned_idx_mask = pruning_idx
                prun_index = pos[pruned_idx_mask].tolist()

                if len(prun_index) == len(weight_norm):
                    del prun_index[np.argmax(weight_norm)]

                num_pruned_channels[name] = len(prun_index)

                plan, ignore_module = DG.get_pruning_plan(
                    layer_to_prune, prune_fn, prun_index
                )
                ignore_modules += ignore_module
                plan.exec()

        self.num_pruned_channels = num_pruned_channels
        # self.ignore_modules = list(set(ignore_modules))
        # for i in self.ignore_modules:
        #     print(i)

        # with open('nano.yaml', 'w') as f:
        #     yaml.safe_dump({'ignore_modules': self.ignore_modules,
        #                     'pruning_ignore_modules': [
        #                         ['.m', '.conv2'],
        #                         '.dconv'
        #                     ],
        #                     'sparse_rate': 0.01,
        #                     'backbone_only': False}, f)
        # print(yaml.safe_load(f))

        # print(len(self.ignore_modules))

        self.bn_analyze(save_path="./after_pruning.jpg")

        with torch.no_grad():

            out = self.model(example_inputs)
            if output_transform:
                out = output_transform(out)
            print("  Params: %s => %s" % (ori_size, tp.utils.count_params(self.model)))
            # if isinstance(out, (list, tuple)):
            #     for o in out:
            #         print("  Output: ", o.shape)
            # else:
            #     print(f"Output: {out.shape}")
            # print("------------------------------------------------------\n")
        return self.model

    def compare(self):
        original_model = ModelDetails(self.ori_model)
        pruned_model = ModelDetails(self.model)

        param_shapes = []
        param_shapes: List[Tuple[int, ...]] = [
            (
                name,
                f"{tuple(pruned_model.shapes[name])} / {tuple(original_model.shapes[name])}",
                f"{pruned_model.params[name]} / {original_model.params[name]}",
                f"{self.num_pruned_channels[name[:name.rfind('.')]]}",
                True if name[: name.rfind(".")] in self.ignore_modules else False,
            )
            for name, _ in self.model.named_parameters()  # if 'backbone' in name
        ]

        param_shapes.append(
            (
                "Total",
                None,
                f"{pruned_model.total_params} / {original_model.total_params}",
                sum(list(self.num_pruned_channels.values())),
            )
        )
        # print(len(param_shapes))
        tab = tabulate.tabulate(
            param_shapes,
            headers=[
                "name",
                "shape(pruned / original)(out, in, *kernel)",
                "parameters(pruned / original)",
                "num_pruned_channels",
                "ignore",
            ],
            tablefmt="fancy_grid",
            missingval="---",
            stralign="center",
            numalign="center",
        )
        # with open('pruned_d2', 'w') as f:
        #     f.write(tab)
        print('\n' + tab)

    def save_pruned(self, save_path="", half=False):
        model = self.model.module if hasattr(self.model, "module") else self.model
        model = model.half() if half else model
        torch.save({"model": model}, save_path)


class ModelDetails:
    def __init__(self, model):
        self.model = model
        (self._params, self._shapes, self._total_params) = self._params_shapes()

    def _params_shapes(self):
        params = {}
        shapes = {}
        total_params = 0
        for name, param in self.model.named_parameters():
            params[name] = param.numel()
            shapes[name] = param.shape

            total_params += params[name]
        return params, shapes, total_params

    def params_shapes_table(self):
        param_shapes = [
            (name, f"{self.shapes[name]}", f"{self.params[name]}")
            for name, _ in self.model.named_parameters()
        ]
        param_shapes.append(("Total", None, f"{self.total_params}"))
        tab = tabulate.tabulate(
            param_shapes,
            headers=["name", "shape", "parameters"],
            tablefmt="fancy_grid",
            missingval="None",
        )
        print("\n" + tab)

    @property
    def total_params(self):
        return _format_size(self._total_params)

    @property
    def params(self):
        return self._params

    @property
    def shapes(self):
        return self._shapes

    def total_flop(self, img=None):
        if img is None:
            img = torch.zeros(
                (1, 3, 416, 416), device=next(self.model.parameters()).device
            )
            logger.warning(
                f"Args img is None, so taking default img with shape {img.shape}"
            )
        try:
            from fvcore.nn import FlopCountAnalysis

            flops = FlopCountAnalysis(self.model, img)
            return _format_size(flops.total())
        except Exception as e:
            logger.warning(e)
            return "0"

    def flop_table(self, img=None):
        if img is None:
            img = torch.zeros(
                (1, 3, 416, 416), device=next(self.model.parameters()).device
            )
            logger.warning(
                f"Args img is None, so taking default img with shape {img.shape}"
            )
        try:
            from fvcore.nn import flop_count_table, FlopCountAnalysis, flop_count_str

            flops = FlopCountAnalysis(self.model, img)
            print("\n" + flop_count_table(flops, max_depth=10))
            # print('\n' + flop_count_str(flops))
        except Exception as e:
            logger.warning(e)


if __name__ == "__main__":
    CFG_FILE = "/home/laughing/yolov5/models/yolov5s.yaml"
    MODEL_FILE = "/home/laughing/yolov5/runs/prune/guiyang_spare6/weights/best.pt"

    PRUNING_FILE = "/home/laughing/yolov5/weights/pruned_auto.pt"

    device = torch.device("cpu")
    model = load_model(CFG_FILE, MODEL_FILE)  # load checkpoint

    pruned_prob = 0.0
    example_inputs = torch.zeros((1, 3, 640, 640), dtype=torch.float32).to()

    output_transform = None
    thres = 0.01

    if thres != 0:
        thres = thres
        pruned_prob = "p.auto"
    else:
        thres = None
        pruned_prob = pruned_prob

    pruning = Pruning(model, (nn.BatchNorm2d,), ignore_modules=ignore_modules)
    pruned_model = pruning.pruning(thres=thres, check_only=False, weight_only=False)
    pruning.compare()

    # data = check_yaml('/home/laughing/yolov5/data/custom/guiyang_phone.yaml')
    # _ = run(data, model=pruned_model.cuda(), batch_size=8, imgsz=640, conf_thres=.001, iou_thres=.6,
    #     device='0', save_json=False, plots=False, half=False)
