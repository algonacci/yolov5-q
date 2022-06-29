# from torch.utils.data import Dataset, DataLoader
#
#
# class Test(Dataset):
#     def __init__(self) -> None:
#         super().__init__()
#
#     def __getitem__(self, index):
#         return dict(
#             box="1",
#             mask="2",
#             label="3",
#         )
#
#     def __len__(self):
#         return 10
#
#
# dataset = Test()
# test_loader = DataLoader(dataset, batch_size=4)

# from yolov5.core import MlvlPointGenerator
# import torch
#
# prior_generator = MlvlPointGenerator([8, 16, 32], offset=[0, 0])
# featmap_sizes = [[80, 80], [40, 40], [20, 20]]
# mlvl_priors = prior_generator.grid_priors(
#     featmap_sizes,
#     with_stride=True,
# )
# print(mlvl_priors[0], mlvl_priors[1])
# flatten_priors = torch.cat(mlvl_priors)
# print(flatten_priors.shape)
# flatten_priors = flatten_priors.repeat(1, 1)
# print(flatten_priors.shape)

from yolov5.core import multi_apply
import torch

def testA(a):
    return torch.tensor([1, 2, 3]), [2, 3, 4]

output = multi_apply(testA, [1, 2, 3, 4])[0]
output = torch.stack(output)
print(output, output.shape, output.dim())
output = output.sum(0).tolist()
print(output)

# import torch
# a = torch.zeros((2, 3), dtype=torch.bool)
# a[1, 2] = 1
# print(a)
# print((a == 1).sum())
