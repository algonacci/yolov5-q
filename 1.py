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

# from yolov5.core import multi_apply
# import torch
#
# def testA(a):
#     return torch.tensor([[1, 1], [2, 2], [3, 3]]), [torch.tensor([2, 2, 2]), torch.tensor([3, 3]), torch.tensor([4])]
#
# output = multi_apply(testA, [1, 2, 3, 4])[1]
# print(*output)
# for i in zip(*output):
#     print(torch.cat(i))
# exit()
# stats = [np.concatenate(x, 0) for x in zip(*output)]  # to numpy
# output = torch.stack(output, dim=1)
# print(output, output.shape, output.dim())
# output = output.sum(0).tolist()
# print(output)

import torch
a = torch.zeros((2, 3), dtype=torch.bool)
a[1, 2] = 1
b = torch.as_tensor(a.clone(), dtype=torch.long)
print(b)
b[b==1] = 2
print(b)
# print((a == 1).sum())
a = [torch.tensor([1, 2, 3, 1]), torch.tensor([4, 1, 6])]
for i, pos in enumerate(a):
    pos[pos == 1] = 50
print(a)
