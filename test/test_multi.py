import torch
import time
from functools import partial

def multi_apply(func, *args, **kwargs):
    """Apply function to a list of arguments.

    Note:
        This function applies the ``func`` to multiple inputs and
        map the multiple outputs of the ``func`` into different
        list. Each list contains the same type of outputs corresponding
        to different inputs.

    Args:
        func (Function): A function that will be applied to a list of
            arguments

    Returns:
        tuple(list): A tuple containing multiple list, each list contains \
            a kind of returned results by the function
    """
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map_results)
    # return tuple(map(list, zip(*map_results)))

def func(p, pre):
    return p @ pre  # (80, 80)


proto_out = torch.randn((300, 80, 80, 32))
pred = torch.randn((300, 32))

result = proto_out @ pred.T
print(result.shape)

t1 = time.time()
for p, pre in zip(proto_out, pred):
    result = p @ pre
t2 = time.time()

results = multi_apply(func, proto_out, pred)
t3 = time.time()
print(t2 - t1, t3 - t2)

print(len(results))
print(results[0].shape)
print(len(results[0]))
print(len(results[0][0]))

final = torch.stack(results, dim=0)
print(final.shape)
