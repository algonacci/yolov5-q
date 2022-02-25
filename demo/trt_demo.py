from collections import OrderedDict, namedtuple
from yolov5.data import letterbox
from yolov5.utils.torch_utils import select_device
import torch
import numpy as np
import cv2
import tensorrt as trt  # https://developer.nvidia.com/nvidia-tensorrt-download

w = '/home/laughing/yolov5/weights/pruned_auto_n.engine'
device = select_device('0')

# check_version(trt.__version__, '8.0.0', verbose=True)  # version requirement
Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
logger = trt.Logger(trt.Logger.INFO)
with open(w, 'rb') as f, trt.Runtime(logger) as runtime:
    model = runtime.deserialize_cuda_engine(f.read())
bindings = OrderedDict()
for index in range(model.num_bindings):
    name = model.get_binding_name(index)
    dtype = trt.nptype(model.get_binding_dtype(index))
    shape = tuple(model.get_binding_shape(index))
    data = torch.from_numpy(np.empty(shape, dtype=np.dtype(dtype))).to(device)
    bindings[name] = Binding(name, dtype, shape, data, int(data.data_ptr()))
binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
context = model.create_execution_context()
batch_size = bindings['images'].shape[0]

img = cv2.imread('/e/datasets/贵阳银行/play_phone/guiyang0721/images/train/3_0.jpg')
img = letterbox(img, auto=False)[0]
img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
img = np.ascontiguousarray(img)
img = torch.from_numpy(img).to(device)
img = img.half()
img /= 255  # 0 - 255 to 0.0 - 1.0
if len(img.shape) == 3:
    img = img[None]  # expand for batch dim
assert img.shape == bindings['images'].shape, (img.shape, bindings['images'].shape)
binding_addrs['images'] = int(img.data_ptr())
context.execute_v2(list(binding_addrs.values()))
y = bindings['output'].data
print(y.shape)
