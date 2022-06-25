dep_mul = 1.0
wid_mul = 1.0
anchors = [
    [10, 13, 16, 30, 33, 23],
    [30, 61, 62, 45, 59, 119],
    [116, 90, 156, 198, 373, 326],
]

img_size = (640, 640)

# model settings
model = dict(
    type="YOLOV5",
    backbone=dict(type="CSPDarknet", dep_mul=dep_mul, wid_mul=wid_mul),
    neck=dict(
        type="YOLOPAFPN",
        in_channels=[int(256 * wid_mul), int(512 * wid_mul), int(1024 * wid_mul)],
        num_csp_blocks=round(3 * dep_mul),
    ),
    head=dict(
        type="YOLOV5Head",
        num_classes=80,
        anchors=anchors,
        strides=[8, 16, 32],
        in_channels=[int(256 * wid_mul), int(512 * wid_mul), int(1024 * wid_mul)],
    ),
)
