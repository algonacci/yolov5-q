# DDP training
# python -m torch.distributed.launch --nproc_per_node 4  tools/train.py\
# 	--weights ./weights/yolov5l.pt\
# 	--cfg ./configs/segment/yolov5l_seg.yaml\
# 	--data ./data/coco.yaml\
# 	--epochs 300\
# 	--batch-size 32\
# 	--project runs/seg0301\
# 	--name coco_l\
# 	--mask\
# 	--mask-ratio 4\
# 	--device 0,1,2,3\
# 	--workers 32\
# 	--exist-ok


# test
python tools/train.py\
	--weights ./weights/yolov5s.pt\
	--cfg ./configs/segment/yolov5s_seg.yaml\
	--data ./data/balloon.yaml\
	--epochs 50\
	--batch-size 8\
	--project runs/train\
	--name balloon\
	--mask\
	--mask-ratio 4\
	--device 0\
	--workers 8\
	--exist-ok
