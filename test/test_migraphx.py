import numpy as np
import migraphx


if __name__ == "__main__":
    img = np.loadtxt('/home/laughing/bus.txt')
    img = img.reshape((416, 768, 3))
    img = img / 255.
    img = img.astype(np.float32)

    img = img.transpose(2, 0, 1)[None, :]
    img = np.ascontiguousarray(img)
    # 加载模型

    model = migraphx.parse_onnx("alexnet.onnx")
    inputName=model.get_parameter_names()[0]
    inputShape=model.get_parameter_shapes()[inputName].lens()
    print("inputName:{0} \ninputShape:{1}".format(inputName,inputShape))

    # FP16
    migraphx.quantize_fp16(model)

    # 编译
    model.compile(migraphx.get_target("gpu"))

    # 推理
    results = model.run({inputName: migraphx.argument(img)})
    print(len(results))
    for i, r in enumerate(results):
        np.save(f'{i}.npy', np.array(r.tolist()))
        print("shape: ", r.get_shape())
        print("size: ", r.lens())
        print("elements: ", r.elements())
        print('--------------------')

    # 获取输出节点属性
    result=results[0] # 获取第一个输出节点的数据,migraphx.argument类型
    outputShape=result.get_shape() # 输出节点的shape,migraphx.shape类型
    outputSize=outputShape.lens() # 每一维大小，维度顺序为(N,C,H,W),list类型
    numberOfOutput=outputShape.elements() # 输出节点元素的个数

    # 获取输出结果
    resultData=result.tolist() # 输出数据转换为list
