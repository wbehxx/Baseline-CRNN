# 这个脚本的作用是打印 PP-OCRv4 简牍识别模型的全景报告，帮助我们深入理解模型结构和参数分布。
from torchinfo import summary
from model_v4 import build_ppocrv4_jiandu
from model_crnn import CRNN
# 1. 实例化你的全新 PP-OCRv4 简牍识别模型
# 传入你转换器里真实的类别数（例如字典字数 + 1）
num_classes = 2243  

# model = build_ppocrv4_jiandu(num_classes=num_classes)

# # 2. 调用 summary 打印全景报告
# # input_size 格式为：(batch_size, channels, height, width)
# # 💡 这里我们输入高度死死守护在 48，宽度给一个标准的 256 像素进行模拟前向传播
# model_stats = summary(
#     model, 
#     input_size=(32, 3, 48, 1024), 
#     device="cpu",               # 在 CPU 上模拟即可，不占显存
#     depth=10,                    # 🔍 纵深控制在 4 层，刚好能看清 PPHGNet 内部的 Block 嵌套
#     col_names=["input_size", "output_size", "num_params", "mult_adds"],
#     row_settings=["var_names"]   # 显示代码中的变量名，方便和源码对齐
# )


model = CRNN(
    img_height=32,   # 👈 经典 CRNN 这里的 H 通常是 32
    nc=1,            # 输入通道数（简牍通常是灰度图，设为 1）
    num_classes=num_classes,
    nh=256           # LSTM 的隐藏层维度
)
model_stats = summary(
    model, 
    # input_size 格式: (BatchSize, Channels, Height, Width)
    # 🛠️ 修复：将 H 从 48 改为 32，C 改为 1（对齐上面 nc=1）
    input_size=(32, 1, 32, 256), 
    device="cpu",               # 在 CPU 上模拟即可
    depth=10,                    # 纵深控制在 4 层，足以看清 CNN->RNN->FC 结构
    col_names=["input_size", "output_size", "num_params", "mult_adds"],
    row_settings=["var_names"]   # 显示变量名
)


# # 这个脚本的作用是将我们构建的 PP-OCRv4 简牍识别模型导出为 ONNX 格式，方便后续在各种推理引擎中部署和优化。
# import torch
# from model_v4 import build_ppocrv4_jiandu

# model = build_ppocrv4_jiandu(num_classes=2243)
# dummy_input = torch.randn(1, 3, 48, 256)

# # 导出一流的通用中间件模型文件 model.onnx
# torch.onnx.export(
#     model, 
#     dummy_input, 
#     "/home/mwl/disk_space/Baseline-CRNN/ppocrv4_model.onnx", 
#     input_names=["images"], 
#     output_names=["logits"],
#     opset_version=12
# )
# print("📦 模型已成功打包导出为 ppocrv4_model.onnx ！")

