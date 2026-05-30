import torch
import torch.nn as nn
import torch.nn.functional as F

class BidirectionalLSTM(nn.Module):
    """
    双向 LSTM 模块
    作用：结合上下文信息（例如通过前后的字来推断当前模糊的字）
    """
    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()
        # bidirectional=True 表示同时捕获从左到右和从右到左的语义序列
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        # 将双向隐藏层的输出维度 (nHidden * 2) 映射到目标维度 nOut
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        # input shape: [seq_length, batch_size, nIn]
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        
        # 将序列展开以应用全连接层
        t_rec = recurrent.view(T * b, h)
        output = self.embedding(t_rec)  # [T * b, nOut]
        
        # 恢复成序列形状
        output = output.view(T, b, -1)
        return output

class CRNN(nn.Module):
    """
    完整的 CRNN 网络模型
    """
    def __init__(self, img_height=32, nc=1, num_classes=2243, nh=256):
        """
        参数:
        img_height: 输入图像的高度 (标准 CRNN 要求输入高度为 32)
        nc: 输入图像的通道数 (简牍灰度图 nc=1)
        num_classes: 字符类别总数 (字典大小 + 1 个 CTC Blank 标签)
        nh: LSTM 的隐藏层维度 (通常设为 256)
        """
        super(CRNN, self).__init__()
        assert img_height % 16 == 0, '输入图像的高度必须是 16 的倍数'

        # 1. CNN 特征提取层 (基于 VGG 架构进行了针对性修改)
        self.cnn = nn.Sequential(
            # Layer 1: 输入 [1, 32, W] -> 输出 [64, 16, W/2]
            nn.Conv2d(nc, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2), 

            # Layer 2: 输出 [128, 8, W/4]
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Layer 3: 维持尺寸 [256, 8, W/4]
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            # Layer 4: 高度减半，宽度不减半 [256, 4, W/4 + 1]
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 1), padding=(0, 1)),

            # Layer 5: 维持尺寸 [512, 4, W/4 + 1]
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            # Layer 6: 高度再次减半，宽度不减半 [512, 2, W/4 + 2]
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 1), padding=(0, 1)),

            # Layer 7: 最终将高度压扁为 1 [512, 1, W/4 + 1]
            nn.Conv2d(512, 512, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )

        # 2. RNN 序列建模层 (两层 BiLSTM)
        # 输入维度是 CNN 提取出的 512 维特征
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, num_classes)
        )

    def forward(self, input):
        """
        前向传播过程
        """
        # 1. 经过 CNN 提取特征
        conv = self.cnn(input)
        
        # 确保输出的高度为 1，否则后续无法当作序列处理
        b, c, h, w = conv.size()
        assert h == 1, "CNN 最后的特征图高度必须为 1"

        # 2. 维度重排，适配 RNN 的输入要求
        # 将 [batch_size, channels, 1, width] 转换为 [width, batch_size, channels]
        conv = conv.squeeze(2)         # 挤掉高度维度 -> [b, c, w]
        conv = conv.permute(2, 0, 1)   # 置换维度 -> [w, b, c]

        # 3. 经过 RNN 预测序列
        output = self.rnn(conv)        # 输出形状 -> [w, b, num_classes]

        # 4. 转换为对数概率 (Log-Softmax)，这是 PyTorch 中 CTCLoss 严格要求的输入格式
        output = F.log_softmax(output, dim=2)

        return output

# ================= 单元测试使用 =================
# 你可以直接运行 python model_crnn.py 来测试网络是否能正常计算
if __name__ == '__main__':
    # 模拟一个 Batch 的简牍图片：Batch_size=4, 灰度图通道数=1, 统一高度=32, 宽度=256
    dummy_input = torch.randn(4, 1, 32, 256)
    
    # 假设你的字典里有 2242 个字，加上 1 个 CTC Blank，总类别数为 2243
    model = CRNN(img_height=32, nc=1, num_classes=2243, nh=256)
    
    # 前向传播
    output = model(dummy_input)
    
    print("输入图像张量形状:", dummy_input.shape)
    print("模型输出张量形状:", output.shape)
    # 期望输出形状: [Sequence_Length(宽度的下采样结果), Batch_Size, Num_Classes]
    # 在 256 的宽度下，经过网络后通常序列长度在 65 左右