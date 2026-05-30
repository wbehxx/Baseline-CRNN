import torch
import sys

sys.path.append('/home/mwl/disk_space/PaddleOCR2Pytorch_Core')

try:
    from pytorchocr.modeling.architectures.base_model import BaseModel
except ImportError as e:
    print(f"❌ 导入失败：{e}")
    sys.exit(1)

def transplant_brain():
    print("========== 🧠 开始进行 PP-OCRv4 Server 换头手术 ==========")
    
    # 🎯 完全照抄官方图纸的 Config
    config = {
        'model_type': 'rec', 
        'algorithm': 'SVTR_HGNet', # 算法换成了 HGNet 专属
        'in_channels': 3,    
        'Transform': None,   
        'Backbone': {
            'name': 'PPHGNet_small' # 真正的天花板级主干网络
        }, 
        'Neck': None,
        'Head': {
            'name': 'MultiHead', 
            'head_list': [
                {'CTCHead': {
                    'Neck': {'name': 'svtr', 'dims': 120, 'depth': 2, 'hidden_dims': 120, 'kernel_size': [1, 3], 'use_guide': True},
                    'Head': {'fc_decay': 1e-05}
                }},
                {'NRTRHead': {'max_text_length': 25, 'nrtr_dim': 384}} # 官方用的是 NRTR 辅助头
            ],
            # 明确告诉解码器输出维度
            'out_channels_list': {'CTCLabelDecode': 6625, 'NRTRLabelDecode': 6625} 
        } 
    }

    print("正在构建 PP-OCRv4 躯体 (PPHGNet_small + SVTR + MultiHead)...")
    model = BaseModel(config) 

    weight_path = '/home/mwl/disk_space/PaddleOCR2Pytorch_Core/weights/ch_ptocr_v4_rec_server_infer.pth'
    print(f"正在读取 92.3M 预训练记忆: {weight_path}")
    pretrained_dict = torch.load(weight_path, map_location='cpu', weights_only=True)

    model_dict = model.state_dict()

    print("正在进行维度匹配与记忆注入...")
    filtered_dict = {}
    unmatched_pretrained = []
    
    model_keys = list(model_dict.keys())
    
    # 移除了翻译函数，直接硬碰硬匹配！
    for k, v in pretrained_dict.items():
        if k in model_dict and v.shape == model_dict[k].shape:
            filtered_dict[k] = v
        else:
            unmatched_pretrained.append(k)

    unfilled_model_keys = [k for k in model_keys if k not in filtered_dict]

    model_dict.update(filtered_dict)
    model.load_state_dict(model_dict)
    
    print(f"✅ 成功注入层数: {len(filtered_dict)}")
    print(f"❌ 预训练字典中被抛弃的层数: {len(unmatched_pretrained)}")
    print(f"⚠️ 空壳躯体中处于随机初始化的层数: {len(unfilled_model_keys)}")
    
    if len(unmatched_pretrained) > 0:
        print("\n--- 🕵️‍♂️ 关键情报：前 5 个被抛弃的预训练层名字 ---")
        for k in unmatched_pretrained[:5]: print(k)

if __name__ == "__main__":
    transplant_brain()




# import torch
# import sys

# sys.path.append('/home/mwl/disk_space/PaddleOCR2Pytorch_Core')

# try:
#     from pytorchocr.modeling.architectures.base_model import BaseModel
# except ImportError as e:
#     print(f"❌ 导入失败：{e}")
#     sys.exit(1)

# def translate_keys(pretrained_dict):
#     """ 🧬 智能基因重组：将预训练权重的 'stem' 翻译为空壳认识的 'conv1_1' """
#     mapped_dict = {}
#     for k, v in pretrained_dict.items():
#         new_k = k
#         # 修复 ResNet 头部命名差异
#         if 'backbone.stem.0.conv' in k: new_k = k.replace('backbone.stem.0.conv', 'backbone.conv1_1._conv')
#         elif 'backbone.stem.0.bn' in k: new_k = k.replace('backbone.stem.0.bn', 'backbone.conv1_1._batch_norm')
#         elif 'backbone.stem.1.conv' in k: new_k = k.replace('backbone.stem.1.conv', 'backbone.conv1_2._conv')
#         elif 'backbone.stem.1.bn' in k: new_k = k.replace('backbone.stem.1.bn', 'backbone.conv1_2._batch_norm')
#         elif 'backbone.stem.2.conv' in k: new_k = k.replace('backbone.stem.2.conv', 'backbone.conv1_3._conv')
#         elif 'backbone.stem.2.bn' in k: new_k = k.replace('backbone.stem.2.bn', 'backbone.conv1_3._batch_norm')
        
#         mapped_dict[new_k] = v
#     return mapped_dict

# def transplant_brain():
#     print("========== 🧠 开始进行 PP-OCRv4 Server 换头手术 ==========")
    
#     # 🎯 终极完美 Config (适配 MultiHead 架构)
#     config = {
#         'model_type': 'rec', 
#         'in_channels': 3,    
#         'Transform': None,   
#         'Backbone': {
#             'name': 'ResNet', 
#             'layers': 34 
#         }, 
#         'Neck': None,  # 脖子置空，让头来接管
#         'Head': {
#             'name': 'MultiHead', 
#             'head_list': [
#                 {'CTCHead': {
#                     'Neck': {'name': 'svtr', 'dims': 120, 'depth': 2, 'hidden_dims': 120, 'use_guide': True}
#                 }},
#                 {'SARHead': {'enc_mac_iters': 1, 'max_text_length': 25}}
#             ],
#             # 👇 【最终修改】把键名换成解码器的名字
#             'out_channels_list': {'CTCLabelDecode': 6625, 'SARLabelDecode': 6625} 
#         }
#     }

#     print("正在构建 PP-OCRv4 躯体 (ResNet34 + MultiHead)...")
#     model = BaseModel(config) 

#     weight_path = '/home/mwl/disk_space/PaddleOCR2Pytorch_Core/weights/ch_ptocr_v4_rec_server_infer.pth'
#     print(f"正在读取 92.3M 预训练记忆: {weight_path}")
#     # 加上 weights_only=True 消除安全警告
#     pretrained_dict = torch.load(weight_path, map_location='cpu', weights_only=True)

#     print("正在翻译记忆字典 (处理命名差异)...")
#     translated_dict = translate_keys(pretrained_dict)

#     model_dict = model.state_dict()

#     print("正在进行维度匹配与记忆注入...")
#     filtered_dict = {}
#     unmatched_pretrained = []
    
#     # 诊断变量
#     model_keys = list(model_dict.keys())
    
#     for k, v in translated_dict.items():
#         if k in model_dict and v.shape == model_dict[k].shape:
#             filtered_dict[k] = v
#         else:
#             unmatched_pretrained.append(k)

#     # 找出空壳躯体里，没有被注入权重的层
#     unfilled_model_keys = [k for k in model_keys if k not in filtered_dict]

#     model_dict.update(filtered_dict)
#     model.load_state_dict(model_dict)
    
#     print(f"✅ 成功注入层数: {len(filtered_dict)}")
#     print(f"❌ 预训练字典中被抛弃的层数: {len(unmatched_pretrained)}")
#     print(f"⚠️ 空壳躯体中处于随机初始化的层数: {len(unfilled_model_keys)}")
    
#     print("\n--- 🕵️‍♂️ 关键情报：前 10 个被抛弃的预训练层名字 ---")
#     for k in unmatched_pretrained[:10]: print(k)
        
#     print("\n--- 🕵️‍♂️ 关键情报：前 10 个等待注入的空壳层名字 ---")
#     for k in unfilled_model_keys[:10]: print(k)

# if __name__ == "__main__":
#     transplant_brain()






# import torch
# import sys

# # 1. 把仓库根目录加入环境变量
# # 确保这个路径指向你刚刚 clone 下来并改名的 PaddleOCR2Pytorch_Core 文件夹
# sys.path.append('/home/mwl/disk_space/PaddleOCR2Pytorch_Core')

# try:
#     # 2. 【核心修改】根据你的目录结构，从 pytorchocr 导入 BaseModel
#     from pytorchocr.modeling.architectures.base_model import BaseModel
# except ImportError as e:
#     print(f"❌ 导入失败，请检查 sys.path 路径！报错信息：{e}")
#     sys.exit(1)

# def transplant_brain():
#     print("========== 🧠 开始进行 PP-OCRv4 Server 换头手术 ==========")
    
#     # 3. 构造 v4 server 版的基础架构配置
#     # 注意：这里的 config 键值对是针对 PaddleOCR 体系设计的
#     config = {
#         'model_type': 'rec', # 必须有，因为 base_model.py 第 26 行用到了它
#         'in_channels': 3,    # 预训练模型通常是接收 3 通道 RGB 图像
#         'Transform': None,   # 识别模型通常不需要在这个层面做 Transform
#         'Backbone': {
#             'name': 'ResNet', 
#             'layers': 34 # 对应 Server 版的主干网络
#         }, 
#         'Neck': {
#             'name': 'SequenceEncoder', 
#             'encoder_type': 'svtr', 
#             # 'hidden_drop_rate': 0.1
#         },
#         'Head': {
#             'name': 'CTCHead', 
#             'fc_decay': 0.00001, 
#             'out_channels': 6625 # 百度原版的字典大小 (含空白符)
#         } 
#     }

#     print("正在构建 PP-OCRv4 躯体 (ResNet34 + SVTR)...")
#     try:
#         # 使用 BaseModel 来实例化
#         model = BaseModel(config) 
#     except Exception as e:
#         print(f"❌ 构建躯体失败，请提供以下报错信息给我以便修正 config：\n{e}")
#         return

#     # 4. 指向你存放权重的真实绝对路径
#     # ⚠️ 请确保你的权重文件名为 ch_ptocr_v4_rec_server_infer.pth，并且路径正确
#     weight_path = '/home/mwl/disk_space/PaddleOCR2Pytorch_Core/weights/ch_ptocr_v4_rec_server_infer.pth'
#     print(f"正在读取 92.3M 预训练记忆: {weight_path}")
    
#     try:
#         pretrained_dict = torch.load(weight_path, map_location='cpu')
#     except Exception as e:
#          print(f"❌ 读取权重失败，检查文件是否存在或损坏：\n{e}")
#          return

#     model_dict = model.state_dict()

#     print("正在进行维度匹配与记忆注入...")
#     # 5. 【极其核心的过滤】：剔除掉最后一层分类头 (Linear 层)
#     filtered_dict = {
#         k: v for k, v in pretrained_dict.items() 
#         if k in model_dict and v.shape == model_dict[k].shape
#     }

#     # 如果 filtered_dict 为空，说明预训练权重的层命名和 BaseModel 构建的完全对不上
#     if not filtered_dict:
#         print("❌ 灾难性失败：没有匹配到任何层！说明预训练权重的字典 key 和我们构建的架构完全不同。")
#         print("-> 预训练字典的一个示例 key:", list(pretrained_dict.keys())[0])
#         print("-> 我们模型字典的一个示例 key:", list(model_dict.keys())[0])
#         return

#     # 6. 将过滤后的聪明大脑注入躯体
#     model_dict.update(filtered_dict)
#     model.load_state_dict(model_dict)
    
#     print(f"✅ 完美融合！共成功注入了 {len(filtered_dict)} 层特征提取权重。")
#     print(f"⚠️ 预训练文件总层数: {len(pretrained_dict)} | 当前躯体总层数: {len(model_dict)}")

# if __name__ == "__main__":
#     transplant_brain()