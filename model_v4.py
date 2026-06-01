import torch
import sys
import types 

# 挂载外部仓库源码
sys.path.append('/mnt/proj/PaddleOCR2Pytorch')
# sys.path.append('/home/mwl/disk_space/PaddleOCR2Pytorch_Core')
from pytorchocr.modeling.architectures.base_model import BaseModel

def build_ppocrv4_jiandu(num_classes, weight_path='/mnt/proj/PaddleOCR2Pytorch/weights/ch_ptocr_v4_rec_server_infer.pth'):
# def build_ppocrv4_jiandu(num_classes, weight_path='/home/mwl/disk_space/PaddleOCR2Pytorch_Core/weights/ch_ptocr_v4_rec_server_infer.pth'):
    # 🎯 恢复双头配置，骗过 len(self.head_list) >= 2 的断言
    config = {
        'model_type': 'rec', 
        'algorithm': 'SVTR_HGNet', 
        'in_channels': 3,    
        'Transform': None,   
        'Backbone': {'name': 'PPHGNet_small'}, 
        'Neck': None,
        'Head': {
            'name': 'MultiHead', 
            'head_list': [
                {'CTCHead': {
                    'Neck': {'name': 'svtr', 'dims': 120, 'depth': 2, 'hidden_dims': 120, 'kernel_size': [1, 3], 'use_guide': True},
                    'Head': {'fc_decay': 1e-05}
                }},
                # 🛠️ 把 NRTRHead 加回来，满足仓库的断言要求
                {'NRTRHead': {'max_text_length': 25, 'nrtr_dim': 384}} 
            ],
            'out_channels_list': {'CTCLabelDecode': num_classes, 'NRTRLabelDecode': num_classes} 
        } 
    }

    model = BaseModel(config) 
    pretrained_dict = torch.load(weight_path, map_location='cpu') #为了适配pytorch1.12去掉了新版本的参数
    # pretrained_dict = torch.load(weight_path, map_location='cpu', weights_only=True)
    model_dict = model.state_dict()
    
    # 完美对齐注入
    filtered_dict = {
        k: v for k, v in pretrained_dict.items() 
        if k in model_dict and v.shape == model_dict[k].shape
    }

    model_dict.update(filtered_dict)
    model.load_state_dict(model_dict)
    
    # ==========================================
    # 🐒 猴子补丁 (Monkey Patch) 核心防御
    # ==========================================
    def safe_multihead_forward(self, x, *args, **kwargs):
        # 我们只走 CTC 路径，彻底绕过源码中那个会报错的 sar_head 调用
        ctc_feat = self.ctc_encoder(x)
        ctc_out = self.ctc_head(ctc_feat)
        return {'CTC': ctc_out}
    
    # 🛠️ 核心修复：通过 type(model.head) 抓取它的父类，直接在类级别重写 forward
    # 坚决不用 types.MethodType，利用 Python 的原生描述符实现运行时动态 self 绑定
    type(model.head).forward = safe_multihead_forward
    # # 强行替换掉那个有 Bug 的 forward
    # model.head.forward = types.MethodType(safe_multihead_forward, model.head)
    # ==========================================

    print(f"🔥 PP-OCRv4 引擎已就绪！成功加载层: {len(filtered_dict)}/{len(pretrained_dict)}。")
    return model