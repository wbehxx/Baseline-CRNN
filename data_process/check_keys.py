import torch

# 顺便加上 weights_only=True，消除那个烦人的警告
weight_path = '/home/mwl/disk_space/PaddleOCR2Pytorch_Core/weights/ch_ptocr_v4_rec_server_infer.pth'
ckpt = torch.load(weight_path, map_location='cpu', weights_only=True)

keys = list(ckpt.keys())

print("========== 🧠 预训练大脑 CT 扫描结果 ==========")
print(f"总层数: {len(keys)}\n")

print("【Backbone 头部的 15 个关键部位】:")
print("\n".join(keys[:15]))

print("\n【Neck / Head 尾部的 15 个关键部位】:")
print("\n".join(keys[-15:]))