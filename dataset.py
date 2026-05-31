import torch
from torch.utils.data import Dataset, DataLoader , Sampler
import cv2
import numpy as np
import os
from torchvision import transforms as T  # 新增
from PIL import Image                    # 新增
import random

class JianduDataset(Dataset):
    def __init__(self, labels_txt_path, img_dir, dict_path, img_height=32, max_width=512, is_train=False):
        """
        labels_txt_path: process_to_sequence.py 生成的 txxt 文件路径
        img_dir: 截取后的序列图片存放文件夹
        dict_path: 字典 dict.txt 路径
        img_height: CRNN 模型的标准输入高度
        max_width: 图片的最大宽度，超过则截断，不足则在 batch 阶段 pad
        """
        
        super().__init__()
        self.img_dir = img_dir
        self.img_height = img_height
        self.max_width = max_width
        # self.is_train = is_train  # 记录当前数据集是否用于训练
        self.is_train = is_train
        
        # 加载字典，0 保留给 CTC 的 Blank 标签
        self.char2id = {}
        self.id2char = {}
        with open(dict_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                char = line.strip()
                self.char2id[char] = idx + 1  # ID 从 1 开始
                self.id2char[idx + 1] = char
                
        self.num_classes = len(self.char2id) + 1 # +1 for Blank
        
        # 加载数据列表
        self.data_list = []
        with open(labels_txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    self.data_list.append((parts[0], parts[1]))

        # ================= 【新增】定义数据增强流水线 =================
        if self.is_train:
            # 第一波：针对像素值的变换 (需要 PIL Image 格式)
            self.color_blur_transforms = T.Compose([
                # 1. 亮度与对比度抖动 (Color Jitter)
                # 亮度上下浮动 40%，对比度上下浮动 40%
                T.ColorJitter(brightness=0.4, contrast=0.4),
                
                # 2. 高斯模糊 (Gaussian Blur)
                # p=0.4 表示有 40% 的概率触发模糊。
                # 【注意】因为图片高度只有 32，kernel_size 只能设很小 (3x3)，否则字就完全糊没了
                T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.1),

            # 👇 新增：几何仿射变换 (神级增强)
                # degrees=2: 随机旋转 -2 到 2 度（模拟竹简放歪了）
                # translate=(0.01, 0.02): 宽和高分别随机平移 1%~2%（模拟坐标漂移）
                # scale=(0.95, 1.05): 随机缩放 95% 到 105%
                # shear=2: 随机剪切扭曲 2 度（模拟视角倾斜）
                # fill=0: 变换后露出的黑边用纯黑色填充
                T.RandomAffine(degrees=2, translate=(0.01, 0.02), scale=(0.95, 1.05), shear=2, fill=0)
            ])
            
            # 第二波：针对张量的变换 - 随机遮挡 (Random Erasing)
            # scale=(0.02, 0.08) 表示遮挡块面积占全图的 2% 到 8%
            # value='random' 会填入随机的彩色/灰色雪花噪点，完美模拟泥土覆盖
            # self.erasing = T.RandomErasing(p=0.5, scale=(0.02, 0.08), ratio=(0.3, 3.3), value='random')

            # 修改点：p 降为 0.3，scale 缩小，value=0 代表用纯黑色块遮挡（模拟墨迹破坏或竹简断裂）
            self.erasing = T.RandomErasing(p=0.3, scale=(0.01, 0.05), ratio=(0.3, 3.3), value=0)
        # ==============================================================
                    
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        img_name, text = self.data_list[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        # 1. 读取单通道灰度图
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        # 修复：加入可视化警告，防止静默流产
        if img is None:
            print(f"⚠️ [数据异常] 图片读取失败，请检查路径或文件是否损坏: {img_path}")
            # 如果是第0张图本身碎了，避免无限递归，直接抛出断言
            if idx == 0:
                raise FileNotFoundError(f"💥 致命错误：数据集第一张图片就无法读取，请检查配置！")
            return self.__getitem__(0)
        # if img is None:
        #     # 容错处理：如果图片损坏，返回第一张图
        #     return self.__getitem__(0)
            
        # 2. Resize 保持长宽比
        h, w = img.shape
        ratio = w / float(h)
        new_w = int(self.img_height * ratio)
        # new_w = min(new_w, self.max_width) # 限制最大长度防止显存爆炸
        
        # 🛠️ 【核心修复】：如果是属于那 1% 的超长怪物简牍，直接抛弃它，转去读第一张图
        if new_w > self.max_width:
            # 1%的样本会超长，我们直接放弃它，转去读随机一张图
            random_idx = random.randint(0, len(self.data_list) - 1)
            return self.__getitem__(random_idx) # 递归调用，随机顶替
        
        # 正常样本继续舒展缩放，字形绝不变形
        img = cv2.resize(img, (new_w, self.img_height))

        # ================= 【新增】应用第一波增强 (色彩与模糊) =================
        if self.is_train:
            # torchvision 的 ColorJitter 需要 PIL 图像
            img_pil = Image.fromarray(img)
            img_pil = self.color_blur_transforms(img_pil)
            img = np.array(img_pil)  # 变回 numpy 数组继续后续处理
        # ===================================================================
        
        # 3. 归一化到 [-1, 1] 区间
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        
        # 4. 文本转 ID 序列
        target_ids = []
        for char in text:
            if char in self.char2id:
                target_ids.append(self.char2id[char])
                
        target_tensor = torch.IntTensor(target_ids)

        # ================= 【修复】正确的张量级数据增强 =================
        if self.is_train:
            # RandomErasing 必须作用在 Tensor 上，且需要 Channel 维度 (1, H, W)
            image_tensor = torch.from_numpy(img).unsqueeze(0).float()
            
            # 执行随机擦除
            image_tensor = self.erasing(image_tensor)
            
            # 为了兼容你原有的 collate_fn，把增强后的 Tensor 再变回二维 numpy 数组
            img = image_tensor.squeeze(0).numpy()
        # ====================================================================

        # 【修复】返回原始的 img 和 target_tensor，让 collate_fn 去处理长度拼接
        return img, target_tensor


def jiandu_collate_fn(batch):
    """
    DataLoader 的后处理函数：用于处理一个 batch 中不同宽度的图片和不同长度的标签。
    因为序列网络要求同一个 batch 里的 tensor 形状必须一致。
    """
    images, labels = zip(*batch)
    
    # 找到这个 batch 中的最大宽度
    max_w = max([img.shape[1] for img in images])

    # 【修改 1】：把通道数从 1 改成 3
    batch_images = torch.zeros(len(images), 3, images[0].shape[0], max_w)
    
    # # 构建 Batch 图片的 Tensor [batch_size, 1, height, max_w]
    # batch_images = torch.zeros(len(images), 1, images[0].shape[0], max_w)
    
    for i, img in enumerate(images):
        w = img.shape[1]
        img_tensor = torch.from_numpy(img)
        # 【修改 2】：把单通道 Tensor 复制成 3 通道 (3, H, W)
        batch_images[i, :, :, :w] = img_tensor.unsqueeze(0).repeat(3, 1, 1)
        
        # batch_images[i, 0, :, :w] = torch.from_numpy(img)
        # # 右侧没赋值的地方自然就是 zeros，完成了 Padding 操作
        
    # 计算 CTC Loss 需要的长度信息
    target_lengths = torch.IntTensor([len(t) for t in labels])
    
    # PyTorch 的 CTCLoss 要求 targets 是一维拼接的张量
    targets = torch.cat(labels, 0)
    
    return batch_images, targets, target_lengths

# ================= 单元测试使用 =================
# 如果直接运行此文件，可以快速测试数据集加载是否正常
if __name__ == '__main__':
    # 确保你已经跑过 process_to_sequence.py
    test_dataset = JianduDataset(
        labels_txt_path='/home/mwl/disk_space/Baseline-CRNN/data/processed_sequence/train_labels.txt',
        img_dir='/home/mwl/disk_space/Baseline-CRNN/data/processed_sequence/train',
        dict_path='/home/mwl/disk_space/Baseline-CRNN/data/processed_sequence/dict.txt'
    )
    
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True, collate_fn=jiandu_collate_fn)
    
    for imgs, targets, target_lengths in test_loader:
        print("Batch Images Shape:", imgs.shape) # 预期: [4, 1, 32, max_w]
        print("Targets Shape (concatenated):", targets.shape)
        print("Target Lengths:", target_lengths)
        break # 只测试第一个 batch

# 分桶采样器：根据图片宽度分组，保证同一 batch 内的图片宽度相近，减少 padding 的浪费
class LengthGroupedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # 1. 统计 dataset 中所有图片按当前高度缩放后的真实宽度
        self.group_ids = []
        for idx in range(len(dataset)):
            # 🛠️ 核心修复：从 data_list 取出的是纯文件名，必须拼接上完整的 img_dir 前缀！
            img_name = dataset.data_list[idx][0]
            img_path = os.path.join(dataset.img_dir, img_name)
            # 读取灰度图的宽高（使用灰度图速度更快）
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            # 🛡️ 鲁棒性防爆护盾：万一真有破损图片，给一个安全的默认宽度，绝对不能卡死
            if img is None:
                print(f"⚠️ [Sampler 警告] 采样器读取图片失败，请检查路径: {img_path}")
                new_w = dataset.max_width  # 坏图直接赋予最大宽度兜底
            else:
                h, w = img.shape[:2]
                ratio = w / float(h)
                new_w = int(dataset.img_height * ratio)
            # 记录这个样本的预测宽度
            self.group_ids.append(new_w)
            
        # 2. 将所有样本的索引按照宽度从短到长进行排序
        self.sorted_indices = sorted(range(len(self.group_ids)), key=lambda x: self.group_ids[x])
        
        # 3. 把排序后的索引，按照 batch_size 切分成一个一个的桶 (Batches)
        self.batches = [self.sorted_indices[i:i + batch_size] 
                        for i in range(0, len(self.sorted_indices), batch_size)]
        
        # 如果最后一个 batch 凑不够人数，丢弃它，防止算梯度时形状不稳
        if self.batches and len(self.batches[-1]) < batch_size:
            self.batches.pop()
    
    def __iter__(self):
        # 4. 如果开启 shuffle，我们只打乱“桶的顺序”，绝不打乱“桶内部的顺序”
        if self.shuffle:
            # 拷贝一份避免破坏原始成员变量
            temp_batches = list(self.batches)
            random.shuffle(temp_batches)
            for batch in temp_batches:
                yield batch
        else:
            for batch in self.batches:
                yield batch

    def __len__(self):
        return len(self.batches)