import os
import yaml
import torch
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import JianduDataset, jiandu_collate_fn
from model_crnn import CRNN
from utils import CTCLabelConverter, calculate_cer, get_logger

def save_visualization(image_tensor, target_str, pred_str, save_path, font_path, is_correct):
    """
    【新增功能】将原图和预测结果拼接并保存为可视化图片
    is_correct: 布尔值，用于决定在图片上打勾还是打叉
    """
    # 1. 图像张量还原
    img_np = image_tensor.squeeze(0).cpu().numpy()
    # 将 [-1, 1] 的值域还原回 [0, 1]
    img_np = (img_np * 0.5) + 0.5 
    # 截断极小误差，防止越界
    img_np = np.clip(img_np, 0, 1) 
    # 再乘 255 转为正常的图片像素
    img_np = (img_np * 255).astype(np.uint8)
    img_rgb = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    
    # 2. 创建底部白色文本画布 (高度 65)
    h, w = img_rgb.shape[:2]
    text_canvas = np.ones((65, w, 3), dtype=np.uint8) * 255
    
    # 3. 拼接图像
    combined_img = np.vstack((img_rgb, text_canvas))
    pil_img = Image.fromarray(combined_img)
    draw = ImageDraw.Draw(pil_img)
    
    # 4. 加载字体
    try:
        font = ImageFont.truetype(font_path, 20)
    except IOError:
        font = ImageFont.load_default()
        print(f"⚠️ 找不到字体 {font_path}，中文可能变乱码！请上传字体文件。")

    # # 5. 绘制文本和标记
    # # 真实标签固定为绿色
    # draw.text((10, h + 5), f"GT: {target_str}", font=font, fill=(0, 150, 0))
    
    # # 预测标签：对的用蓝色，错的用红色
    # if is_correct:
    #     draw.text((10, h + 35), f"PR: {pred_str} (√)", font=font, fill=(0, 0, 255))
    # else:
    #     draw.text((10, h + 35), f"PR: {pred_str} (×)", font=font, fill=(255, 0, 0))

    # 5. 绘制文本和标记
    # 【新增】将真实的空心方框替换为实心方块，用来和“字体缺失豆腐块”区分
    vis_target = target_str.replace('□', '■')
    vis_pred = pred_str.replace('□', '■')

    # 真实标签固定为绿色 (注意这里换成了 vis_target)
    draw.text((10, h + 5), f"GT: {vis_target}", font=font, fill=(0, 150, 0))
    
    # 预测标签：对的用蓝色，错的用红色 (注意这里换成了 vis_pred)
    if is_correct:
        draw.text((10, h + 35), f"PR: {vis_pred} (√)", font=font, fill=(0, 0, 255))
    else:
        draw.text((10, h + 35), f"PR: {vis_pred} (×)", font=font, fill=(255, 0, 0))
    
    pil_img.save(save_path)


def evaluate():
    # 1. 读取配置
    with open('configs/config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # ================= 初始化日志 =================
    os.makedirs(config['log']['logs_dir'], exist_ok=True)
    log_file = os.path.join(config['log']['logs_dir'], 'eval_crnn.log')
    logger = get_logger(log_file)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"========== 🚀 开始测试评估阶段 ==========")
    logger.info(f"💻 使用设备: {device}")

    # 2. 仅加载测试集 (Test Set)
    logger.info("📦 正在加载测试集...")
    test_dataset = JianduDataset(
        labels_txt_path=config['data']['test_list'],
        img_dir=config['data']['test_dir'],
        dict_path=config['data']['dict_path'],
        img_height=config['model']['img_height'],
        max_width=config['model']['max_width']
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=config['train']['batch_size'], 
        shuffle=False, num_workers=config['train']['num_workers'], 
        collate_fn=jiandu_collate_fn
    )

    # 3. 初始化标签转换器和模型
    converter = CTCLabelConverter(config['data']['dict_path'])
    num_classes = len(converter.char2id) + 1

    model = CRNN(
        img_height=config['model']['img_height'],
        nc=config['model']['nc'],
        num_classes=num_classes,
        nh=config['model']['nh']
    ).to(device)

    # 4. 加载最优权重 (best_model.pth)
    weight_path = os.path.join(config['log']['weights_dir'], 'best_model.pth')
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"找不到权重文件 {weight_path}，请确保已经跑完训练环节！")
        
    model.load_state_dict(torch.load(weight_path, map_location=device))
    logger.info(f"✅ 成功加载最优权重: {weight_path}")

    # ================= 【新增】可视化目录与字体设置 =================
    base_vis_dir = os.path.join(config['log']['logs_dir'], 'vis_results')
    correct_dir = os.path.join(base_vis_dir, 'correct_cases')
    error_dir = os.path.join(base_vis_dir, 'error_cases')
    
    os.makedirs(correct_dir, exist_ok=True)
    os.makedirs(error_dir, exist_ok=True)
    logger.info(f"📁 已就绪可视化保存目录: \n  - {correct_dir}\n  - {error_dir}")
    
    # 【注意】请确保项目根目录下有这个字体文件！
    font_path = "/home/mwl/disk_space/Baseline-CRNN/data/raw_deepjiandu/DeepJiandu/NotoSerifTC-Regular.ttf" 
    # ===============================================================

    # 5. 开始评估
    model.eval()
    all_preds_str = []
    all_targets_str = []
    correct_lines = 0  # 用于计算行准确率
    bad_cases = []     # 收集预测错误的样本用于分析
    
    # 记录可视化保存数量
    correct_count = 0
    error_count = 0

    logger.info("⏳ 正在测试集中进行推理计算并生成可视化图...")
    with torch.no_grad():
        for images, targets, target_lengths in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            targets = targets.to(device)
            
            # 前向传播
            preds = model(images)
            batch_size = images.size(0)
            input_lengths = torch.IntTensor([preds.size(0)] * batch_size)
            
            # 提取最大概率索引并解码
            _, preds_index = preds.max(2)
            preds_index = preds_index.transpose(1, 0).contiguous()
            preds_str = converter.decode(preds_index.cpu(), input_lengths)
            all_preds_str.extend(preds_str)
            
            # 还原真实标签
            start = 0
            for i, length in enumerate(target_lengths):
                t = targets[start:start+length]
                target_str = ''.join([converter.id2char[int(x)] for x in t])
                all_targets_str.append(target_str)
                start += length
                
                # 统计行准确率与收集错误样本，并进行可视化保存
                is_correct = (preds_str[i] == target_str)
                if is_correct:
                    correct_lines += 1
                    # 保存预测正确的图片
                    save_name = os.path.join(correct_dir, f"correct_{correct_count}.jpg")
                    save_visualization(images[i], target_str, preds_str[i], save_name, font_path, True)
                    correct_count += 1
                else:
                    bad_cases.append((target_str, preds_str[i]))
                    # 保存预测错误的图片
                    save_name = os.path.join(error_dir, f"error_{error_count}.jpg")
                    save_visualization(images[i], target_str, preds_str[i], save_name, font_path, False)
                    error_count += 1

    # 6. 计算最终指标
    total_samples = len(all_targets_str)
    final_cer = calculate_cer(all_preds_str, all_targets_str)
    line_accuracy = correct_lines / total_samples

    # 7. 打印学术级评估报告
    logger.info("\n" + "="*40)
    logger.info("🏆 最终测试集评估报告 (Test Report)")
    logger.info("="*40)
    logger.info(f"总测试样本数 : {total_samples} 行简牍文本")
    logger.info(f"字符错误率 (CER)   : {final_cer * 100:.2f}%  <-- 越低越好，论文核心指标")
    logger.info(f"行准确率 (Line Acc): {line_accuracy * 100:.2f}%  <-- 极其严苛的完全匹配率")
    logger.info("="*40)

    # 8. 错误样本分析 (Bad Case Analysis)
    if bad_cases:
        logger.info("\n🔍 错误样本抽查 (Bad Cases Analysis):")
        logger.info("这能帮你分析模型到底在哪种字上容易栽跟头：")
        # 最多展示 5 个错误样本防止刷屏
        for i, (truth, pred) in enumerate(bad_cases[:5]):
            logger.info(f"  [样本 {i+1}]")
            logger.info(f"  - 真实标签 (Ground Truth): {truth}")
            logger.info(f"  - 模型预测 (Prediction)  : {pred}")
            logger.info(f"  -------------------------")
            
    # 【新增】可视化保存报告
    logger.info(f"\n📸 可视化结果已全量保存！")
    logger.info(f"  - 完美预测的样本: {correct_count} 张 -> 存在 {correct_dir}")
    logger.info(f"  - 存在错误的样本: {error_count} 张 -> 存在 {error_dir}")

if __name__ == '__main__':
    evaluate()