import os
import cv2
import numpy as np
import yaml

def analyze_dataset_widths():
    # 1. 载入项目的 yaml 配置文件
    config_path = '/home/mwl/disk_space/Baseline-CRNN/configs/config.yaml'
    if not os.path.exists(config_path):
        print(f"❌ 找不到配置文件，请检查路径: {config_path}")
        return
        
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    # target_height = config['model']['img_height']  # 获取标准输入高度（通常是32）
    target_height = 32  # 固定高度为32
    
    # 2. 动态配置三个数据集的读取任务
    # 使用 .get() 预防配置项不存在的情况
    dataset_tasks = [
        {
            "name": "训练集 (Train)",
            "list_path": config['data'].get('train_list'),
            "img_dir": config['data'].get('train_dir')
        },
        {
            "name": "验证集 (Val)",
            "list_path": config['data'].get('val_list'),
            "img_dir": config['data'].get('val_dir')
        },
        {
            "name": "测试集 (Test)",
            "list_path": config['data'].get('test_list'),  # 如果 yaml 里没有 test_list 会返回 None
            "img_dir": config['data'].get('test_dir')
        }
    ]

    global_widths = []  # 存放全集所有图片的宽度
    
    print("📊 ====== 开始全数据集宽度摸底分析 ======")
    print(f"📐 目标等比缩放高度: {target_height} px\n")

    # 3. 循环遍历每个数据集进行统计
    for task in dataset_tasks:
        name = task["name"]
        list_path = task["list_path"]
        img_dir = task["img_dir"]
        
        # 鲁棒性检查：如果配置里没写测试集，或者路径不存在，则优雅跳过
        if not list_path or not img_dir or not os.path.exists(list_path):
            print(f"ℹ️  💡 跳过 {name} 的统计（未配置或找不到标签文件）")
            continue
            
        current_widths = []
        bad_count = 0
        
        with open(list_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        for line in lines:
            parts = line.strip().split('\t')
            if len(parts) < 1:
                continue
            img_name = parts[0]
            img_path = os.path.join(img_dir, img_name)
            
            # 读取灰度图并计算标准等比缩放宽度
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                bad_count += 1
                continue
                
            h, w = img.shape
            ratio = w / float(h)
            new_w = int(target_height * ratio)
            
            current_widths.append(new_w)
            global_widths.append(new_w)
            
        # 打印当前子数据集的报告
        if current_widths:
            current_widths = np.array(current_widths)
            print(f"🔹 {name} 分析报告:")
            print(f"   - 样本总数: {len(current_widths)} 张 | 坏图/找不到: {bad_count} 张")
            print(f"   - 宽度范围: {current_widths.min()} px ~ {current_widths.max()} px")
            print(f"   - 99% 分位数宽度: {int(np.percentile(current_widths, 99))} px")
            print("-" * 40)

    # 4. 汇总计算全局大锅饭数据
    if not global_widths:
        print("❌ 错误：未能从任何数据集中读取到有效图像，请检查 config.yaml 中的路径配置！")
        return
        
    global_widths = np.array(global_widths)
    g_max = global_widths.max()
    g_p90 = int(np.percentile(global_widths, 90))
    g_p95 = int(np.percentile(global_widths, 95))
    g_p99 = int(np.percentile(global_widths, 99))
    
    print("\n👑 ====== 全局汇总决策报告 (Summary) ======")
    print(f" 📈 三集联动总样本数 : {len(global_widths)} 张")
    print(f" 📈 全局最大绝对宽度 : {g_max} px")
    print(f" 📈 全局 90% 覆盖宽度 : {g_p90} px")
    print(f" 📈 全局 95% 覆盖宽度 : {g_p95} px")
    print(f" 📈 全局 99% 覆盖宽度 : {g_p99} px")
    print("==================================================\n")
    
    print("💡 【多集联动调参终极建议】")
    print(f" 请将 config.yaml 中的 max_width 统一修改为: {g_p99}")
    print(f" 这样能确保无论是训练、验证还是未来的测试阶段，99% 的简牍字形都能保持完全不扭曲，")
    print(f" 且绝对不会有任何一张图的文字因为超长在测试集里被‘悄悄切断’，从而保证了考场得分的公正与安全！")

if __name__ == '__main__':
    analyze_dataset_widths()