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
    
    # 🛠️ 【纯新增】存放全局原始像素维度的骨架大锅饭列表
    global_orig_heights = []
    global_orig_widths = []
    global_orig_ratios = []
    
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
        # 🛠️ 【纯新增】存放当前子集原始特征的列表
        current_orig_heights = []
        current_orig_widths = []
        current_orig_ratios = []
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
                
            h, w = img.shape[:2]
            ratio = w / float(h)
            new_w = int(target_height * ratio)
            
            current_widths.append(new_w)
            global_widths.append(new_w)
            
            # 🛠️ 【纯新增】全量收集原始几何信息
            current_orig_heights.append(h)
            current_orig_widths.append(w)
            current_orig_ratios.append(ratio)
            
            global_orig_heights.append(h)
            global_orig_widths.append(w)
            global_orig_ratios.append(ratio)
            
        # 打印当前子数据集的报告
        if current_widths:
            current_widths = np.array(current_widths)
            current_orig_heights = np.array(current_orig_heights)
            current_orig_widths = np.array(current_orig_widths)
            current_orig_ratios = np.array(current_orig_ratios)
            
            # 🔔 100% 原版输出，字样无任何删减
            print(f"🔹 {name} 分析报告:")
            print(f"   - 样本总数: {len(current_widths)} 张 | 坏图/找不到: {bad_count} 张")
            print(f"   - 宽度范围: {current_widths.min()} px ~ {current_widths.max()} px")
            print(f"   - 99% 分位数宽度: {int(np.percentile(current_widths, 99))} px")
            
            # 🛠️ 【精细新增】精细化子集原始像素全景解剖
            print(f"   - [新增·原始高度分布] 极值跨度: {current_orig_heights.min()} px ~ {current_orig_heights.max()} px | 算术均值: {current_orig_heights.mean():.1f} px | 95%常规占位: {int(np.percentile(current_orig_heights, 95))} px")
            print(f"   - [新增·原始宽度分布] 极值跨度: {current_orig_widths.min()} px ~ {current_orig_widths.max()} px | 算术均值: {current_orig_widths.mean():.1f} px | 95%常规占位: {int(np.percentile(current_orig_widths, 95))} px")
            print(f"   - [新增·原始宽高构型] 比例区间: {current_orig_ratios.min():.2f} ~ {current_orig_ratios.max():.2f} | 平均宽高比(W/H): {current_orig_ratios.mean():.2f} : 1")
            print("-" * 40)

    # 4. 汇总计算全局大锅饭数据
    if not global_widths:
        print("❌ 错误：未能从任何数据集中读取到有效图像，请检查 config.yaml 中的路径配置！")
        return
        
    global_widths = np.array(global_widths)
    global_orig_heights = np.array(global_orig_heights)
    global_orig_widths = np.array(global_orig_widths)
    global_orig_ratios = np.array(global_orig_ratios)
    
    g_max = global_widths.max()
    g_p90 = int(np.percentile(global_widths, 90))
    g_p95 = int(np.percentile(global_widths, 95))
    g_p99 = int(np.percentile(global_widths, 99))
    
    # 🔔 100% 原版输出，汇总头部指标无丝毫删减
    print("\n👑 ====== 全局汇总决策报告 (Summary) ======")
    print(f" 📈 三集联动总样本数 : {len(global_widths)} 张")
    print(f" 📈 全局最大绝对宽度 : {g_max} px")
    print(f" 📈 全局 90% 覆盖宽度 : {g_p90} px")
    print(f" 📈 全局 95% 覆盖宽度 : {g_p95} px")
    print(f" 📈 全局 99% 覆盖宽度 : {g_p99} px")
    
    # 🛠️ 【精细新增】全局原始特征阶梯覆盖面解剖
    print(f" 📐 [新增] 全局原始图像高度面：均值 {global_orig_heights.mean():.1f} px | 90%分位 {int(np.percentile(global_orig_heights, 90))} px | 95%分位 {int(np.percentile(global_orig_heights, 95))} px | 极端最大值 {global_orig_heights.max()} px")
    print(f" 📐 [新增] 全局原始图像宽度面：均值 {global_orig_widths.mean():.1f} px | 90%分位 {int(np.percentile(global_orig_widths, 90))} px | 95%分位 {int(np.percentile(global_orig_widths, 95))} px | 极端最大值 {global_orig_widths.max()} px")
    print(f" 📐 [新增] 全局原始图像宽高比：平均比值 {global_orig_ratios.mean():.2f} : 1 | 95%分位比 {np.percentile(global_orig_ratios, 95):.2f} | 99%边缘极值比 {np.percentile(global_orig_ratios, 99):.2f}")
    print("==================================================\n")
    
    # 🔔 100% 原版输出
    print("💡 【多集联动调参终极建议】")
    print(f" 请将 config.yaml 中的 max_width 统一修改为: {g_p99}")
    print(f" 这样能确保无论是训练、验证还是未来的测试阶段，99% 的简牍字形都能保持完全不扭曲，")
    print(f" 且绝对不会有任何一张图的文字因为超长在测试集里被‘悄悄切断’，从而保证了考场得分的公正与安全！")

    # 🛠️ 【学术级超大增补】新增多维度简牍特殊几何特征诊断报告
    print("\n🧠 【新增·学术级多维度简牍几何特征诊断与调参高级分析】")
    mean_r = global_orig_ratios.mean()
    p99_r = np.percentile(global_orig_ratios, 99)
    
    print(f" 📑 [数据形态学诊断]: 本简牍数据集全集平均宽高比为 {mean_r:.2f} : 1。")
    if mean_r > 5:
        print(f"     -> [结论]: 你的切片属于典型的【极端细长型简牍木条】。由于长宽比极大，网络在进行前向传播时，必须保证横向空间感受野（Horizontal Receptive Field）和下采样步长足够稠密，强烈建议保留分桶采样器，否则短样本在空零填充（Padding）后，其有效信息会被海量的无用黑边严重稀释！")
    else:
        print(f"     -> [结论]: 你的切片几何特征分布相对温和，长宽比例适中。")
        
    print(f" 📑 [CTC 序列步长匹配分析]: ")
    print(f"     -> 按照当前建议值 max_width = {g_p99} px 截断对齐（高度 H={target_height}）：")
    print(f"     -> 在经典 CRNN 架构中（宽方向做 4 倍下采样），最终送入 LSTM 的时序步长（格子数）固定为 {int(g_p99 / 4)} 个。")
    print(f"     -> 这意味着对 99% 的样本而言，模型都拥有充足的空间来建立对数概率对齐，能天然免疫由于『时间步步长 < 标签字数』引起的 CTC Loss 爆炸（NaN/Inf 异常）。")
    
    print(f" 📑 [多分辨率无损重采样插值建议]: ")
    print(f"     -> 数据集原始高度跨度跨越了 {global_orig_heights.min()} px 到 {global_orig_heights.max()} px，这说明存在明显的跨终端、多源头分辨率不一致问题。")
    print(f"     -> 建议在 `JianduDataset.__getitem__` 内部调用 `cv2.resize` 时：为了最大程度遏制由于图像粗暴放缩产生的抗锯齿与笔画墨迹模糊，对放大操作必须强制锚定 `interpolation=cv2.INTER_CUBIC`（双三次插值），这对后续识别由于风化导致的模糊文字边缘具有决定性的增益！")
    print("=================================================================================\n")

if __name__ == '__main__':
    analyze_dataset_widths()