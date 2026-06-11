import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import html
from tqdm import tqdm
from pathlib import Path

# ================= 严格对齐你服务器的原始路径 =================
DATA_DIR = Path("/home/mwl/disk_space/Baseline-CRNN/data")
RAW_IMG_DIR = DATA_DIR / "raw_deepjiandu" / "DeepJiandu" / "DeepJiandu"
RAW_XML_DIR = DATA_DIR / "raw_deepjiandu" / "DeepJiandu" / "DeepJiandu_labels"

# 🚀 全新指定的目标存放地：正体横排数据集
OUT_DIR = DATA_DIR / "processed_horizontal_sequence"

TARGET_HEIGHT = 48    # 官方指定黄金高度

def parse_xml(xml_path):
    """解析单个 XML 文件，返回单字框的列表"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    boxes = []
    for obj in root.findall('object'):
        # 解析 HTML 实体编码，如 &#19977; -> '三'
        char_name = html.unescape(obj.find('name').text.strip())
        
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        
        cx = (xmin + xmax) // 2
        cy = (ymin + ymax) // 2
        
        boxes.append({
            'char': char_name,
            'xmin': xmin, 'ymin': ymin, 
            'xmax': xmax, 'ymax': ymax,
            'cx': cx, 'cy': cy
        })
    return boxes

def group_boxes_into_lines(boxes):
    """根据 x 坐标中心点将散乱的单字框聚类成竖行 (保留你的增强版算法)"""
    if not boxes:
        return []
    
    avg_width = sum(b['xmax'] - b['xmin'] for b in boxes) / len(boxes)
    dynamic_tolerance = avg_width * 0.6 
    
    boxes = sorted(boxes, key=lambda b: b['cx'])
    
    lines = []
    current_line = [boxes[0]]
    
    for box in boxes[1:]:
        current_line_cx_mean = sum(b['cx'] for b in current_line) / len(current_line)
        
        if abs(box['cx'] - current_line_cx_mean) < dynamic_tolerance:
            current_line.append(box)
        else:
            lines.append(current_line)
            current_line = [box]
    lines.append(current_line)
    
    # 核心：列内字框严格按照 ymin (从上到下) 排序，拼接后自然变成从左到右
    for line in lines:
        line.sort(key=lambda b: b['cy'])
        
    return lines

def process_dataset_split(split_name):
    """处理 train / val / test 某一划分的数据并执行横排拼接"""
    img_dir = RAW_IMG_DIR / split_name
    xml_dir = RAW_XML_DIR / split_name
    
    out_img_dir = OUT_DIR / split_name
    out_img_dir.mkdir(parents=True, exist_ok=True)
    
    label_txt_path = OUT_DIR / f"{split_name}_labels.txt"
    xml_files = list(xml_dir.glob('*.xml'))
    char_vocab = set()
    
    with open(label_txt_path, 'w', encoding='utf-8') as f_label:
        for xml_file in tqdm(xml_files, desc=f"Processing {split_name}"):
            img_file = img_dir / f"{xml_file.stem}.bmp"
            
            if not img_file.exists():
                continue
            
            # 读取原始灰度图像
            img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
                
            boxes = parse_xml(xml_file)
            lines = group_boxes_into_lines(boxes)
            
            for line_idx, line in enumerate(lines):
                if not line: continue
                
                text_sequence = "".join([b['char'] for b in line])
                char_vocab.update(list(text_sequence))
                
                # 👇 =================【核心重构：正体单字拼接逻辑】================= 👇
                cropped_chars = []
                for b in line:
                    # 坐标安全裁剪
                    ymin = max(0, b['ymin'])
                    ymax = min(img.shape[0], b['ymax'])
                    xmin = max(0, b['xmin'])
                    xmax = min(img.shape[1], b['xmax'])
                    
                    char_crop = img[ymin:ymax, xmin:xmax]
                    if char_crop.size == 0: 
                        continue
                    
                    # 每一个直立的汉字等比缩放到高度为 48
                    h, w = char_crop.shape[:2]
                    resized_w = int(w * (TARGET_HEIGHT / float(h)))
                    if resized_w <= 0: 
                        resized_w = 1
                        
                    char_resized = cv2.resize(char_crop, (resized_w, TARGET_HEIGHT))
                    cropped_chars.append(char_resized)
                
                if not cropped_chars: 
                    continue
                
                # 将直立的汉字从左到右水平拼接 (Horizontal Stack)
                stitched_line_img = np.hstack(cropped_chars)
                # 👆 ================================================================== 👆
                
                # 保存为标准横排图片
                save_filename = f"{xml_file.stem}_line_{line_idx}.bmp"
                save_path = out_img_dir / save_filename
                cv2.imwrite(str(save_path), stitched_line_img)
                
                # 写入带有制表符 \t 的标准标签文件
                f_label.write(f"{save_filename}\t{text_sequence}\n")
                
    return char_vocab

if __name__ == '__main__':
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    all_vocab = set()
    for split in ['train', 'val', 'test']:
        print(f"\n--- Starting {split} split ---")
        vocab = process_dataset_split(split)
        all_vocab.update(vocab)
        
    # 生成全新的、纯净的全局字典文件 dict.txt
    dict_path = OUT_DIR / "dict.txt"
    with open(dict_path, 'w', encoding='utf-8') as f:
        for char in sorted(all_vocab):
            f.write(f"{char}\n")
            
    print(f"\n🎉 数据集完美重构完毕!")
    print(f"独立汉字总数: {len(all_vocab)}")
    print(f"新数据与字典已全部保存在: {OUT_DIR}")