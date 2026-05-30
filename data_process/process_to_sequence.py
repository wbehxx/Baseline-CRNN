import os
import cv2
import xml.etree.ElementTree as ET
import html
from tqdm import tqdm
from pathlib import Path

# ================= 配置路径 =================
# 根据你的软链接结构配置基础路径
DATA_DIR = Path("/home/mwl/disk_space/Baseline-CRNN/data")
RAW_IMG_DIR = DATA_DIR / "raw_deepjiandu" / "DeepJiandu" / "DeepJiandu"
RAW_XML_DIR = DATA_DIR / "raw_deepjiandu" / "DeepJiandu" / "DeepJiandu_labels"
OUT_DIR = DATA_DIR / "processed_sequence"

# 简牍通常较窄，设置同一行单字中心点 x 轴波动的最大容忍像素值
X_TOLERANCE = 80 

def parse_xml(xml_path):
    """解析单个 XML 文件，返回单字框的列表"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    boxes = []
    for obj in root.findall('object'):
        # 核心：解析 HTML 实体编码，如 &#26032; -> '新'
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
    """根据 x 坐标中心点将散乱的单字框聚类成竖行 (增强版)"""
    if not boxes:
        return []
    
    # 优化 1：动态计算当前简牍上所有单字的“平均宽度”
    # 用平均宽度的比例来作为容忍度，比硬编码 80 像素要健壮得多
    avg_width = sum(b['xmax'] - b['xmin'] for b in boxes) / len(boxes)
    # 设定容忍度为平均宽度的 0.6 倍 (如果两个字左右中心偏离超过 0.6 个字宽，就认为是另一列)
    dynamic_tolerance = avg_width * 0.6 
    
    # 先按 x 坐标排序
    boxes = sorted(boxes, key=lambda b: b['cx'])
    
    lines = []
    current_line = [boxes[0]]
    
    for box in boxes[1:]:
        # 优化 2：计算当前列的“动态中心线”，而不是只看上一个字
        # 这能有效防止简牍写歪导致的“坐标漂移”把隔壁列卷进来
        current_line_cx_mean = sum(b['cx'] for b in current_line) / len(current_line)
        
        if abs(box['cx'] - current_line_cx_mean) < dynamic_tolerance:
            current_line.append(box)
        else:
            lines.append(current_line)
            current_line = [box]
    lines.append(current_line)
    
    # 对聚类好的每一行，内部按照 y 坐标（从上到下）进行排序
    for line in lines:
        line.sort(key=lambda b: b['cy'])
        
    return lines

# def group_boxes_into_lines(boxes):
#     """根据 x 坐标中心点将散乱的单字框聚类成竖行"""
#     if not boxes:
#         return []
    
#     # 先按 x 坐标排序
#     boxes = sorted(boxes, key=lambda b: b['cx'])
    
#     lines = []
#     current_line = [boxes[0]]
    
#     for box in boxes[1:]:
#         # 如果当前字框的 x 中心点与上一行的 x 中心点在容忍误差内，认为是同一行
#         if abs(box['cx'] - current_line[-1]['cx']) < X_TOLERANCE:
#             current_line.append(box)
#         else:
#             lines.append(current_line)
#             current_line = [box]
#     lines.append(current_line)
    
#     # 对聚类好的每一行，内部按照 y 坐标（从上到下）进行排序
#     for line in lines:
#         line.sort(key=lambda b: b['cy'])
        
#     return lines

def process_dataset_split(split_name):
    """处理 train / val / test 某一划分的数据"""
    img_dir = RAW_IMG_DIR / split_name
    xml_dir = RAW_XML_DIR / split_name
    
    out_img_dir = OUT_DIR / split_name
    out_img_dir.mkdir(parents=True, exist_ok=True)
    
    # 存放生成的 sequence label，格式: 文件名.bmp\t汉字序列
    label_txt_path = OUT_DIR / f"{split_name}_labels.txt"
    
    xml_files = list(xml_dir.glob('*.xml'))
    char_vocab = set()
    
    with open(label_txt_path, 'w', encoding='utf-8') as f_label:
        for xml_file in tqdm(xml_files, desc=f"Processing {split_name}"):
            img_file = img_dir / f"{xml_file.stem}.bmp"
            
            if not img_file.exists():
                continue
            
            img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            
                
            boxes = parse_xml(xml_file)
            lines = group_boxes_into_lines(boxes)
            
            for line_idx, line in enumerate(lines):
                if not line: continue
                
                # 提取整行的序列文本并记录字典
                text_sequence = "".join([b['char'] for b in line])
                char_vocab.update(list(text_sequence))
                
                # 计算这行文字的全局外接矩形 (Bounding Box)
                l_xmin = max(0, min(b['xmin'] for b in line))
                l_xmax = min(img.shape[1], max(b['xmax'] for b in line))
                l_ymin = max(0, min(b['ymin'] for b in line))
                l_ymax = min(img.shape[0], max(b['ymax'] for b in line))
                
                # 裁剪竖条图像
                crop_img = img[l_ymin:l_ymax, l_xmin:l_xmax]
                if crop_img.size == 0: continue
                
                # 核心：逆时针旋转 90 度，竖排转横排
                rotated_img = cv2.rotate(crop_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                
                # 保存处理后的图片
                save_filename = f"{xml_file.stem}_line_{line_idx}.bmp"
                save_path = out_img_dir / save_filename
                cv2.imwrite(str(save_path), rotated_img)
                
                # 写入标签文件
                f_label.write(f"{save_filename}\t{text_sequence}\n")
                
    return char_vocab

if __name__ == '__main__':
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    all_vocab = set()
    for split in ['train', 'val', 'test']:
        print(f"--- Starting {split} split ---")
        vocab = process_dataset_split(split)
        all_vocab.update(vocab)
        
    # 生成全局字典文件 dict.txt (0留给CTC Blank)
    dict_path = OUT_DIR / "dict.txt"
    with open(dict_path, 'w', encoding='utf-8') as f:
        for idx, char in enumerate(sorted(all_vocab)):
            f.write(f"{char}\n")
            
    print(f"Processing complete! Total unique characters: {len(all_vocab)}")
    print(f"Vocabulary saved to {dict_path}")