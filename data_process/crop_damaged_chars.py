import os
import xml.etree.ElementTree as ET
import html
import cv2
from tqdm import tqdm

def extract_damaged_crops(images_base_dir, labels_base_dir, output_dir):
    # 创建输出文件夹
    os.makedirs(output_dir, exist_ok=True)
    print(f"========== 🕵️‍♂️ 开始提取残损占位符 '□' 的真实切片 ==========")
    print(f"💾 提取结果将保存在: {output_dir}\n")

    crop_count = 0

    # 遍历 train, val, test 三个子集
    splits = ['train', 'val', 'test']
    for split in splits:
        xml_split_dir = os.path.join(labels_base_dir, split)
        img_split_dir = os.path.join(images_base_dir, split)
        
        if not os.path.exists(xml_split_dir):
            continue
            
        xml_files = [f for f in os.listdir(xml_split_dir) if f.endswith('.xml')]
        print(f"⏳ 正在处理 {split} 集 (共 {len(xml_files)} 个 XML 文件)...")
        
        for xml_file in tqdm(xml_files, desc=split):
            xml_path = os.path.join(xml_split_dir, xml_file)
            
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()
                
                # 获取对应的图片文件名
                filename_node = root.find('filename')
                if filename_node is not None:
                    img_filename = filename_node.text
                else:
                    img_filename = xml_file.replace('.xml', '.bmp')
                    
                img_path = os.path.join(img_split_dir, img_filename)
                
                if not os.path.exists(img_path):
                    continue

                # 【核心优化】延迟读取：只有当这根简牍上有 □ 时，才去硬盘读取它，极大节省时间
                img = None 
                
                for obj in root.findall('object'):
                    name_node = obj.find('name')
                    if name_node is not None and name_node.text is not None:
                        # 解码 XML 实体，比如把 &#26032; 变成文字
                        char_name = html.unescape(name_node.text.strip())
                        
                        # 🎯 抓到残损占位符了！
                        if char_name == '□':
                            bndbox = obj.find('bndbox')
                            xmin = int(float(bndbox.find('xmin').text))
                            ymin = int(float(bndbox.find('ymin').text))
                            xmax = int(float(bndbox.find('xmax').text))
                            ymax = int(float(bndbox.find('ymax').text))
                            
                            # 如果还没读取图片，现在读取入内存
                            if img is None:
                                img = cv2.imread(img_path)
                                if img is None:
                                    break # 图片损坏或无法读取，跳过
                            
                            # OpenCV 裁剪图像 (注意 numpy 数组切片是 [y开始:y结束, x开始:x结束])
                            # 加入 max() 防止边界坐标越界报错
                            crop_img = img[max(0, ymin):ymax, max(0, xmin):xmax]
                            
                            # 如果切片有效且不为空
                            if crop_img.size > 0:
                                # 保存裁剪的图片
                                # 命名规则：原图名_坐标.jpg (例如: 17_37_41.jpg)，防止重名覆盖
                                save_name = f"{os.path.splitext(img_filename)[0]}_{xmin}_{ymin}.jpg"
                                save_path = os.path.join(output_dir, save_name)
                                cv2.imwrite(save_path, crop_img)
                                crop_count += 1
                                
            except Exception as e:
                print(f"❌ 解析或裁剪出错 {xml_path}: {e}")

    print(f"\n✅ 提取大功告成！")
    print(f"📸 累计抠出了 {crop_count} 张被标记为 '□' 的单字图片。")
    print(f"📁 请在 VS Code 左侧目录打开 {output_dir} 文件夹，亲自去当一次鉴定专家吧！")


if __name__ == '__main__':
    # ========================================================
    # ⚠️ 请核对以下路径是否与你服务器上的目录结构一致
    # 刚才你的 prompt 里图片目录写的是 DeepJianduDeepJiandu，请确认是否有拼写错误
    # ========================================================
    IMAGES_DIR = "/home/mwl/disk_space/Baseline-CRNN/data/raw_deepjiandu/DeepJiandu/DeepJiandu" 
    LABELS_DIR = "/home/mwl/disk_space/Baseline-CRNN/data/raw_deepjiandu/DeepJiandu/DeepJiandu_labels"
    
    # 专门存放抠出来的小图的新文件夹
    OUTPUT_DIR = "/home/mwl/disk_space/Baseline-CRNN/logs/check_damaged_crops"
    
    extract_damaged_crops(IMAGES_DIR, LABELS_DIR, OUTPUT_DIR)