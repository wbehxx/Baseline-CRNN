import os
import xml.etree.ElementTree as ET
import html

def check_raw_xml_dataset(xml_base_dir, dict_path):
    print(f"========== 🔍 开始进行原始 XML 数据体检 ==========")
    print(f"📂 正在扫描目录: {xml_base_dir}")

    all_chars_in_dataset = set()
    total_xml_files = 0
    total_boxes = 0

    # 1. 遍历 train, val, test 文件夹下的所有 xml 文件
    for root_dir, _, files in os.walk(xml_base_dir):
        for file in files:
            if file.endswith('.xml'):
                total_xml_files += 1
                xml_path = os.path.join(root_dir, file)
                
                try:
                    tree = ET.parse(xml_path)
                    root = tree.getroot()
                    
                    # 查找所有的 object 标签
                    for obj in root.findall('object'):
                        name_element = obj.find('name')
                        if name_element is not None and name_element.text is not None:
                            raw_name = name_element.text.strip()
                            
                            # 【核心处理】将 &#26032; 解码为真正的汉字
                            decoded_char = html.unescape(raw_name)
                            all_chars_in_dataset.add(decoded_char)
                            total_boxes += 1
                            
                except Exception as e:
                    print(f"❌ 解析 XML 失败 {xml_path}: {e}")

    print(f"\n📊 【第一部分：原始数据统计】")
    print(f"  - 共扫描 XML 文件: {total_xml_files} 个")
    print(f"  - 共提取字符边界框: {total_boxes} 个")
    print(f"  - 提取到的唯一字符种类: {len(all_chars_in_dataset)} 种")

    if '□' in all_chars_in_dataset:
        print("  - ℹ️ 发现残损占位符 '□'。")

    # 2. 检查 OOV (字典覆盖率)
    print(f"\n🧠 【第二部分：模型字典 OOV 检查】")
    if os.path.exists(dict_path):
        with open(dict_path, 'r', encoding='utf-8') as f:
            # 读取字典并去重
            vocab = set([line.strip() for line in f.readlines() if line.strip()])
        
        print(f"  - 成功加载字典: {dict_path} (共 {len(vocab)} 个字)")
        
        # 找出 XML 中存在，但字典里没有的字
        oov_chars = all_chars_in_dataset - vocab
        
        if oov_chars:
            print(f"  - 🚨 危险警告：发现 {len(oov_chars)} 个 OOV 字符！")
            print(f"  - 你的字典缺少以下字符（它们出现在了 XML 标注中）：")
            # 格式化打印 OOV 字符，方便查看
            oov_list = list(oov_chars)
            for i in range(0, len(oov_list), 20):
                print(f"    {' '.join(oov_list[i:i+20])}")
        else:
            print(f"  - ✅ OOV 检查完美通过！原始 XML 里的每一个字都在你的字典中。")
    else:
        print(f"  - ❌ 找不到字典文件: {dict_path}")

if __name__ == '__main__':
    # ========================================================
    # ⚠️ 请修改为你服务器上的真实绝对路径
    # ========================================================
    # 指向包含 train, val, test 文件夹的那个总目录
    XML_LABELS_DIR = "/home/mwl/disk_space/Baseline-CRNN/data/raw_deepjiandu/DeepJiandu/DeepJiandu_labels" 
    # 指向你的字典文件
    DICT_FILE = "/home/mwl/disk_space/Baseline-CRNN/data/processed_sequence/dict.txt" 
    
    check_raw_xml_dataset(XML_LABELS_DIR, DICT_FILE)