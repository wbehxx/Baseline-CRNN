import os
import yaml
import glob
import pprint

# 搜索仓库里的所有配置文件
search_pattern = '/home/mwl/disk_space/PaddleOCR2Pytorch_Core/configs/**/*.yml'
files = glob.glob(search_pattern, recursive=True)

# 寻找带有 v4 和 rec 的核心文件
target_files = [f for f in files if 'v4' in f.lower() and 'rec' in f.lower()]

if not target_files:
    print("❌ 没找到 v4 的配置文件，请检查路径。")
else:
    for f in target_files:
        print(f"\n📄 破解图纸: {f}")
        with open(f, 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)
            if 'Architecture' in data:
                print("================ 核心 Backbone 架构 ==================")
                pprint.pprint(data['Architecture'])
                print("======================================================")