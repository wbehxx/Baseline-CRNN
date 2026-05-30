config/
    config.yaml #模型配置文件
data/
    processed_sequence   #预处理后的数据集，自适应字体宽度，大部分能正确切割每一列
    processed_sequence   #预处理后的数据集，第一次固定阈值的，切割效果不好
    raw_deepjiandu   #原来的Deepjiandu数据集
data_process/
    analyze_widths.py   #统计数据集图片统一高度缩放后宽度分布
    check_keys.py   #检查预训练模型的名字
    check_oov.py   #检测是否有测试集外文字
    crop_damaged_chars.py   #把所有标签是残损占位符的字切出来
    find_blueprint.py   #查找paddle仓库里的yaml文件
    process_to_sequence.py   #预处理：把竖着的图片转成横的，然后切成单行的
    test_transplant.py   #测试脚本：验证能否成功把预训练权重完美接入 2243 分类（2242个字 + 1个空白符）任务
logs/
    check_damaged_crops   #所有标签为残损占位符的字的单字切割图
    vis_result   #无数据增强、无预训练权重的模型在测试集上跑完的可视化图
    eval_crnn.log   #测试集上的推理日志
    train_crnn.log   #使用原始crnn模型的训练日志
    train_ppocrv4.log   #使用PPOCR-v4模型的训练日志
weights
    best_model_v1.pth   #无数据增强、无预训练权重的模型
    best_model_v2.pth   #加入数据增强、但无预训练权重的模型

dataset.py   
evaluate.py   #在测试集上推理
model_crnn.py
model_v4.py     #PPOCR-v4模型
README.md
requirments.txt
train.py
utils.py
