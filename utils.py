import os
import logging
import torch
import editdistance

def get_logger(log_file_path):
    """
    配置日志记录器
    同时将训练过程的输出打印到终端（Console）和保存到文件（.log）中
    """
    # 确保日志文件夹存在
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    
    logger = logging.getLogger('CRNN_Baseline')
    logger.setLevel(logging.INFO)
    
    # 防止重复添加 handler 导致日志重复打印
    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        
        # 1. 终端输出
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        
        # 2. 文件输出
        file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger

class CTCLabelConverter(object):
    """
    CTC 标签转换器 (翻译官)
    负责将模型的预测 ID 序列解码为人类可读的汉字字符串
    """
    def __init__(self, dict_path):
        # 0 被保留为 CTC Blank 占位符
        self.char2id = {}
        self.id2char = {}
        
        with open(dict_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                char = line.strip()
                self.char2id[char] = idx + 1
                self.id2char[idx + 1] = char
                
    def decode(self, text_index, length):
        """
        核心逻辑：CTC 贪心解码 (Greedy Decoding)
        text_index: 形状为 [batch_size, sequence_length] 的预测 ID 矩阵
        length: 实际预测的序列长度列表
        """
        texts = []
        for i in range(text_index.size(0)):
            t = text_index[i][:length[i]]
            
            char_list = []
            for idx in range(length[i]):
                # CTC 解码规则 1：忽略连续重复的字符
                # (例如预测出 [1, 1, 0, 2, 2, 2] -> 真实意图是 [1, 0, 2])
                if int(t[idx]) != 0 and (not (idx > 0 and t[idx - 1] == t[idx])):
                    # CTC 解码规则 2：忽略 Blank 占位符 (ID 为 0)
                    char_list.append(self.id2char[int(t[idx])])
                    
            texts.append(''.join(char_list))
        return texts

def calculate_cer(predicts, targets):
    """
    计算字符错误率 (Character Error Rate, CER)
    公式：CER = (替换字符数 + 插入字符数 + 删除字符数) / 真实字符串总长度
    """
    if len(predicts) != len(targets):
        raise ValueError("预测列表和真实标签列表的长度不一致！")
        
    total_distance = 0
    total_length = 0
    
    for pred, target in zip(predicts, targets):
        # 使用 Levenshtein 编辑距离算法
        distance = editdistance.eval(pred, target)
        total_distance += distance
        total_length += len(target)
        
    # 防止除以 0 的情况（虽然在这个数据集中不太可能发生）
    if total_length == 0:
        return 0.0
        
    cer = total_distance / total_length
    return cer