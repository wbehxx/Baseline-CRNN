import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR, ReduceLROnPlateau
from torchaudio.models.decoder import ctc_decoder
# 导入我们之前写好的模块
from dataset import JianduDataset, jiandu_collate_fn , LengthGroupedBatchSampler
from model_crnn import CRNN
from utils import get_logger, CTCLabelConverter, calculate_cer
from model_v4 import build_ppocrv4_jiandu
from itertools import count  # 新增这一行

def train():
    # 1. 读取配置文件
    with open('/home/mwl/disk_space/Baseline-CRNN/configs/config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    # 固定随机种子
    torch.manual_seed(config['project']['seed'])
    
    # 初始化日志记录器
    os.makedirs(config['log']['logs_dir'], exist_ok=True)
    logger = get_logger(os.path.join(config['log']['logs_dir'], 'train_ppocrv4.log'))
    logger.info(f"🚀 开始训练: {config['project']['name']}")

    # 2. 设置设备 (优先使用 GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"💻 使用设备: {device}")


    # # ================= 【修改后】3. 实例化 Dataset 和 DataLoader =================
    logger.info("📦 正在加载数据集...")
    
    # 实例化 Dataset（完全保持你原有的精准传参，不需要动）
    train_dataset = JianduDataset(
        labels_txt_path=config['data']['train_list'],
        img_dir=config['data']['train_dir'],
        dict_path=config['data']['dict_path'],
        img_height=config['model']['img_height'],
        max_width=config['model']['max_width'],
        is_train=True  # 激活数据增强
    )
    
    val_dataset = JianduDataset(
        labels_txt_path=config['data']['val_list'],
        img_dir=config['data']['val_dir'],
        dict_path=config['data']['dict_path'],
        img_height=config['model']['img_height'],
        max_width=config['model']['max_width'],
        is_train=False # 保持验证集纯净
    )

    # 实例化分桶采样器（使用你 config 里的专属 Key）
    train_batch_sampler = LengthGroupedBatchSampler(
        dataset=train_dataset,
        batch_size=config['train']['batch_size'],
        shuffle=True  # 训练集打乱桶的顺序
    )
    
    val_batch_sampler = LengthGroupedBatchSampler(
        dataset=val_dataset,
        batch_size=config['train']['batch_size'],
        shuffle=False # 验证集绝对不打乱
    )

    # 实例化 DataLoader
    # 【⚠️ 核心注意】：一旦指定了 batch_sampler，DataLoader 内部原有的 batch_size 和 shuffle 参数必须删掉！
    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_sampler=train_batch_sampler, # 👈 接入训练分桶器
        num_workers=config['train']['num_workers'], 
        collate_fn=jiandu_collate_fn, 
        pin_memory=True
    )
    
    val_loader = DataLoader(
        dataset=val_dataset, 
        batch_sampler=val_batch_sampler,   # 👈 接入验证分桶器
        num_workers=config['train']['num_workers'], 
        collate_fn=jiandu_collate_fn, 
        pin_memory=True
    )
    # ==============================================================================
    # # 3. 实例化 Dataset 和 DataLoader
    # logger.info("📦 正在加载数据集...")
    # train_dataset = JianduDataset(
    #     labels_txt_path=config['data']['train_list'],
    #     img_dir=config['data']['train_dir'],
    #     dict_path=config['data']['dict_path'],
    #     img_height=config['model']['img_height'],
    #     max_width=config['model']['max_width'],
    #     is_train=True  # <--- 【关键修改】激活数据增强！
    # )
    
    # val_dataset = JianduDataset(
    #     labels_txt_path=config['data']['val_list'],
    #     img_dir=config['data']['val_dir'],
    #     dict_path=config['data']['dict_path'],
    #     img_height=config['model']['img_height'],
    #     max_width=config['model']['max_width'],
    #     is_train=False # <--- 保持验证集纯净
    # )

    # train_loader = DataLoader(
    #     train_dataset, batch_size=config['train']['batch_size'], 
    #     shuffle=True, num_workers=config['train']['num_workers'], 
    #     collate_fn=jiandu_collate_fn, pin_memory=True
    # )
    
    # val_loader = DataLoader(
    #     val_dataset, batch_size=config['train']['batch_size'], 
    #     shuffle=False, num_workers=config['train']['num_workers'], 
    #     collate_fn=jiandu_collate_fn, pin_memory=True
    # )

    # 初始化标签转换器 (翻译官)
    converter = CTCLabelConverter(config['data']['dict_path'])
    # num_classes = len(converter.char2id) + 1  # 字典字数 + 1个Blank标签

    # # 👇 ================= 【新增】初始化 Beam Search 解码器 ================= 👇
    # # 1. 构建符合 torchaudio 格式的字符列表 
    # # 假设 0 是 Blank，对应的字符我们设为 '-'
    # tokens = ['-'] 
    # for i in range(1, num_classes):
    #     tokens.append(converter.id2char[i])

    # # 2. 实例化解码器 (只需实例化一次)
    # # beam_size 可以根据你的速度容忍度在 5~20 之间调整，10 是一个很好的平衡点
    # beam_search_decoder = ctc_decoder(
    #     lexicon=None,       
    #     tokens=tokens,
    #     blank_token='-',
    #     sil_token='-',      # 👈 新增这一行：覆盖官方默认的 '|'，告诉它静音符也是 '-'
    #     beam_size=5        
    # )
    # # 👆 ====================================================================== 👆

    # 4. 实例化模型、损失函数和优化器
    # model = build_ppocrv4_jiandu(num_classes=train_dataset.num_classes).to(device)
    
    model = CRNN(
        img_height=config['model']['img_height'],
        nc=3,
        num_classes=train_dataset.num_classes,
        nh=config['model']['nh']
    ).to(device)

    # zero_infinity=True 是防止早期预测极端导致 Loss 变为 NaN 的保命设置
    criterion = nn.CTCLoss(blank=0, zero_infinity=True).to(device)
    
    # ---------------- 替换原来的 optimizer = optim.Adam(...) ----------------
    # 区分预训练参数和随机初始化的头部参数
    # head_params = []
    # backbone_params = []
    # for name, param in model.named_parameters():
    #     if 'head' in name: 
    #         head_params.append(param)
    #     else:
    #         backbone_params.append(param)
    # 使用 AdamW 替代 Adam，完美解耦权重衰减
    # optimizer = optim.AdamW([
    #     {'params': backbone_params, 'lr': config['train']['learning_rate'] * 0.01},
    #     {'params': head_params, 'lr': config['train']['learning_rate']}
    # ], weight_decay=5e-5)  # 衰减从 1e-4 降为 1e-5

    # 🛠️ 一键替换为优雅的全局 AdamW 优化器
    optimizer = optim.AdamW(model.parameters(), lr=config['train']['learning_rate'], weight_decay=1e-4)

    # 记录初始学习率以便 Warmup 使用
    initial_lrs = [group['lr'] for group in optimizer.param_groups]

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-7)
    # factor=0.5 表示每次衰减一半
    # patience=5 表示如果验证集 CER 连续 5 轮没下降，就触发衰减
    # min_lr=1e-7 是兜底的最低学习率
    # ------------------------------------------------------------------------
    
    # optimizer = optim.Adam(model.parameters(), lr=config['train']['learning_rate'])
    
    # 差异化学习率 (Differential Learning Rate)
    # 躯干用 1/10 的学习率保护记忆，新长出来的头用正常学习率学习新字典
    # optimizer = optim.Adam([
    #     {'params': backbone_params, 'lr': config['train']['learning_rate'] * 0.01},
    #     {'params': head_params, 'lr': config['train']['learning_rate']}
    # ], weight_decay=1e-4)  # 👈 新增这一句：L2 正则化，防止过拟合

    # 5. 【核心】早停机制 (Early Stopping) 设置
    best_cer = float('inf')   # 记录历史最低字符错误率，初始为无穷大
    patience = 30             # 容忍度：如果连续 10 个 Epoch 错误率没下降，就早停
    patience_counter = 0      # 计数器
    os.makedirs(config['log']['weights_dir'], exist_ok=True)

    # ================= 开始训练循环 ===============================================
    # epochs = config['train']['epochs']
    # for epoch in range(1, epochs + 1):

    #梯度累积
    # accumulation_steps = 4  # 既然真实 batch_size 是 8，累积 8 次等效于 64
    # ---------------- 修改后 ----------------
    epochs = "∞"  # 仅用于进度条显示，看起来更酷
    for epoch in count(1):  # 从 1 开始无限循环，直到被早停打破
        # ================= 在每轮 Epoch 开始前的部分 =================
        # 手动 Warmup 逻辑（前 5 轮）
        if epoch <= 5:
            for group, init_lr in zip(optimizer.param_groups, initial_lrs):
                # 从 1% 线性增长到 100%
                group['lr'] = init_lr * (0.01 + 0.99 * (epoch / 5))
        
        epoch_anomaly_count = 0 # 👈 新增：重置本轮异常计数器
        model.train()
        train_loss = 0.0
        
        # 包装 tqdm 进度条
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]")
        for images, targets, target_lengths in pbar:
        # for batch_idx, (images, targets, target_lengths) in enumerate(pbar):
            images = images.to(device)
            targets = targets.to(device)
            
            # 清空梯度
            optimizer.zero_grad()
            # 前向传播
            # preds = model(images)['CTC']
            preds = model(images) # 形状: [序列长度W, batch_size, num_classes]

            # # 👇 【真正救命的一行代码】将裸 Logits 转换为对数概率！
            # preds = F.log_softmax(preds, dim=2)
            # # 👇 【关键修复】一键转置，互换第 0 维(Batch)和第 1 维(Time)
            # preds = preds.permute(1, 0, 2)
            # # 现在 preds 的形状变成了 CTCLoss 喜欢的 [Time, Batch, Num_Classes]

            batch_size = images.size(0)
            
            # 获取模型输出的序列长度 (W)，填入一个张量中，并送到显卡
            input_lengths = torch.IntTensor([preds.size(0)] * batch_size).to(device)
            target_lengths = target_lengths.to(device)
            
            # # 计算 CTC Loss
            # loss = criterion(preds, targets, input_lengths, target_lengths)

            # 👇 ================= 【非干涉型：法医级静默探针】 ================= 👇
            # 计算 CTC Loss 前，先做一次安全计算
            loss = criterion(preds, targets, input_lengths, target_lengths)
            
            # # 👇 【关键】Loss 缩放，保证累积后的梯度大小与 batch_size=64 相同
            # loss = loss / accumulation_steps

            # # 如果捕捉到 NaN 或 Inf
            # if torch.isinf(loss) or torch.isnan(loss):
            #     logger.warning("\n" + "🔥"*25)
            #     logger.warning(f"🚨 警告：检测到异常 Loss ({loss.item()})！已启动静默排雷协议。")
                
            #     # 嫌疑 1：CTC 铁律检查 (时间步够不够？)
            #     if (input_lengths < target_lengths).any():
            #         logger.warning("❌ 死因确诊 [1/3]：存在时间步 < 标签字数的样本！")
            #         for i in range(batch_size):
            #             if input_lengths[i] < target_lengths[i]:
            #                 logger.warning(f"    -> 致命样本 {i}：模型格子数={input_lengths[i].item()}，真实字数={target_lengths[i].item()}")
            #                 logger.warning(f"    -> 诊断：你的 max_width=800 在当前网络下只能提供 {input_lengths[i].item()} 个格子，但这张简牍太长了。")
            #     # 嫌疑 2：模型输出中毒 (梯度爆炸)
            #     if torch.isnan(preds).any() or torch.isinf(preds).any():
            #         logger.warning("❌ 死因确诊 [2/3]：模型输出张量 (preds) 已被 NaN 污染！")
            #         logger.warning("    -> 诊断：学习率可能过大，导致前几个 Batch 梯度爆炸，破坏了权重。请调小主干学习率至 1e-5。")
            #     # 嫌疑 3：标签越界
            #     if (targets >= num_classes).any() or (targets < 0).any():
            #         logger.warning(f"❌ 死因确诊 [3/3]：标签 target 里出现了越界 ID！最大合法ID应为 {num_classes-1}。")
            #         logger.warning(f"    -> 越界值: {targets[(targets >= num_classes) | (targets < 0)].tolist()}")  
            #     logger.warning("🛡️ 护盾激活：已强行跳过此异常 Batch，阻止脏梯度污染模型！训练继续...")
            #     logger.warning("🔥"*25 + "\n")
                
            #     # 核心：清空刚刚试图产生的脏梯度，跳过本次参数更新，继续下一个 Batch
            #     optimizer.zero_grad()
            #     continue
            # # 👆 ============================================================== 👆
            
            # 👇 ================= 【限流精准探针 2.0】 ================= 👇
            if torch.isinf(loss) or torch.isnan(loss):
                epoch_anomaly_count += 1
                
                # 只有前 3 次异常会打印详细分析，防止刷屏
                if epoch_anomaly_count <= 3:
                    # 区分是 inf 还是 nan
                    loss_type = "INF (数学下溢/极度自信误判)" if torch.isinf(loss) else "NAN (梯度污染/权重损坏)"
                    logger.warning(f"🚨 异常拦截 [{epoch_anomaly_count}/3] | 类型: {loss_type}")
                    
                    if torch.isinf(loss):
                        logger.warning("    -> 诊断: 模型对正确路径预测概率趋近于0。通常因为初期权重未对齐，或遇到了极度离谱的样本。")
                    elif torch.isnan(loss):
                        logger.warning("    -> 诊断: 模型的输入 preds 中存在 NaN！这说明之前的更新可能已产生脏梯度，请检查学习率是否过大。")
                        
                elif epoch_anomaly_count == 4:
                    logger.warning("🔇 护盾提示：本轮异常次数已达限额，后续将开启【静默拦截模式】不再打印...")
                
                # 核心救命操作：清空当前可能产生的脏梯度，跳过本次更新
                optimizer.zero_grad()
                continue
            # 👆 ======================================================== 👆

            # 反向传播与优化
            loss.backward()

            # # 👇 【关键】梯度累积逻辑：每攒够 accumulation_steps 个 batch，才更新一次
            # if (batch_idx + 1) % accumulation_steps == 0:
            #     # 1. 梯度裁剪 (防止累积后的梯度爆炸)
            #     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            #     # 2. 真正更新模型权重
            #     optimizer.step()
            #     # 3. 清空梯度池，迎接下一轮累积
            #     optimizer.zero_grad()
            # # 恢复 Loss 用于显示 (乘回去)
            # train_loss += (loss.item() * accumulation_steps)
            # pbar.set_postfix({'loss': f"{(loss.item() * accumulation_steps):.4f}"})

            # 👇 【新增】梯度裁剪（防爆锁）：把过激的梯度强行压制在 5.0 以内
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        avg_train_loss = train_loss / len(train_loader)
        logger.info(f"Epoch {epoch} | 训练 Loss: {avg_train_loss:.4f}")
        # 👇 汇报本轮到底拦截了多少次
        if epoch_anomaly_count > 0:
            logger.warning(f"🛡️ 战况总结：本轮共成功拦截了 {epoch_anomaly_count} 个异常 Batch。")
        
        # ================= 开始验证循环 =======================================
        if epoch % config['log']['val_interval'] == 0:
            model.eval()
            val_loss = 0.0
            all_preds_str = []
            all_targets_str = []
            
            with torch.no_grad():
                for images, targets, target_lengths in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Val]"):
                    images = images.to(device)
                    targets = targets.to(device)
                    
                    # preds_dict = model(images)
                    # preds = preds_dict['CTC'] # 明确取出 CTC 的输出张量
                    
                    # # 👇 【验证集也要加上这一行】
                    # preds = F.log_softmax(preds, dim=2)
                    # # 👇 【关键修复】验证集同样需要转置
                    # preds = preds.permute(1, 0, 2)
                    
                    # batch_size = images.size(0)
                    preds = model(images)
                    batch_size = images.size(0)
                    input_lengths = torch.IntTensor([preds.size(0)] * batch_size).to(device)
                    target_lengths = target_lengths.to(device)
                    
                    # 验证集的 Loss
                    loss = criterion(preds, targets, input_lengths, target_lengths)
                    val_loss += loss.item()
                    
                    # ----------------- 解码与评估 -----------------
                    # 1. 提取预测概率最大值的索引
                    _, preds_index = preds.max(2) # [W, batch_size]
                    # 转置为 [batch_size, W] 以便转换器解码
                    preds_index = preds_index.transpose(1, 0).contiguous() 
                    
                    # 2. 将预测 ID 解码为汉字字符串
                    preds_str = converter.decode(preds_index.cpu(), input_lengths)
                    all_preds_str.extend(preds_str)
                    
                    # 3. 将真实 ID 还原为汉字字符串 (用于计算 CER)
                    start = 0
                    for length in target_lengths:
                        t = targets[start:start+length]
                        target_str = ''.join([converter.id2char[int(x)] for x in t])
                        all_targets_str.append(target_str)
                        start += length
                    
                    # # ----------------- 解码与评估 (束搜索 Beam Search) -----------------
                    # # 1. 转换维度并推入 CPU
                    # # torchaudio 接收的格式是 [Batch, Time, Num_Classes]，且必须在 CPU 上运行
                    # preds_for_beam = preds.transpose(0, 1).cpu() 
                    # lengths_for_beam = input_lengths.cpu()

                    # # 2. 执行 Beam Search 解码
                    # # 输入的 preds 已经是经过 log_softmax 处理的
                    # beam_results = beam_search_decoder(preds_for_beam, lengths_for_beam)

                    # # 3. 提取解码出的最优汉字字符串
                    # for result in beam_results:
                    #     # result[0] 包含了该样本得分最高的那条路径
                    #     best_tokens = result[0].tokens
                    #     # 遍历 token ID 并映射回汉字，跳过 blank
                    #     pred_text = ''.join([tokens[tid] for tid in best_tokens])
                    #     all_preds_str.append(pred_text)

                    # # 4. 将真实 ID 还原为汉字字符串 (用于计算 CER，这部分保持你原来的逻辑)
                    # start = 0
                    # for length in target_lengths:
                    #     t = targets[start:start+length]
                    #     target_str = ''.join([converter.id2char[int(x)] for x in t])
                    #     all_targets_str.append(target_str)
                    #     start += length
                    # # ------------------------------------------------------------------
                        
            avg_val_loss = val_loss / len(val_loader)
            # 计算当前 Epoch 的平均字符错误率
            current_cer = calculate_cer(all_preds_str, all_targets_str)
            
            logger.info(f"Epoch {epoch} | 验证 Loss: {avg_val_loss:.4f} | CER: {current_cer:.4f}")
            # 打印几个样本对照，直观感受模型学得怎么样
            # # 🛠️ 修复：动态寻找整个验证集中真实字数最多的那个样本的索引
            print_idx = max(range(len(all_targets_str)), key=lambda i: len(all_targets_str[i]))-50
            logger.info(f"👀抽查样本|预测：{all_preds_str[print_idx]}")
            logger.info(f"👀抽查样本|真实：{all_targets_str[print_idx]}")

            # logger.info(f"👀 抽查样本 -> 预测: {all_preds_str[0]} | 真实: {all_targets_str[0]}")
            # logger.info(f"抽查样本|预测：{all_preds_str[0]}")
            # logger.info(f"抽查样本|真实：{all_targets_str[0]}")

            # ================= 在每轮 Epoch 结束时的部分 =================
            # 从第 6 轮开始，把决定权交给自适应衰减器
            if epoch > 5:
                # 注意：ReduceLROnPlateau 的 step 必须传入你当前监控的指标！
                scheduler.step(current_cer)
                
            # 打印当前学习率, 打印头部学习率，因为它是主角
            # current_lr = optimizer.param_groups[1]['lr'] 
            # logger.info(f"📉 当前 Head 学习率: {current_lr:.6f}")
            current_lr = optimizer.param_groups[0]['lr'] 
            logger.info(f"📉 当前学习率: {current_lr:.6f}")

            # ================= 早停与模型保存逻辑 =================
            if current_cer < best_cer:
                best_cer = current_cer
                patience_counter = 0 # 破纪录了，计数器清零
                
                # 保存最佳模型
                best_model_path = os.path.join(config['log']['weights_dir'], 'best_model.pth')
                torch.save(model.state_dict(), best_model_path)
                logger.info(f"🌟 发现更优模型！CER 降至 {best_cer:.4f}，已保存至 {best_model_path}")
            else:
                patience_counter += 1
                logger.info(f"⚠️ CER 未下降，早停计数器: {patience_counter}/{patience}")
                if patience_counter >= patience:
                    logger.info(f"🛑 连续 {patience} 个 Epoch 指标未提升，触发早停机制，训练结束！")
                    break # 跳出训练大循环

    logger.info(f"🎉 训练任务全部完成！历史最佳验证集 CER 为: {best_cer:.4f}")

if __name__ == '__main__':
    train()