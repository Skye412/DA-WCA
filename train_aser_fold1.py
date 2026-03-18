"""
=============================================================================
Experiment: ASER-lite (Ambiguity-Suppressed Edge Refinement) Phase 1
Objective: 验证在 100% Patch 采样基线下，加入局部边缘细化与歧义抑制模块，
           能否突破当前 Fold 1 的最高 F1 (0.6491)，解决断连与局部背景误检。
Strategy: 
  - Backbone: WPFormer (PVTv2)
  - Sampling: 100% Patch (ESDI_dataloader.py with p=1.0)
  - Loss: Seg Loss (Aux 0.5 + Refined 1.0) + Edge Loss (0.1) + Ambiguity Loss (0.1)
  - Data: Fold 1 Only
=============================================================================
"""

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from PIL import Image
import cv2
import os
import logging
from tqdm import tqdm

from model.WPFormer import WPFormer
from ESDI_dataloader import get_loader

# ================= 1. 配置区 =================
FOLD = 1
# 使用绝对路径存放在你要求的 save 目录下
BASE_DIR = "/home/skye/data/Skye/DA-WCA/save/stage3_patch_aser_fold1"
S2DS_DIR = "/home/skye/data/Skye/databases/s2ds5"
PRETRAINED_WEIGHTS = "/home/skye/data/Skye/DA-WCA/save/stage2_512/checkpoints/WPFormer_synth_512_best.pth"

# 自动创建各种输出目录
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")
LOG_DIR = os.path.join(BASE_DIR, "logs")
PRED_DIR = os.path.join(BASE_DIR, "predictions")
COMP_DIR = os.path.join(BASE_DIR, "composites_3in1")

for d in [CKPT_DIR, LOG_DIR, PRED_DIR, COMP_DIR]:
    os.makedirs(d, exist_ok=True)

# ================= 2. 日志系统 =================
def get_logger(name, log_file):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if logger.handlers:
        logger.handlers.clear()
    
    fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(message)s'))
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

train_logger = get_logger("train", os.path.join(LOG_DIR, "train.log"))
eval_logger = get_logger("eval", os.path.join(LOG_DIR, "eval.log"))

# ================= 3. 核心机制 =================
def calculate_metrics(pred, gt_img):
    # 只计算非 ignore (0 和 255) 区域
    valid_mask = (gt_img == 0) | (gt_img == 255)
    pred_valid = pred[valid_mask]
    gt_valid = gt_img[valid_mask]
    if len(pred_valid) == 0: return 0, 0, 0, 0
    
    gt_bin = (gt_valid == 255).astype(np.float32)
    pred_bin = (pred_valid >= 128).astype(np.float32)
    
    tp = (pred_bin * gt_bin).sum()
    fp = (pred_bin * (1 - gt_bin)).sum()
    fn = ((1 - pred_bin) * gt_bin).sum()
    
    iou = tp / (tp + fp + fn + 1e-8)
    f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    
    return iou, f1, precision, recall

def total_loss(pred, mask):
    pred = torch.sigmoid(pred)
    valid_mask = (mask != 255).float()
    clean_mask = torch.where(mask == 255, torch.zeros_like(mask), mask)
    bce = (F.binary_cross_entropy(pred, clean_mask, reduction='none') * valid_mask).sum() / (valid_mask.sum() + 1e-8)
    inter = (pred * clean_mask * valid_mask).sum(dim=(2, 3))
    union = ((pred + clean_mask) * valid_mask).sum(dim=(2, 3))
    iou = 1 - inter / (union - inter + 1e-8)
    return iou.mean() + bce

def sliding_window_inference(model, image_tensor, window_size=512, stride=256):
    b, c, h, w = image_tensor.size()
    full_mask = torch.zeros((b, 1, h, w), device=image_tensor.device)
    count_map = torch.zeros((b, 1, h, w), device=image_tensor.device)

    pad_h = max(0, window_size - h)
    pad_w = max(0, window_size - w)
    if pad_h > 0 or pad_w > 0:
        image_tensor = F.pad(image_tensor, (0, pad_w, 0, pad_h), mode='reflect')
        _, _, h_pad, w_pad = image_tensor.size()
    else:
        h_pad, w_pad = h, w

    y_steps = list(range(0, h_pad - window_size + 1, stride))
    if h_pad - window_size not in y_steps: y_steps.append(h_pad - window_size)
    x_steps = list(range(0, w_pad - window_size + 1, stride))
    if w_pad - window_size not in x_steps: x_steps.append(w_pad - window_size)

    for y in y_steps:
        for x in x_steps:
            patch = image_tensor[:, :, y:y+window_size, x:x+window_size]
            with torch.no_grad():
                # 推理时 return_aux 默认为 False，preds 是一个 list
                preds = model(patch) 
                pred_patch = torch.sigmoid(preds[-1]) # 取最终的 refined_mask
            full_mask[:, :, y:y+window_size, x:x+window_size] += pred_patch
            count_map[:, :, y:y+window_size, x:x+window_size] += 1.0

    return (full_mask[:, :, :h, :w] / (count_map[:, :, :h, :w] + 1e-8))

def eval_sliding_window(val_list_path, image_root, gt_root, model, epoch, save_visuals=False):
    model.eval()
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    with open(val_list_path, 'r') as f:
        filenames = [line.strip() for line in f.readlines() if line.strip()]
    
    results = []
    for filename in tqdm(filenames, desc=f"Eval Ep {epoch}", leave=False):
        img_path = os.path.join(image_root, filename)
        gt_path = os.path.join(gt_root, filename)
        
        image = img_transform(Image.open(img_path).convert("RGB")).unsqueeze(0).cuda()
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        
        pred = sliding_window_inference(model, image).cpu().numpy().squeeze()
        pred_bw = (pred >= 0.5).astype(np.uint8) * 255
        
        iou, f1, prec, rec = calculate_metrics(pred_bw, gt)
        results.append([iou, f1, prec, rec])

        # 保存图片：预测 BW图 和 3合1 对比图 (Orig | GT | Pred)
        if save_visuals:
            cv2.imwrite(os.path.join(PRED_DIR, filename), pred_bw)
            
            orig_img = cv2.imread(img_path)
            gt_color = cv2.cvtColor(np.where(gt==255, 255, 0).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            pred_color = cv2.cvtColor(pred_bw, cv2.COLOR_GRAY2BGR)
            
            # 为三合一图添加文字标签
            def add_label(img, text):
                overlay = img.copy()
                cv2.rectangle(overlay, (0, 0), (280, 40), (0, 0, 0), -1)
                img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)
                cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                return img
                
            orig_img = add_label(orig_img, "Image")
            gt_color = add_label(gt_color, "Ground Truth")
            pred_color = add_label(pred_color, "ASER Pred")

            # 三合一图水平拼接
            comp = np.hstack((orig_img, gt_color, pred_color))
            cv2.imwrite(os.path.join(COMP_DIR, filename), comp)

    return np.mean(results, axis=0)

# ================= 4. 训练流程 =================
def train():
    epoch_num = 40  
    epoch_val = 2   
    train_size = 512 
    batch_size = 4 
    
    train_list_path = os.path.join(S2DS_DIR, f"fold{FOLD}_train.txt")
    val_list_path = os.path.join(S2DS_DIR, f"fold{FOLD}_val.txt")
    img_root = os.path.join(S2DS_DIR, "images", "")
    gt_root = os.path.join(S2DS_DIR, "labs", "")
    
    train_logger.info(f"{'='*20} 启动 ASER-lite Fold {FOLD} 训练 {'='*20}")
    
    # 确保此处的 dataloader 是 100% Patch 采样版本
    train_loader = get_loader(train_list_path, img_root, gt_root, batchsize=batch_size, trainsize=train_size, is_train=True)
    
    net = WPFormer(method="pvt_v2_b2", channel=64).cuda()
    if os.path.exists(PRETRAINED_WEIGHTS):
        net.load_state_dict(torch.load(PRETRAINED_WEIGHTS), strict=False)
        train_logger.info(f"✅ 成功加载 Stage 2 合成域预训练权重")

    optimizer = optim.Adam(net.parameters(), lr=1e-4)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=epoch_num, eta_min=1e-7)

    best_f1 = 0
    best_ckpt = ""

    for epoch in range(epoch_num):
        net.train()
        run_loss_seg, run_loss_edge, run_loss_amb, run_loss_tot = 0.0, 0.0, 0.0, 0.0
        
        pbar = tqdm(train_loader, desc=f"Ep {epoch}", leave=False)
        for data in pbar:
            images = data['image'].cuda()
            gts = data['label'].cuda()
            edges = data['edge'].cuda() 
            
            optimizer.zero_grad()
            
            # 开启 return_aux 获取辅助信息
            preds, edge_logits, ambiguity_logits = net(images, return_aux=True)
            
            # 1. 主分割 Loss 分离权重
            loss_aux = sum(total_loss(p, gts) for p in preds[:-1]) / len(preds[:-1])
            loss_refined = total_loss(preds[-1], gts)
            loss_seg = 0.5 * loss_aux + 1.0 * loss_refined
            
            # 2. 准备 Mask 与动态 GT
            valid_mask = (gts != 255).float()
            clean_gt = torch.where(gts == 255, torch.zeros_like(gts), gts)
            
            # 3. Edge Loss
            edge_bce = F.binary_cross_entropy_with_logits(edge_logits, edges, reduction='none')
            loss_edge = (edge_bce * valid_mask).sum() / (valid_mask.sum() + 1e-8)
            
            # 4. Ambiguity Loss (生成膨胀歧义环)
            dilated = F.max_pool2d(clean_gt, kernel_size=5, stride=1, padding=2)
            ambiguity_gt = (dilated - clean_gt).clamp(0, 1) * valid_mask
            
            amb_bce = F.binary_cross_entropy_with_logits(ambiguity_logits, ambiguity_gt, reduction='none')
            loss_amb = (amb_bce * valid_mask).sum() / (valid_mask.sum() + 1e-8)
            
            # 5. 总 Loss (辅助权重保守设为 0.1)
            loss = loss_seg + 0.1 * loss_edge + 0.1 * loss_amb
            
            loss.backward()
            optimizer.step()
            
            # 记录数据
            run_loss_seg += loss_seg.item()
            run_loss_edge += loss_edge.item()
            run_loss_amb += loss_amb.item()
            run_loss_tot += loss.item()
            
            pbar.set_postfix({'L_Tot': f"{loss.item():.3f}", 'L_Seg': f"{loss_seg.item():.3f}"})

        lr_scheduler.step()
        
        # 记录训练 Loss 到 train.log
        num_batches = len(train_loader)
        train_logger.info(f"Ep {epoch:02d} | L_Tot: {run_loss_tot/num_batches:.4f} | L_Seg: {run_loss_seg/num_batches:.4f} | L_Edg: {run_loss_edge/num_batches:.4f} | L_Amb: {run_loss_amb/num_batches:.4f}")
        
        # 定期评估
        if (epoch+1) >= epoch_val:
            avg_metrics = eval_sliding_window(val_list_path, img_root, gt_root, net, epoch, save_visuals=False)
            iou, f1, prec, rec = avg_metrics[0], avg_metrics[1], avg_metrics[2], avg_metrics[3]
            
            eval_msg = f"Epoch {epoch:02d} | IoU: {iou:.4f} | F1: {f1:.4f} | Pre: {prec:.4f} | Rec: {rec:.4f}"
            if f1 > best_f1:
                best_f1 = f1
                best_ckpt = os.path.join(CKPT_DIR, "WPFormer_aser_fold1_best.pth")
                torch.save(net.state_dict(), best_ckpt)
                eval_msg += " [★ F1 最优]"
            eval_logger.info(eval_msg)

    # 训练结束，使用最佳权重生成测试图
    train_logger.info(f"🏆 Fold 1 训练结束，最佳 F1: {best_f1:.4f}")
    eval_logger.info(f"🏆 开始生成最终预测图和三合一图 (基于最佳权重)...")
    
    net.load_state_dict(torch.load(best_ckpt))
    eval_sliding_window(val_list_path, img_root, gt_root, net, "Final", save_visuals=True)
    eval_logger.info(f"✅ 生成完毕，结果保存在: \n - 预测图: {PRED_DIR} \n - 三合一: {COMP_DIR}")

if __name__ == '__main__':
    train()