"""
=============================================================================
Experiment: Offline Threshold Scanning (离线阈值扫描)
Objective: 固定训练好的最佳模型权重，仅在推理后处理阶段扫描不同的二值化阈值，
           寻找 Precision 和 Recall 的完美平衡点，排除训练随机性的干扰。
=============================================================================
"""

import os
import cv2
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm

from model.WPFormer import WPFormer

# ================= 1. 配置区 =================
FOLD = 1
# 务必指向你之前 70/30 跑出 0.6495 的那个最优权重！
CKPT_PATH = "/home/skye/data/Skye/DA-WCA/save/stage3_patch_aser_7030_5fold/checkpoints/WPFormer_aser_fold1_best.pth"
S2DS_DIR = "/home/skye/data/Skye/databases/s2ds5"

# 要扫描的阈值列表
THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50]

# ================= 2. 核心机制 =================
def calculate_metrics(pred_bw, gt_img):
    valid_mask = (gt_img == 0) | (gt_img == 255)
    pred_valid = pred_bw[valid_mask]
    gt_valid = gt_img[valid_mask]
    
    if len(pred_valid) == 0: 
        return 0, 0, 0, 0
    
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

def sliding_window_inference(model, image_tensor, window_size=512, stride=256):
    b, c, h, w = image_tensor.size()
    full_mask = torch.zeros((b, 1, h, w), device=image_tensor.device)
    count_map = torch.zeros((b, 1, h, w), device=image_tensor.device)

    pad_h = max(0, window_size - h)
    pad_w = max(0, window_size - w)
    if pad_h > 0 or pad_w > 0:
        image_tensor = torch.nn.functional.pad(image_tensor, (0, pad_w, 0, pad_h), mode='reflect')
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
                preds = model(patch) 
                pred_patch = torch.sigmoid(preds[-1]) 
            full_mask[:, :, y:y+window_size, x:x+window_size] += pred_patch
            count_map[:, :, y:y+window_size, x:x+window_size] += 1.0

    return (full_mask[:, :, :h, :w] / (count_map[:, :, :h, :w] + 1e-8))

# ================= 3. 扫描流程 =================
def run_scan():
    print(f"\n{'='*50}")
    print(f"🚀 启动离线阈值扫描 (Fold {FOLD})")
    print(f"加载权重: {CKPT_PATH}")
    print(f"{'='*50}\n")

    # 1. 加载模型
    model = WPFormer(method="pvt_v2_b2", channel=64).cuda()
    model.load_state_dict(torch.load(CKPT_PATH))
    model.eval()

    # 2. 准备数据
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_list_path = os.path.join(S2DS_DIR, f"fold{FOLD}_val.txt")
    image_root = os.path.join(S2DS_DIR, "images")
    gt_root = os.path.join(S2DS_DIR, "labs")

    with open(val_list_path, 'r') as f:
        filenames = [line.strip() for line in f.readlines() if line.strip()]

    # 3. 初始化结果字典
    results = {t: [] for t in THRESHOLDS}

    # 4. 开始扫描
    for filename in tqdm(filenames, desc="扫描验证集"):
        img_path = os.path.join(image_root, filename)
        gt_path = os.path.join(gt_root, filename)
        
        image = img_transform(Image.open(img_path).convert("RGB")).unsqueeze(0).cuda()
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        
        # 核心：只推理一次，得到浮点数概率图
        pred_prob = sliding_window_inference(model, image).cpu().numpy().squeeze()
        
        # 遍历所有阈值进行切割和评测
        for t in THRESHOLDS:
            pred_bw = (pred_prob >= t).astype(np.uint8) * 255
            iou, f1, prec, rec = calculate_metrics(pred_bw, gt)
            results[t].append([iou, f1, prec, rec])

    # 5. 打印对比表格
    print("\n\n" + "="*60)
    print(f"📊 离线阈值扫描结果汇总 (基于同一 Checkpoint)")
    print("="*60)
    print(f"{'Threshold':<10} | {'F1 Score':<10} | {'Precision':<10} | {'Recall':<10} | {'IoU':<10}")
    print("-" * 60)
    
    best_t, best_f1 = 0.50, 0.0
    
    for t in THRESHOLDS:
        avg_metrics = np.mean(results[t], axis=0)
        iou, f1, prec, rec = avg_metrics[0], avg_metrics[1], avg_metrics[2], avg_metrics[3]
        
        marker = "⭐" if f1 > best_f1 else "  "
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
            
        print(f"{t:<10.2f} | {f1:<10.4f} | {prec:<10.4f} | {rec:<10.4f} | {iou:<10.4f} {marker}")
        
    print("="*60)
    print(f"🏆 结论: 最佳判定阈值为 {best_t:.2f}，对应的最高 F1 为 {best_f1:.4f}")
    print("="*60)

if __name__ == '__main__':
    run_scan()