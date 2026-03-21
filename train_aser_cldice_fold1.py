"""
=============================================================================
Experiment: ASER-lite + Soft-clDice (Fold 1 Only)
Objective: 针对极细裂缝的断连问题，在最终的 refined_mask 上引入极轻量(0.05)
           的拓扑保持损失(Soft-clDice)，测试能否在不损害 Precision 的情况下
           强行拉升细微裂缝的 Recall 和连通性。
Strategy: 
  - Backbone: WPFormer (PVTv2)
  - Sampling: 70/30 Patch (ESDI_dataloader.py with p=0.7)
  - Loss: Seg(0.5/1.0) + Edge(0.1) + Ambiguity(0.1) + clDice(0.05)
=============================================================================
"""

import numpy as np
import torch
import torch.nn as nn
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

# ================= Soft-clDice 拓扑损失实现 =================
class SoftSkeletonize(nn.Module):
    def __init__(self, num_iter=3):
        super(SoftSkeletonize, self).__init__()
        self.num_iter = num_iter

    def soft_erode(self, img):
        p1 = -F.max_pool2d(-img, (3,1), (1,1), (1,0))
        p2 = -F.max_pool2d(-img, (1,3), (1,1), (0,1))
        return torch.min(p1, p2)

    def soft_dilate(self, img):
        return F.max_pool2d(img, (3,3), (1,1), (1,1))

    def soft_open(self, img):
        return self.soft_dilate(self.soft_erode(img))

    def forward(self, img):
        img1 = img
        skel = torch.zeros_like(img)
        for i in range(self.num_iter):
            eroded = self.soft_erode(img1)
            opened = self.soft_open(eroded)
            skel = torch.max(skel, eroded - opened)
            img1 = eroded
        return skel

def soft_cldice_loss(pred, target):
    """
    计算预测与目标的拓扑连通性损失 (值越小越好)
    """
    skel_func = SoftSkeletonize(num_iter=3)
    pred_skel = skel_func(pred)
    target_skel = skel_func(target)
    
    tprec = (pred_skel * target).sum() / (pred_skel.sum() + 1e-8)
    tsens = (target_skel * pred).sum() / (target_skel.sum() + 1e-8)
    
    cl_dice = 2.0 * tprec * tsens / (tprec + tsens + 1e-8)
    return 1.0 - cl_dice
# =========================================================

# ================= 1. 配置区 =================
# 单独的输出目录，防止覆盖之前的战果
BASE_DIR = "/home/skye/data/Skye/DA-WCA/save/stage3_patch_aser_cldice_fold1"
S2DS_DIR = "/home/skye/data/Skye/databases/s2ds5"
PRETRAINED_WEIGHTS = "/home/skye/data/Skye/DA-WCA/save/stage2_512/checkpoints/WPFormer_synth_512_best.pth"

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
                preds = model(patch) 
                pred_patch = torch.sigmoid(preds[-1]) 
            full_mask[:, :, y:y+window_size, x:x+window_size] += pred_patch
            count_map[:, :, y:y+window_size, x:x+window_size] += 1.0

    return (full_mask[:, :, :h, :w] / (count_map[:, :, :h, :w] + 1e-8))

def eval_sliding_window(val_list_path, image_root, gt_root, model, fold, epoch, save_visuals=False):
    model.eval()
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    with open(val_list_path, 'r') as f:
        filenames = [line.strip() for line in f.readlines() if line.strip()]
    
    results = []
    for filename in tqdm(filenames, desc=f"Eval Fold {fold} Ep {epoch}", leave=False):
        img_path = os.path.join(image_root, filename)
        gt_path = os.path.join(gt_root, filename)
        
        image = img_transform(Image.open(img_path).convert("RGB")).unsqueeze(0).cuda()
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        
        pred = sliding_window_inference(model, image).cpu().numpy().squeeze()
        # 注意：这里依然保持绝对标准的 0.5 判定阈值
        pred_bw = (pred >= 0.5).astype(np.uint8) * 255
        
        iou, f1, prec, rec = calculate_metrics(pred_bw, gt)
        results.append([iou, f1, prec, rec])

        if save_visuals:
            cv2.imwrite(os.path.join(PRED_DIR, filename), pred_bw)
            
            orig_img = cv2.imread(img_path)
            gt_color = cv2.cvtColor(np.where(gt==255, 255, 0).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            pred_color = cv2.cvtColor(pred_bw, cv2.COLOR_GRAY2BGR)
            
            def add_label(img, text):
                overlay = img.copy()
                cv2.rectangle(overlay, (0, 0), (280, 40), (0, 0, 0), -1)
                img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)
                cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                return img
                
            orig_img = add_label(orig_img, "Image")
            gt_color = add_label(gt_color, "Ground Truth")
            pred_color = add_label(pred_color, "clDice Pred")

            comp = np.hstack((orig_img, gt_color, pred_color))
            cv2.imwrite(os.path.join(COMP_DIR, filename), comp)

    return np.mean(results, axis=0)

# ================= 4. 训练流程 =================
def train_one_fold(fold):
    epoch_num = 40  
    epoch_val = 2   
    train_size = 512 
    batch_size = 4  
    
    train_list_path = os.path.join(S2DS_DIR, f"fold{fold}_train.txt")
    val_list_path = os.path.join(S2DS_DIR, f"fold{fold}_val.txt")
    img_root = os.path.join(S2DS_DIR, "images", "")
    gt_root = os.path.join(S2DS_DIR, "labs", "")
    
    train_logger.info(f"\n{'='*20} 启动 ASER + clDice Fold {fold} 训练 {'='*20}")
    
    train_loader = get_loader(train_list_path, img_root, gt_root, batchsize=batch_size, trainsize=train_size, is_train=True)
    
    net = WPFormer(method="pvt_v2_b2", channel=64).cuda()
    if os.path.exists(PRETRAINED_WEIGHTS):
        net.load_state_dict(torch.load(PRETRAINED_WEIGHTS), strict=False)
        train_logger.info(f"✅ Fold {fold}: 成功加载合成域权重")

    optimizer = optim.Adam(net.parameters(), lr=1e-4)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=epoch_num, eta_min=1e-7)

    best_f1 = 0
    best_ckpt = ""

    for epoch in range(epoch_num):
        net.train()
        run_loss_seg, run_loss_edge, run_loss_amb, run_loss_cldice, run_loss_tot = 0.0, 0.0, 0.0, 0.0, 0.0
        
        pbar = tqdm(train_loader, desc=f"Fold {fold} Ep {epoch}", leave=False)
        for data in pbar:
            images = data['image'].cuda()
            gts = data['label'].cuda()
            edges = data['edge'].cuda() 
            
            optimizer.zero_grad()
            
            preds, edge_logits, ambiguity_logits = net(images, return_aux=True)
            
            # 1. 主分割 Loss (保持 70/30 最佳配置 0.5 / 1.0)
            loss_aux = sum(total_loss(p, gts) for p in preds[:-1]) / len(preds[:-1])
            loss_refined = total_loss(preds[-1], gts)
            loss_seg = 0.5 * loss_aux + 1.0 * loss_refined
            
            # 2. 掩码准备
            valid_mask = (gts != 255).float()
            clean_gt = torch.where(gts == 255, torch.zeros_like(gts), gts)
            
            # 3. 边缘与歧义 Loss
            edge_bce = F.binary_cross_entropy_with_logits(edge_logits, edges, reduction='none')
            loss_edge = (edge_bce * valid_mask).sum() / (valid_mask.sum() + 1e-8)
            
            dilated = F.max_pool2d(clean_gt, kernel_size=5, stride=1, padding=2)
            ambiguity_gt = (dilated - clean_gt).clamp(0, 1) * valid_mask
            amb_bce = F.binary_cross_entropy_with_logits(ambiguity_logits, ambiguity_gt, reduction='none')
            loss_amb = (amb_bce * valid_mask).sum() / (valid_mask.sum() + 1e-8)
            
            # ================= 4. 新增: 拓扑保持损失 (clDice) =================
            pred_refined_sig = torch.sigmoid(preds[-1])
            # 注意：仅对非 ignore 区域提取骨架
            loss_cldice = soft_cldice_loss(pred_refined_sig * valid_mask, clean_gt * valid_mask)
            # ================================================================
            
            # 5. 组合最终 Loss (clDice 权重极轻: 0.05)
            loss = loss_seg + 0.1 * loss_edge + 0.1 * loss_amb + 0.05 * loss_cldice
            
            loss.backward()
            optimizer.step()
            
            run_loss_seg += loss_seg.item()
            run_loss_edge += loss_edge.item()
            run_loss_amb += loss_amb.item()
            run_loss_cldice += loss_cldice.item()
            run_loss_tot += loss.item()
            
            pbar.set_postfix({'L_Tot': f"{loss.item():.3f}"})

        lr_scheduler.step()
        
        num_batches = len(train_loader)
        train_logger.info(f"Ep {epoch:02d} | Tot:{run_loss_tot/num_batches:.3f} | Seg:{run_loss_seg/num_batches:.3f} | Edg:{run_loss_edge/num_batches:.3f} | Amb:{run_loss_amb/num_batches:.3f} | clD:{run_loss_cldice/num_batches:.3f}")
        
        if (epoch+1) >= epoch_val:
            avg_metrics = eval_sliding_window(val_list_path, img_root, gt_root, net, fold, epoch, save_visuals=False)
            iou, f1, prec, rec = avg_metrics[0], avg_metrics[1], avg_metrics[2], avg_metrics[3]
            
            eval_msg = f"Fold {fold} | Epoch {epoch:02d} | IoU: {iou:.4f} | F1: {f1:.4f} | Pre: {prec:.4f} | Rec: {rec:.4f}"
            if f1 > best_f1:
                best_f1 = f1
                best_ckpt = os.path.join(CKPT_DIR, f"WPFormer_cldice_fold{fold}_best.pth")
                torch.save(net.state_dict(), best_ckpt)
                eval_msg += " [★ F1 最优]"
            eval_logger.info(eval_msg)

    train_logger.info(f"🏆 Fold {fold} 训练结束，最佳 F1: {best_f1:.4f}")
    eval_logger.info(f"🏆 Fold {fold} 开始生成最终预测图 (基于最佳权重)...")
    
    net.load_state_dict(torch.load(best_ckpt))
    eval_sliding_window(val_list_path, img_root, gt_root, net, fold, "Final", save_visuals=True)
    
    return best_f1

if __name__ == '__main__':
    # 【核心！】只跑 Fold 1 验资
    train_one_fold(1)