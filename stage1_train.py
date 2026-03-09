import numpy as np
import torch
from torch.autograd import Variable
from torchvision import transforms
import torch.optim as optim
import os
import logging
from PIL import Image
from torch.optim.lr_scheduler import CosineAnnealingLR
from model.WPFormer import WPFormer
from ESDI_dataloader import get_loader
import cv2
import torch.nn.functional as F
import time
from tqdm import tqdm
# 导入原文使用的评估库
from sod_metrics import Smeasure, MAE

# ================= 目录初始化 =================
BASE_DIR = "./save/stage1_s2ds"
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")
PRED_DIR = os.path.join(BASE_DIR, "predictions")
COMP_DIR = os.path.join(BASE_DIR, "composites")

os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)
os.makedirs(COMP_DIR, exist_ok=True)

def get_logger(filename):
    logger = logging.getLogger(filename)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(filename, mode='a', encoding='utf-8')
        fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger

train_logger = get_logger(os.path.join(BASE_DIR, "train.txt"))
eval_logger = get_logger(os.path.join(BASE_DIR, "eval.txt"))

def calculate_metrics(pred, gt_img):
    """
    修正后的评估逻辑：
    gt_img 是原始灰度图 [0, 149/225, 255]
    """
    # 1. 确定有效区域：背景(0)和裂缝(255)是有效的，其他中间值(149等)屏蔽
    valid_mask = (gt_img == 0) | (gt_img == 255)
    
    # 2. 提取有效像素
    pred_valid = pred[valid_mask]
    gt_valid = gt_img[valid_mask]
    
    if len(pred_valid) == 0: return 0, 0, 0, 0

    # 3. 转换为二值标签进行 IoU/F1 计算
    # 裂缝像素在原始图中是 255
    gt_bin = (gt_valid == 255).astype(np.float32)
    pred_bin = (pred_valid > 0.5).astype(np.float32)
    
    tp = (pred_bin * gt_bin).sum()
    fp = (pred_bin * (1 - gt_bin)).sum()
    fn = ((1 - pred_bin) * gt_bin).sum()
    
    iou = tp / (tp + fp + fn + 1e-8)
    f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)

    # 4. S-measure 和 MAE 同样只针对裂缝计算
    gt_clean = np.where(gt_img == 255, 255, 0).astype(np.uint8)
    
    _MAE = MAE()
    _MAE.step(pred=pred, gt=gt_clean)
    mae = _MAE.get_results()["mae"]

    _SM = Smeasure()
    _SM.step(pred=pred, gt=gt_clean)
    sm = _SM.get_results()["sm"]

    return mae, iou, f1, sm

def eval_validation(val_list_path, image_root, gt_root, model):
    img_transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    with open(val_list_path, 'r') as f:
        filenames = [line.strip() for line in f.readlines() if line.strip()]
    
    model.eval()
    results = []

    for filename in filenames:
        img_path = os.path.join(image_root, filename)
        gt_path = os.path.join(gt_root, filename)

        image = img_transform(Image.open(img_path).convert("RGB")).unsqueeze(0).cuda()
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        
        with torch.no_grad():
            res = model(image)[-1]

        res = torch.sigmoid(res).data.cpu().numpy().squeeze()
        pred = cv2.resize(res, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)

        results.append(calculate_metrics(pred, gt))

    return np.mean(results, axis=0)

# ================= 训练与保存 =================
def total_loss(pred, mask):
    pred = torch.sigmoid(pred)
    valid_mask = (mask != 255).float()
    clean_mask = torch.where(mask == 255, torch.zeros_like(mask), mask)
    
    bce = F.binary_cross_entropy(pred, clean_mask, reduction='none')
    bce = (bce * valid_mask).sum() / (valid_mask.sum() + 1e-8)

    inter = (pred * clean_mask * valid_mask).sum(dim=(2, 3))
    union = ((pred + clean_mask) * valid_mask).sum(dim=(2, 3))
    iou = 1 - inter / (union - inter + 1e-8)
    
    return iou.mean() + bce

def train(fold):
    epoch_num = 60
    epoch_val = 10 
    train_size = 384
    file_dir = "/home/skye/data/Skye/databases/s2ds5/"

    train_list_path = os.path.join(file_dir, f"fold{fold}_train.txt")
    val_list_path = os.path.join(file_dir, f"fold{fold}_val.txt")
    img_root = os.path.join(file_dir, "images", "")
    gt_root = os.path.join(file_dir, "labs", "")

    train_loader = get_loader(train_list_path, img_root, gt_root, batchsize=8, trainsize=train_size, is_train=True)
    net = WPFormer(method="pvt_v2_b2", channel=64).cuda()

    optimizer = optim.Adam(net.parameters(), lr=8e-5)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=epoch_num, eta_min=1e-7)

    train_logger.info(f"\n--- 启动 Fold {fold} 训练 (Stage 1) ---")
    best_sm = 0
    best_ckpt = ""

    for epoch in range(epoch_num):
        start_time = time.time()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Fold {fold} Ep {epoch}", leave=False)
        for data in pbar:
            images = data['image'].cuda()
            gts = data['label'].cuda()
            
            optimizer.zero_grad()
            preds = net(images)

            loss = sum(total_loss(p, gts) for p in preds)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({'Loss': f"{loss.item():.4f}"})

        lr_scheduler.step()
        
        if (epoch+1) >= epoch_val:
            mae, iou, f1, sm = eval_validation(val_list_path, img_root, gt_root, net)
            status = f"Epoch {epoch:02d} | Loss: {running_loss/len(train_loader):.4f} | MAE: {mae:.4f} | IoU: {iou:.4f} | F1: {f1:.4f} | SM: {sm:.4f}"
            
            if sm > best_sm:
                best_sm = sm
                best_ckpt = os.path.join(CKPT_DIR, f"WPFormer_fold{fold}_best.pth")
                torch.save(net.state_dict(), best_ckpt)
                status += " [★]"
            train_logger.info(status)
        net.train()

    # 该折结束后生成评估图
    generate_results(fold, val_list_path, img_root, gt_root, best_ckpt)

def generate_results(fold, list_path, img_root, gt_root, ckpt):
    eval_logger.info(f"--- 评估 Fold {fold} 最佳模型 ---")
    model = WPFormer(method="pvt_v2_b2", channel=64).cuda()
    model.load_state_dict(torch.load(ckpt))
    model.eval()

    with open(list_path, 'r') as f:
        filenames = [line.strip() for line in f.readlines()]

    fold_pred = os.path.join(PRED_DIR, f"fold{fold}")
    fold_comp = os.path.join(COMP_DIR, f"fold{fold}")
    os.makedirs(fold_pred, exist_ok=True)
    os.makedirs(fold_comp, exist_ok=True)

    img_transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    results = []
    for filename in tqdm(filenames, desc="Saving Images"):
        img_path = os.path.join(img_root, filename)
        gt_path = os.path.join(gt_root, filename)
        
        raw_img = cv2.imread(img_path)
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        H, W = gt.shape

        input_tensor = img_transform(Image.open(img_path).convert("RGB")).unsqueeze(0).cuda()
        with torch.no_grad():
            res = torch.sigmoid(model(input_tensor)[-1]).cpu().numpy().squeeze()
        
        pred = cv2.resize(res, (W, H))
        results.append(calculate_metrics(pred, gt))

        # 保存图片
        pred_u8 = (pred * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(fold_pred, filename), pred_u8)

        # 拼接原图/GT/预测
        gt_color = cv2.applyColorMap(gt, cv2.COLORMAP_PARULA)
        pred_color = cv2.applyColorMap(pred_u8, cv2.COLORMAP_JET)
        comp = np.hstack((cv2.resize(raw_img, (W, H)), gt_color, pred_color))
        cv2.imwrite(os.path.join(fold_comp, filename), comp)

    avg = np.mean(results, axis=0)
    eval_logger.info(f"Fold {fold} Avg -> MAE: {avg[0]:.4f}, IoU: {avg[1]:.4f}, F1: {avg[2]:.4f}, SM: {avg[3]:.4f}")

if __name__ == '__main__':
    for f in range(1, 6):
        train(f)