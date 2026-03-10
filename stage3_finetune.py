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
from sod_metrics import Smeasure, MAE

# ================= 目录初始化 =================
BASE_DIR = "./save/stage3_finetune"
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")
PRED_DIR = os.path.join(BASE_DIR, "predictions")
COMP_DIR = os.path.join(BASE_DIR, "composites_bw")

S2DS_DIR = "/home/skye/data/Skye/databases/s2ds5"
# 指定 Stage 2 训练出的合成域最优权重路径
PRETRAINED_WEIGHTS = "./save/stage2_zeroshot/checkpoints/WPFormer_synth_best.pth"

os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)
os.makedirs(COMP_DIR, exist_ok=True)

def get_logger(filename):
    logger = logging.getLogger(filename)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(filename, mode='w', encoding='utf-8')
        fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger

train_logger = get_logger(os.path.join(BASE_DIR, "train.txt"))
eval_logger = get_logger(os.path.join(BASE_DIR, "eval.txt"))

# ================= 指标与损失计算 (带掩码屏蔽) =================
def calculate_metrics(pred, gt_img):
    # 严格屏蔽 149/225 等忽略区域
    valid_mask = (gt_img == 0) | (gt_img == 255)
    pred_valid = pred[valid_mask]
    gt_valid = gt_img[valid_mask]
    
    if len(pred_valid) == 0: return 0, 0, 0, 0

    gt_bin = (gt_valid == 255).astype(np.float32)
    pred_bin = (pred_valid > 0.5).astype(np.float32)
    
    tp = (pred_bin * gt_bin).sum()
    fp = (pred_bin * (1 - gt_bin)).sum()
    fn = ((1 - pred_bin) * gt_bin).sum()
    
    iou = tp / (tp + fp + fn + 1e-8)
    f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)

    gt_clean = np.where(gt_img == 255, 255, 0).astype(np.uint8)
    _MAE = MAE()
    _MAE.step(pred=pred, gt=gt_clean)
    mae = _MAE.get_results()["mae"]

    _SM = Smeasure()
    _SM.step(pred=pred, gt=gt_clean)
    sm = _SM.get_results()["sm"]

    return mae, iou, f1, sm

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

# ================= 微调与评估流 =================
def evaluate_best_model(fold, val_list_path, image_root, gt_root, ckpt_path):
    eval_logger.info(f"\n--- 正在评估 Fold {fold} 最佳微调模型 ---")
    
    model = WPFormer(method="pvt_v2_b2", channel=64).cuda()
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()

    img_transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    fold_pred = os.path.join(PRED_DIR, f"fold{fold}")
    fold_comp = os.path.join(COMP_DIR, f"fold{fold}")
    os.makedirs(fold_pred, exist_ok=True)
    os.makedirs(fold_comp, exist_ok=True)

    with open(val_list_path, 'r') as f:
        filenames = [line.strip() for line in f.readlines()]

    results = []
    for filename in tqdm(filenames, desc=f"Evaluating S2DS Fold {fold}"):
        img_path = os.path.join(image_root, filename)
        gt_path = os.path.join(gt_root, filename)
        
        ori_img = cv2.imread(img_path)
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        H, W = gt.shape

        input_tensor = img_transform(Image.open(img_path).convert("RGB")).unsqueeze(0).cuda()
        with torch.no_grad():
            res = torch.sigmoid(model(input_tensor)[-1]).cpu().numpy().squeeze()
        
        pred = cv2.resize(res, (W, H))
        results.append(calculate_metrics(pred, gt))

        # 保存黑白预测图
        pred_bw = np.where(pred >= 0.5, 255, 0).astype(np.uint8)
        cv2.imwrite(os.path.join(fold_pred, filename), pred_bw)

        # 生成纯黑白三合一图 [原图 | 真值(白) | 预测(白)]
        gt_bw = np.where(gt == 255, 255, 0).astype(np.uint8)
        gt_3c = cv2.cvtColor(gt_bw, cv2.COLOR_GRAY2BGR)
        pred_3c = cv2.cvtColor(pred_bw, cv2.COLOR_GRAY2BGR)
        comp = np.hstack((cv2.resize(ori_img, (W, H)), gt_3c, pred_3c))
        cv2.imwrite(os.path.join(fold_comp, filename), comp)

    avg = np.mean(results, axis=0)
    eval_logger.info(f"S2DS Fold {fold} Fine-Tuned -> MAE: {avg[0]:.4f}, IoU: {avg[1]:.4f}, F1: {avg[2]:.4f}, SM: {avg[3]:.4f}")
    
    # 核心返回值增加 IoU
    return avg[1], avg[2]

def finetune_fold(fold):
    # 【优化点2】: epoch 降至 40
    epoch_num = 40
    epoch_val = 2  # 前期即可开始密集验证
    train_size = 384
    base_lr = 2e-5 

    train_list_path = os.path.join(S2DS_DIR, f"fold{fold}_train.txt")
    val_list_path = os.path.join(S2DS_DIR, f"fold{fold}_val.txt")
    img_root = os.path.join(S2DS_DIR, "images", "")
    gt_root = os.path.join(S2DS_DIR, "labs", "")

    train_loader = get_loader(train_list_path, img_root, gt_root, batchsize=8, trainsize=train_size, is_train=True)
    
    net = WPFormer(method="pvt_v2_b2", channel=64).cuda()
    
    if os.path.exists(PRETRAINED_WEIGHTS):
        net.load_state_dict(torch.load(PRETRAINED_WEIGHTS))
        train_logger.info(f"✅ Fold {fold}: 成功加载合成域预训练权重")
    else:
        raise FileNotFoundError(f"找不到预训练权重：{PRETRAINED_WEIGHTS}")

    optimizer = optim.Adam(net.parameters(), lr=base_lr)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=epoch_num, eta_min=1e-7)

    train_logger.info(f"\n========== 启动 S2DS Fold {fold} 微调 ==========")
    
    # 【优化点3 & 4】: 记录 best_epoch 和对应的 best_iou
    best_f1 = 0.0
    best_iou = 0.0
    best_epoch = 0
    best_ckpt = ""

    for epoch in range(epoch_num):
        start_time = time.time()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Finetune Fold {fold} Ep {epoch}", leave=False)
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
            status = f"Epoch {epoch:02d} | Loss: {running_loss/len(train_loader):.4f} | S2DS-MAE: {mae:.4f} | S2DS-IoU: {iou:.4f} | S2DS-F1: {f1:.4f} | S2DS-SM: {sm:.4f}"
            
            if f1 > best_f1:
                best_f1 = f1
                best_iou = iou
                best_epoch = epoch
                best_ckpt = os.path.join(CKPT_DIR, f"WPFormer_finetune_fold{fold}_best.pth")
                torch.save(net.state_dict(), best_ckpt)
                status += " [★ 微调最优]"
            train_logger.info(status)
        net.train()
    
    # 【优化点3结语】: 在每一折结束时，专门打印这一折的终极汇总
    train_logger.info(f"🏆 Fold {fold} 微调结束 | Best Epoch: {best_epoch:02d} | Best F1: {best_f1:.4f} | Best IoU: {best_iou:.4f}\n")
        
    return best_ckpt, val_list_path, img_root, gt_root

if __name__ == '__main__':
    all_ious = []
    all_f1s = []
    for f in range(1, 6):
        best_ckpt, val_txt, img_dir, gt_dir = finetune_fold(f)
        iou_score, f1_score = evaluate_best_model(f, val_txt, img_dir, gt_dir, best_ckpt)
        all_ious.append(iou_score)
        all_f1s.append(f1_score)
        
    # 【优化点4】: 将平均 F1 和平均 IoU 同时作为主结论打印
    eval_logger.info(f"\n=> 🎯 最终 S2DS 5-Fold 微调平均指标 -> mIoU: {np.mean(all_ious):.4f} | F1-Score: {np.mean(all_f1s):.4f}")