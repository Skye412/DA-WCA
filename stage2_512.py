import numpy as np
import torch
import torch.optim as optim
import os
import logging
from PIL import Image
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
from model.WPFormer import WPFormer
from ESDI_dataloader import get_loader
import cv2
import torch.nn.functional as F
import time
from tqdm import tqdm
import random
from sod_metrics import Smeasure, MAE

# ================= 目录初始化 =================
BASE_DIR = "./save/stage2_512"
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")

SYNTH_DIR = "/home/skye/data/Skye/databases/synthcrack"

os.makedirs(CKPT_DIR, exist_ok=True)

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

train_logger = get_logger(os.path.join(BASE_DIR, "train_synth.txt"))

# ================= 合成数据预处理 =================
def prepare_synth_data():
    synth_img_dir = os.path.join(SYNTH_DIR, "images")
    train_txt = os.path.join(BASE_DIR, "synth_train.txt")
    val_txt = os.path.join(BASE_DIR, "synth_val.txt")
    
    if os.path.exists(train_txt) and os.path.exists(val_txt):
        return train_txt, val_txt
        
    train_logger.info("--- 正在划分 Synthcrack 数据集 (80% Train, 20% Val) ---")
    img_files = [f for f in os.listdir(synth_img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    img_files.sort()
    
    random.seed(42)
    random.shuffle(img_files)
    
    split_idx = int(len(img_files) * 0.8)
    train_files = img_files[:split_idx]
    val_files = img_files[split_idx:]
    
    with open(train_txt, 'w') as f:
        f.write('\n'.join(train_files) + '\n')
    with open(val_txt, 'w') as f:
        f.write('\n'.join(val_files) + '\n')
        
    train_logger.info(f"划分完成: Train {len(train_files)} 张, Val {len(val_files)} 张")
    return train_txt, val_txt

def calculate_metrics(pred, gt_img):
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
        transforms.Resize((512, 512)),
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

        if not os.path.exists(gt_path):
            gt_path = os.path.splitext(gt_path)[0] + '.png'

        image = img_transform(Image.open(img_path).convert("RGB")).unsqueeze(0).cuda()
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        
        with torch.no_grad():
            res = model(image)[-1]

        res = torch.sigmoid(res).data.cpu().numpy().squeeze()
        pred = cv2.resize(res, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
        results.append(calculate_metrics(pred, gt))

    return np.mean(results, axis=0)

def train_synth():
    train_txt, val_txt = prepare_synth_data()
    
    epoch_num = 30  # 合成数据量大，30轮即可
    epoch_val = 2   
    train_size = 512 # 【核心修改：512分辨率】
    batch_size = 8   # 【核心修改：适配16G显存】
    
    img_root = os.path.join(SYNTH_DIR, "images", "")
    gt_root = os.path.join(SYNTH_DIR, "gt", "")

    train_loader = get_loader(train_txt, img_root, gt_root, batchsize=batch_size, trainsize=train_size, is_train=True)
    net = WPFormer(method="pvt_v2_b2", channel=64).cuda()

    optimizer = optim.Adam(net.parameters(), lr=1e-4)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=epoch_num, eta_min=1e-7)

    train_logger.info("\n========== 开始 Stage 2: Synthcrack 512px 预训练 ==========")
    best_sm = 0
    best_f1 = 0
    best_ckpt = ""

    for epoch in range(epoch_num):
        start_time = time.time()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Synth Train Ep {epoch}", leave=False)
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
        
        if (epoch+1) >= epoch_val or epoch == epoch_num - 1:
            mae, iou, f1, sm = eval_validation(val_txt, img_root, gt_root, net)
            status = f"Epoch {epoch:02d} | Loss: {running_loss/len(train_loader):.4f} | Synth-IoU: {iou:.4f} | Synth-F1: {f1:.4f} | Synth-SM: {sm:.4f}"
            
            if f1 > best_f1:
                best_f1 = f1
                best_sm = sm
                best_ckpt = os.path.join(CKPT_DIR, f"WPFormer_synth_512_best.pth")
                torch.save(net.state_dict(), best_ckpt)
                status += " [★ Synth F1最优]"
            train_logger.info(status)
        net.train()
        
    train_logger.info(f"预训练完成！权重已保存至: {best_ckpt}")
    return best_ckpt

if __name__ == '__main__':
    train_synth()