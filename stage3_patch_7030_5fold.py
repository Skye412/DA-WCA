import numpy as np
import torch
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
from tqdm import tqdm

# ================= 配置区 =================
# 建立独立的文件夹，防止覆盖之前 100% 采样的 0.6411 结果
BASE_DIR = "./save/stage3_patch_7030"
S2DS_DIR = "/home/skye/data/Skye/databases/s2ds5"
PRETRAINED_WEIGHTS = "./save/stage2_512/checkpoints/WPFormer_synth_512_best.pth"

os.makedirs(BASE_DIR, exist_ok=True)

def get_logger(filename):
    logger = logging.getLogger("train_logger_7030")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        logger.handlers.clear()
    fh = logging.FileHandler(filename, mode='a', encoding='utf-8')
    fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

LOG_FILE = os.path.join(BASE_DIR, "train_7030.txt")

# ================= 核心：滑窗推理机制 =================
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

def calculate_metrics(pred, gt_img):
    valid_mask = (gt_img == 0) | (gt_img == 255)
    pred_valid = pred[valid_mask]
    gt_valid = gt_img[valid_mask]
    if len(pred_valid) == 0: return 0, 0
    gt_bin = (gt_valid == 255).astype(np.float32)
    pred_bin = (pred_valid > 0.5).astype(np.float32)
    tp = (pred_bin * gt_bin).sum()
    fp = (pred_bin * (1 - gt_bin)).sum()
    fn = ((1 - pred_bin) * gt_bin).sum()
    iou = tp / (tp + fp + fn + 1e-8)
    f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)
    return iou, f1

def total_loss(pred, mask):
    pred = torch.sigmoid(pred)
    valid_mask = (mask != 255).float()
    clean_mask = torch.where(mask == 255, torch.zeros_like(mask), mask)
    bce = (F.binary_cross_entropy(pred, clean_mask, reduction='none') * valid_mask).sum() / (valid_mask.sum() + 1e-8)
    inter = (pred * clean_mask * valid_mask).sum(dim=(2, 3))
    union = ((pred + clean_mask) * valid_mask).sum(dim=(2, 3))
    iou = 1 - inter / (union - inter + 1e-8)
    return iou.mean() + bce

def eval_sliding_window(val_list_path, image_root, gt_root, model, fold, save_visuals=False):
    model.eval()
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    with open(val_list_path, 'r') as f:
        filenames = [line.strip() for line in f.readlines() if line.strip()]
    
    results = []
    fold_pred = os.path.join(BASE_DIR, "predictions", f"fold{fold}")
    fold_comp = os.path.join(BASE_DIR, "composites", f"fold{fold}")
    if save_visuals:
        os.makedirs(fold_pred, exist_ok=True)
        os.makedirs(fold_comp, exist_ok=True)

    for filename in tqdm(filenames, desc=f"Fold {fold} Eval", leave=False):
        img_path = os.path.join(image_root, filename)
        gt_path = os.path.join(gt_root, filename)
        image = img_transform(Image.open(img_path).convert("RGB")).unsqueeze(0).cuda()
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        
        pred = sliding_window_inference(model, image).cpu().numpy().squeeze()
        iou, f1 = calculate_metrics(pred, gt)
        results.append([iou, f1])

        if save_visuals:
            pred_bw = (pred >= 0.5).astype(np.uint8) * 255
            cv2.imwrite(os.path.join(fold_pred, filename), pred_bw)
            gt_bw = (gt == 255).astype(np.uint8) * 255
            comp = np.hstack((cv2.imread(img_path), cv2.cvtColor(gt_bw, cv2.COLOR_GRAY2BGR), cv2.cvtColor(pred_bw, cv2.COLOR_GRAY2BGR)))
            cv2.imwrite(os.path.join(fold_comp, filename), comp)

    return np.mean(results, axis=0)

def train_one_fold(fold):
    epoch_num = 40  
    epoch_val = 2   
    train_size = 512 
    batch_size = 8
    
    train_list_path = os.path.join(S2DS_DIR, f"fold{fold}_train.txt")
    val_list_path = os.path.join(S2DS_DIR, f"fold{fold}_val.txt")
    img_root = os.path.join(S2DS_DIR, "images", "")
    gt_root = os.path.join(S2DS_DIR, "labs", "")
    
    ckpt_dir = os.path.join(BASE_DIR, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    
    train_logger = get_logger(LOG_FILE)
    train_logger.info(f"\n{'='*20} 启动 Fold {fold} 70/30 Patch 训练 {'='*20}")

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
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Fold {fold} Ep {epoch}", leave=False)
        for data in pbar:
            images, gts = data['image'].cuda(), data['label'].cuda()
            optimizer.zero_grad()
            preds = net(images)
            loss = sum(total_loss(p, gts) for p in preds)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            pbar.set_postfix({'Loss': f"{loss.item():.4f}"})

        lr_scheduler.step()
        
        if (epoch+1) >= epoch_val:
            iou, f1 = eval_sliding_window(val_list_path, img_root, gt_root, net, fold, save_visuals=False)
            status = f"Fold {fold} | Ep {epoch:02d} | Loss: {running_loss/len(train_loader):.4f} | IoU: {iou:.4f} | F1: {f1:.4f}"
            if f1 > best_f1:
                best_f1 = f1
                best_ckpt = os.path.join(ckpt_dir, f"WPFormer_patch_7030_fold{fold}_best.pth")
                torch.save(net.state_dict(), best_ckpt)
                status += " [★ F1 最优]"
            train_logger.info(status)

    train_logger.info(f"🏆 Fold {fold} 训练结束，最佳 F1: {best_f1:.4f}")
    net.load_state_dict(torch.load(best_ckpt))
    eval_sliding_window(val_list_path, img_root, gt_root, net, fold, save_visuals=True)
    return best_f1

if __name__ == '__main__':
    all_f1s = []
    # 这一次我们从 Fold 1 开始完整地跑一遍 1 到 5 折
    for f in range(1, 6):
        fold_best_f1 = train_one_fold(f)
        all_f1s.append(fold_best_f1)
    
    final_logger = get_logger(LOG_FILE)
    final_logger.info("\n" + "="*50)
    final_logger.info(f"🎯 5-Fold (70/30 策略) 汇总结果:")
    final_logger.info(f"各折最佳 F1: {all_f1s}")
    final_logger.info(f"平均 F1: {np.mean(all_f1s):.4f}")
    final_logger.info("="*50)