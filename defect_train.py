import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
from torchvision import transforms
import torch.optim as optim
import os
import logging
import sys
from sod_metrics import MAE, Emeasure, Fmeasure, Smeasure, WeightedFmeasure
from PIL import Image
from torch.optim.lr_scheduler import CosineAnnealingLR
from model.WPFormer import WPFormer
from ESDI_dataloader import get_loader
import cv2
import torch.nn.functional as F
import time
from tqdm import tqdm

def setup_logger(name, save_dir, filename="log.txt", mode='w'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        fh = logging.FileHandler(os.path.join(save_dir, filename), mode=mode) 
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

def eval_psnr(test_list_path, test_image_root, test_gt_root, train_size, model):
    FM = Fmeasure()
    WFM = WeightedFmeasure()
    SM = Smeasure()
    EM = Emeasure()
    M = MAE()

    img_transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    with open(test_list_path, 'r') as f:
        filenames = [line.strip() for line in f.readlines() if line.strip()]

    images = [os.path.join(test_image_root, f) for f in filenames]
    gts = [os.path.join(test_gt_root, f) for f in filenames]
    
    model.eval()
    for index in range(len(images)):
        ori_image = Image.open(images[index]).convert("RGB")
        image = img_transform(ori_image).unsqueeze(0).cuda()

        gt = cv2.imread(gts[index], cv2.IMREAD_GRAYSCALE)
        H, W = gt.shape
        with torch.no_grad():
            predictions_mask = model(image)
            res = predictions_mask[-1]

        res = torch.sigmoid(res).data.cpu().numpy().squeeze()
        pred = (res - res.min()) / (res.max() - res.min() + 1e-8)
        pred = Image.fromarray(pred * 255).convert("L")
        pred = pred.resize((W, H), resample=Image.BILINEAR)

        pred = np.array(pred)

        FM.step(pred=pred, gt=gt)
        WFM.step(pred=pred, gt=gt)
        SM.step(pred=pred, gt=gt)
        EM.step(pred=pred, gt=gt)
        M.step(pred=pred, gt=gt)
        
    fm = FM.get_results()["fm"]
    wfm = WFM.get_results()["wfm"]
    sm = SM.get_results()["sm"]
    em = EM.get_results()["em"]
    mae = M.get_results()["mae"]

    return mae, wfm, sm

def total_loss(pred, mask):
    pred = torch.sigmoid(pred)
    
    # 生成有效像素掩码：值为 255 的区域设为 0（不参与计算），其余设为 1
    valid_mask = (mask != 255).float()
    
    # 将标签中的 255 暂时替换为 0，防止底层运算越界报错
    clean_mask = torch.where(mask == 255, torch.zeros_like(mask), mask)
    
    # 计算带掩码的 BCE Loss
    bce = F.binary_cross_entropy(pred, clean_mask, reduction='none')
    bce = (bce * valid_mask).sum() / (valid_mask.sum() + 1e-8)

    # 计算带掩码的 IoU Loss
    inter = (pred * clean_mask * valid_mask).sum(dim=(2, 3))
    union = ((pred + clean_mask) * valid_mask).sum(dim=(2, 3))
    iou = 1 - inter / (union - inter + 1e-8)
    iou = iou.mean()

    return iou + bce

def train(fold, model_name, dataset_name):
    epoch_num = 60
    epoch_val = 10 

    net = WPFormer(method="pvt_v2_b2", channel=64)
    train_size = 384

    file_dir = "/home/skye/data/Skye/databases/s2ds5/"

    train_list_path = os.path.join(file_dir, f"fold{fold}_train.txt")
    val_list_path = os.path.join(file_dir, f"fold{fold}_val.txt")

    train_image_root = os.path.join(file_dir, "images", "")
    train_gt_root = os.path.join(file_dir, "labs", "")
    test_image_root = os.path.join(file_dir, "images", "")
    test_gt_root = os.path.join(file_dir, "labs", "")

    train_loader1 = get_loader(train_list_path, train_image_root, train_gt_root, batchsize=8, trainsize=train_size, is_train=True)

    if torch.cuda.is_available():
        net = net.cuda()

    optimizer = optim.Adam(net.parameters(), lr=8e-5)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=epoch_num, eta_min=1e-7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"---Fold {fold}: start training...")
    running_loss = 0.0
    best_sm = 0

    for epoch in range(0, epoch_num):
        print(f"\nEpoch: {epoch}/{epoch_num-1}")
        start_time = time.time()
        running_loss = 0.0

        # 【加入 tqdm 进度条】
        pbar = tqdm(train_loader1, desc=f"Fold {fold} Training")
        for i, data in enumerate(pbar):
            inputs, labels = data['image'], data['label']
            inputs = inputs.type(torch.FloatTensor)
            labels = labels.type(torch.FloatTensor)

            images, gts = Variable(inputs.to(device), requires_grad=False), Variable(labels.to(device), requires_grad=False)
            
            optimizer.zero_grad()
            predictions_mask = net(images)

            mask_losses = 0
            for j in range(len(predictions_mask)):
                mask_losses = mask_losses + total_loss(predictions_mask[j], gts)

            losses = mask_losses
            losses.backward()
            optimizer.step()

            running_loss += losses.item()
            
            # 【实时在进度条尾部显示当前批次的 Loss】
            pbar.set_postfix({'Loss': f"{losses.item():.4f}"})

        end_time = time.time()
        # 【打印当前 epoch 的平均 Loss】
        avg_loss = running_loss / len(train_loader1)
        print('Cost time: {:.4f}s | Avg Loss: {:.4f}'.format(end_time - start_time, avg_loss))

        lr_scheduler.step()

        if (epoch+1) >= epoch_val:
            mae, wfm, sm = eval_psnr(val_list_path, test_image_root, test_gt_root, train_size, net)

            if sm > best_sm:
                save_path = os.path.join("./save", dataset_name)
                os.makedirs(save_path, exist_ok=True)
                torch.save(net.state_dict(), os.path.join(save_path, f"{model_name}-{dataset_name}-fold{fold}-{sm:.4f}.pth"))
                best_sm = sm
                print(f"New best SM: {best_sm:.4f}")
            print("mae:%.4f, best_sm:%.4f, sm: %.4f" % (mae, best_sm, sm))
            net.train()

if __name__ == '__main__':
    for fold in range(1, 6):
        print(f"\n=============================================")
        print(f"========== 正在启动 Fold {fold} 的训练 ==========")
        print(f"=============================================")
        train(fold=fold, model_name="WPFormer", dataset_name="s2ds")