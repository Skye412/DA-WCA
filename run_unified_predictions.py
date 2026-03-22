"""
=============================================================================
阶段一：双规制预测图生成器 (Dual-Protocol Prediction Generator) - 终极版
=============================================================================
目标：冻结阈值(0.50)，严谨滑窗，为核心演进模型生成标准的二值化预测图。
=============================================================================
"""

import os
import cv2
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import torch.nn.functional as F

from model.WPFormer import WPFormer

# ================= 1. 历史功勋簿 (Model Manifest) =================
MODELS_TO_RUN = [
    {"name": "Stage3_512_NoPatch", 
     "path": "/home/skye/data/Skye/DA-WCA/save/stage3_512/checkpoints/WPFormer_stage3_512_fold1_best.pth", 
     "mode": "resize", "size": 512},
     
    {"name": "Patch_7030_Baseline", 
     "path": "/home/skye/data/Skye/DA-WCA/save/stage3_patch_7030/checkpoints/WPFormer_patch_7030_fold1_best.pth", 
     "mode": "sliding", "window": 512, "stride": 256},
     
    {"name": "ASER_7030_Best",      
     "path": "/home/skye/data/Skye/DA-WCA/save/stage3_patch_aser_7030_5fold/checkpoints/WPFormer_aser_fold1_best.pth", 
     "mode": "sliding", "window": 512, "stride": 256},
     
    # 🌟 新晋究极完全体
    {"name": "ASER_Centerline_Fold1", 
     "path": "/home/skye/data/Skye/DA-WCA/save/stage3_patch_aser_centerline_fold1/checkpoints/WPFormer_centerline_fold1_best.pth", 
     "mode": "sliding", "window": 512, "stride": 256},

     {"name": "ASER_Centerline_Fold1_0.02loss_centerline", 
     "path": "/home/skye/data/Skye/DA-WCA/save/stage3_patch_aser_centerline_fold1_0.02loss_centerline/checkpoints/WPFormer_centerline_fold1_best.pth", 
     "mode": "sliding", "window": 512, "stride": 256}
]

S2DS_DIR = "/home/skye/data/Skye/databases/s2ds5"
FOLD = 1
OUTPUT_BASE_DIR = "/home/skye/data/Skye/DA-WCA/unified_predictions_final"
THRESHOLD = 0.50

# ================= 2. 模型输出适配器 =================
def extract_prediction(preds):
    if isinstance(preds, tuple): raise ValueError("评估时必须 return_aux=False！")
    if isinstance(preds, list): return preds[-1]
    return preds

# ================= 3. 两套推理引擎 =================
def infer_whole_image(model, img_pil, target_size):
    orig_w, orig_h = img_pil.size
    img_resized = img_pil.resize((target_size, target_size), Image.BILINEAR)
    img_tensor = transforms.ToTensor()(img_resized)
    img_tensor = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img_tensor).unsqueeze(0).cuda()
    
    with torch.no_grad():
        preds = model(img_tensor)
        pred_mask = extract_prediction(preds)
    
    pred_mask = F.interpolate(pred_mask, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
    return torch.sigmoid(pred_mask).cpu().numpy().squeeze()

def infer_sliding_window(model, img_pil, window_size, stride):
    img_tensor = transforms.ToTensor()(img_pil)
    img_tensor = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img_tensor).unsqueeze(0).cuda()
    
    b, c, h, w = img_tensor.size()
    full_mask = torch.zeros((b, 1, h, w), device=img_tensor.device)
    count_map = torch.zeros((b, 1, h, w), device=img_tensor.device)

    pad_h = max(0, window_size - h)
    pad_w = max(0, window_size - w)
    if pad_h > 0 or pad_w > 0:
        img_tensor = F.pad(img_tensor, (0, pad_w, 0, pad_h), mode='reflect')
        _, _, h_pad, w_pad = img_tensor.size()
    else:
        h_pad, w_pad = h, w

    # 极其严谨的滑窗边界保护逻辑
    y_steps = list(range(0, h_pad - window_size + 1, stride))
    if h_pad - window_size not in y_steps: y_steps.append(h_pad - window_size)
    x_steps = list(range(0, w_pad - window_size + 1, stride))
    if w_pad - window_size not in x_steps: x_steps.append(w_pad - window_size)

    for y in y_steps:
        for x in x_steps:
            patch = img_tensor[:, :, y:y+window_size, x:x+window_size]
            with torch.no_grad():
                preds = model(patch)
                pred_patch = torch.sigmoid(extract_prediction(preds))
            full_mask[:, :, y:y+window_size, x:x+window_size] += pred_patch
            count_map[:, :, y:y+window_size, x:x+window_size] += 1.0

    return (full_mask[:, :, :h, :w] / (count_map[:, :, :h, :w] + 1e-8)).cpu().numpy().squeeze()

# ================= 4. 主流程 =================
def run_all_predictions():
    print(f"\n{'='*60}")
    print(f"🚀 启动终极预测图生成器 (Threshold: {THRESHOLD:.2f})")
    print(f"{'='*60}\n")
    
    val_list_path = os.path.join(S2DS_DIR, f"fold{FOLD}_val.txt")
    image_root = os.path.join(S2DS_DIR, "images")

    with open(val_list_path, 'r') as f:
        filenames = [line.strip() for line in f.readlines() if line.strip()]

    for cfg in MODELS_TO_RUN:
        model_name = cfg["name"]
        ckpt_path = cfg["path"]
        
        if not os.path.exists(ckpt_path):
            print(f"⚠️ [跳过] 找不到权重文件: {ckpt_path}")
            continue
            
        print(f"\n⚡ 推理模型: [{model_name}]")
        
        pred_out_dir = os.path.join(OUTPUT_BASE_DIR, model_name)
        os.makedirs(pred_out_dir, exist_ok=True)
        
        model = WPFormer(method="pvt_v2_b2", channel=64).cuda()
        model.load_state_dict(torch.load(ckpt_path, map_location='cuda'), strict=False)
        model.eval()

        for filename in tqdm(filenames, desc="推理进度"):
            img_path = os.path.join(image_root, filename)
            img_pil = Image.open(img_path).convert("RGB")
            
            if cfg["mode"] == "resize":
                pred_prob = infer_whole_image(model, img_pil, target_size=cfg["size"])
            elif cfg["mode"] == "sliding":
                pred_prob = infer_sliding_window(model, img_pil, window_size=cfg["window"], stride=cfg["stride"])
            else:
                raise ValueError("未知的推理模式！")
            
            pred_bw = (pred_prob >= THRESHOLD).astype(np.uint8) * 255
            cv2.imwrite(os.path.join(pred_out_dir, filename), pred_bw)
            
    print(f"\n✅ 预测图生成完毕！存放于: {OUTPUT_BASE_DIR}")

if __name__ == '__main__':
    run_all_predictions()