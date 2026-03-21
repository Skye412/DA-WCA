"""
=============================================================================
终极 5-Fold 统一评测流水线 (5-Fold Unified Evaluation)
目标: 跑出能直接放进论文 Table 1 的最终平均数据。
修正: 引入严谨的滑窗边界保护；修正多折动态路径映射。
=============================================================================
"""

import os
import cv2
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
from skimage.morphology import skeletonize, disk
from sklearn.metrics import jaccard_score
import torch.nn.functional as F
from model.WPFormer import WPFormer

# ================= 1. 配置区 =================
S2DS_DIR = "/home/skye/data/Skye/databases/s2ds5"
TOLERANCE = 4
THRESHOLD = 0.50

# 动态折数权重路径映射
MODELS = [
    {
        "name": "Stage3_512_NoPatch", 
        "path_template": "/home/skye/data/Skye/DA-WCA/save/stage3_512/checkpoints/WPFormer_stage3_512_fold{fold}_best.pth", 
        "mode": "resize", "size": 512
    },
    {
        "name": "Patch_7030_Baseline", 
        # 注意：如果这个路径不对，请修改为真实的 Patch_7030 路径！
        "path_template": "/home/skye/data/Skye/DA-WCA/save/stage3_patch_7030/checkpoints/WPFormer_patch_7030_fold{fold}_best.pth", 
        "mode": "sliding", "window": 512, "stride": 256
    },
    {
        "name": "ASER_7030_Best", 
        "path_template": "/home/skye/data/Skye/DA-WCA/save/stage3_patch_aser_7030_5fold/checkpoints/WPFormer_aser_fold{fold}_best.pth", 
        "mode": "sliding", "window": 512, "stride": 256
    }
]

# ================= 2. 核心函数 =================
def extract_prediction(preds):
    if isinstance(preds, tuple): raise ValueError("须 return_aux=False")
    return preds[-1] if isinstance(preds, list) else preds

def infer_whole_image(model, img_pil, target_size):
    orig_w, orig_h = img_pil.size
    img_tensor = transforms.ToTensor()(img_pil.resize((target_size, target_size), Image.BILINEAR))
    img_tensor = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img_tensor).unsqueeze(0).cuda()
    with torch.no_grad(): pred_mask = extract_prediction(model(img_tensor))
    pred_mask = F.interpolate(pred_mask, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
    return torch.sigmoid(pred_mask).cpu().numpy().squeeze()

def infer_sliding_window(model, img_pil, window_size, stride):
    """【修复版】严格且安全的滑窗逻辑，确保右侧与底侧边缘完美覆盖"""
    img_tensor = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(transforms.ToTensor()(img_pil)).unsqueeze(0).cuda()
    b, c, h, w = img_tensor.size()
    full_mask = torch.zeros((b, 1, h, w), device='cuda')
    count_map = torch.zeros((b, 1, h, w), device='cuda')

    pad_h = max(0, window_size - h)
    pad_w = max(0, window_size - w)
    if pad_h > 0 or pad_w > 0:
        img_tensor = F.pad(img_tensor, (0, pad_w, 0, pad_h), mode='reflect')
        _, _, h_pad, w_pad = img_tensor.size()
    else:
        h_pad, w_pad = h, w

    y_steps = list(range(0, h_pad - window_size + 1, stride))
    if h_pad - window_size not in y_steps:
        y_steps.append(h_pad - window_size)

    x_steps = list(range(0, w_pad - window_size + 1, stride))
    if w_pad - window_size not in x_steps:
        x_steps.append(w_pad - window_size)

    for y in y_steps:
        for x in x_steps:
            with torch.no_grad():
                pred_patch = torch.sigmoid(extract_prediction(model(img_tensor[:, :, y:y+window_size, x:x+window_size])))
            full_mask[:, :, y:y+window_size, x:x+window_size] += pred_patch
            count_map[:, :, y:y+window_size, x:x+window_size] += 1.0

    return (full_mask[:, :, :h, :w] / (count_map[:, :, :h, :w] + 1e-8)).cpu().numpy().squeeze()

def apply_tolerance_official(true_line, pred_line, tol=4):
    true_dil = cv2.dilate(true_line, disk(tol), iterations=1)
    pred_dil = cv2.dilate(pred_line, disk(tol), iterations=1)
    tp = true_line * pred_dil
    fp = pred_line - (pred_line * true_dil)
    fn = true_line - tp
    return tp + fn, tp + fp

# ================= 3. 主循环 =================
def run_5fold_pipeline():
    final_results = {m["name"]: {"macro_f1": [], "global_f1": [], "prec": [], "rec": [], "cliou": []} for m in MODELS}

    for fold in range(1, 6):
        print(f"\n{'='*40} 开始处理 Fold {fold} {'='*40}")
        val_list = [l.strip() for l in open(os.path.join(S2DS_DIR, f"fold{fold}_val.txt"), 'r') if l.strip()]
        
        for cfg in MODELS:
            ckpt_path = cfg["path_template"].format(fold=fold)
            if not os.path.exists(ckpt_path):
                print(f"⚠️ 找不到 Fold {fold} 的权重: {ckpt_path}，跳过此折该模型。")
                continue
                
            print(f"[{cfg['name']}] 推理 & 评测...")
            model = WPFormer(method="pvt_v2_b2", channel=64).cuda()
            model.load_state_dict(torch.load(ckpt_path, map_location='cuda'), strict=False)
            model.eval()

            macro_f1s, global_tp, global_fp, global_fn = [], 0, 0, 0
            trues_cl_all, preds_cl_all = np.empty((0,), dtype=bool), np.empty((0,), dtype=bool)

            for filename in tqdm(val_list, leave=False):
                img_path = os.path.join(S2DS_DIR, "images", filename)
                gt_path = os.path.join(S2DS_DIR, "labs", filename)
                if not os.path.exists(gt_path): continue

                # 推理
                img_pil = Image.open(img_path).convert("RGB")
                if cfg["mode"] == "resize": pred_prob = infer_whole_image(model, img_pil, cfg["size"])
                else: pred_prob = infer_sliding_window(model, img_pil, cfg["window"], cfg["stride"])
                pred_img = (pred_prob >= THRESHOLD).astype(np.uint8) * 255

                # 评测准备 (屏蔽 Ignore 区域)
                gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                valid_mask = ((gt_img == 0) | (gt_img == 255))
                gt_b = ((gt_img == 255) & valid_mask).astype(np.uint8)
                pred_b = ((pred_img >= 128) & valid_mask).astype(np.uint8)

                # 传统指标
                tp, fp, fn = (pred_b*gt_b).sum(), (pred_b*(1-gt_b)).sum(), ((1-pred_b)*gt_b).sum()
                macro_f1s.append(2*tp / (2*tp + fp + fn + 1e-8))
                global_tp += tp; global_fp += fp; global_fn += fn

                # 容差指标
                gt_skel = skeletonize(gt_b).astype(np.uint8)
                pred_skel = skeletonize(pred_b).astype(np.uint8)
                t_adj, p_adj = apply_tolerance_official(gt_skel, pred_skel, tol=TOLERANCE)
                t_adj, p_adj = t_adj.flatten(), p_adj.flatten()
                keep = np.where((t_adj == 1) | (p_adj == 1))[0]
                trues_cl_all = np.append(trues_cl_all, t_adj[keep])
                preds_cl_all = np.append(preds_cl_all, p_adj[keep])

            # 记录本折结果
            m_f1 = np.mean(macro_f1s)
            g_f1 = 2*global_tp / (2*global_tp + global_fp + global_fn + 1e-8)
            g_pr = global_tp / (global_tp + global_fp + 1e-8)
            g_rc = global_tp / (global_tp + global_fn + 1e-8)
            cl_iou = jaccard_score(trues_cl_all, preds_cl_all) if len(trues_cl_all)>0 else 0.0

            final_results[cfg["name"]]["macro_f1"].append(m_f1)
            final_results[cfg["name"]]["global_f1"].append(g_f1)
            final_results[cfg["name"]]["prec"].append(g_pr)
            final_results[cfg["name"]]["rec"].append(g_rc)
            final_results[cfg["name"]]["cliou"].append(cl_iou)

    # 打印最终 5折平均表
    print("\n\n" + "="*85)
    print(f"🏆 终极 5-Fold 平均结果 (Threshold={THRESHOLD}, Tolerance={TOLERANCE}px)")
    print("="*85)
    print(f"{'Model_Name':<20} | {'Macro_F1':<10} | {'Global_F1':<10} | {'Precision':<10} | {'Recall':<10} | {'clIoU@4px':<10}")
    print("-" * 85)
    for model_name, res in final_results.items():
        if len(res["macro_f1"]) == 0: continue
        print(f"{model_name:<20} | {np.mean(res['macro_f1']):<10.4f} | {np.mean(res['global_f1']):<10.4f} | {np.mean(res['prec']):<10.4f} | {np.mean(res['rec']):<10.4f} | {np.mean(res['cliou']):<10.4f}")
    print("="*85)

if __name__ == '__main__':
    run_5fold_pipeline()