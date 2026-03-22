"""
=============================================================================
阶段二：官方协议适配版评测器 (Adapted OmniCrack30k Evaluator) - 终极版
=============================================================================
说明: 
1. 提取骨架时严格屏蔽了 0 < val < 255 的模糊/容差区域。
2. 1:1 还原官方 apply_tolerance_official 与 jaccard_score。
=============================================================================
"""

import os
import cv2
import numpy as np
from tqdm import tqdm
import csv
from skimage.morphology import skeletonize, disk
from sklearn.metrics import jaccard_score

# ================= 1. 配置区 =================
S2DS_DIR = "/home/skye/data/Skye/databases/s2ds5"
PRED_BASE_DIR = "/home/skye/data/Skye/DA-WCA/unified_predictions_final"
FOLD = 1
TOLERANCE = 4

# 必须与上面的生成器名字严格一致！
MODELS_TO_EVAL = [
    "Stage3_512_NoPatch",
    "Patch_7030_Baseline",
    "ASER_7030_Best",
    "ASER_Centerline_Fold1"
]

# ================= 2. 核心逻辑 =================
def apply_tolerance_official(true_line, pred_line, tol=4):
    true_dil = cv2.dilate(true_line, disk(tol), iterations=1)
    pred_dil = cv2.dilate(pred_line, disk(tol), iterations=1)

    tp = true_line * pred_dil
    fp = pred_line - (pred_line * true_dil)
    fn = true_line - tp

    true_adjusted = tp + fn
    pred_adjusted = tp + fp
    return true_adjusted, pred_adjusted

def get_tp_fp_fn(pred_b, gt_b):
    tp = (pred_b * gt_b).sum()
    fp = (pred_b * (1 - gt_b)).sum()
    fn = ((1 - pred_b) * gt_b).sum()
    return tp, fp, fn

def calc_metrics_from_counts(tp, fp, fn):
    iou = tp / (tp + fp + fn + 1e-8)
    f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    return iou, f1, prec, rec

# ================= 3. 评测主流程 =================
def run_evaluation():
    print(f"\n{'='*85}")
    print(f"📊 启动官方协议适配版评测 (Fold {FOLD}, Tolerance={TOLERANCE}px)")
    print(f"{'='*85}\n")

    val_list_path = os.path.join(S2DS_DIR, f"fold{FOLD}_val.txt")
    gt_root = os.path.join(S2DS_DIR, "labs")

    with open(val_list_path, 'r') as f:
        filenames = [line.strip() for line in f.readlines() if line.strip()]

    csv_file = "unified_leaderboard_adapted_cliou_final.csv"
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Model_Name", "Macro_F1", "Global_F1", "Precision", "Recall", "Adapted_clIoU@4px"])

        for model_name in MODELS_TO_EVAL:
            pred_dir = os.path.join(PRED_BASE_DIR, model_name)
            if not os.path.exists(pred_dir):
                print(f"⚠️ 找不到预测图目录，跳过: {pred_dir}")
                continue
                
            print(f"\n✨ 正在评测: [{model_name}]...")
            macro_results = []
            global_tp, global_fp, global_fn = 0, 0, 0
            trues_cl_all = np.empty((0,), dtype=bool)
            preds_cl_all = np.empty((0,), dtype=bool)

            for filename in tqdm(filenames, desc="计算中", leave=False):
                gt_path = os.path.join(gt_root, filename)
                pred_path = os.path.join(pred_dir, filename)
                if not os.path.exists(pred_path): continue

                gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                pred_img = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
                
                # 严谨处理：屏蔽 0 < val < 255 的中间灰度/容差区域
                valid_mask = ((gt_img == 0) | (gt_img == 255))
                gt_b = ((gt_img == 255) & valid_mask).astype(np.uint8)
                pred_b = ((pred_img >= 128) & valid_mask).astype(np.uint8)
                
                # A. 传统指标
                tp, fp, fn = get_tp_fp_fn(pred_b, gt_b)
                macro_results.append(calc_metrics_from_counts(tp, fp, fn))
                global_tp += tp
                global_fp += fp
                global_fn += fn
                
                # B. 官方容差指标 (Adapted clIoU)
                gt_skel = skeletonize(gt_b).astype(np.uint8)
                pred_skel = skeletonize(pred_b).astype(np.uint8)
                
                true_adj, pred_adj = apply_tolerance_official(gt_skel, pred_skel, tol=TOLERANCE)
                true_adj, pred_adj = true_adj.flatten(), pred_adj.flatten()
                
                keep_idxs = np.where((true_adj == 1) | (pred_adj == 1))[0]
                true_adj, pred_adj = true_adj[keep_idxs], pred_adj[keep_idxs]
                
                trues_cl_all = np.append(trues_cl_all, true_adj)
                preds_cl_all = np.append(preds_cl_all, pred_adj)

            # 汇总结果
            avg_macro = np.mean(macro_results, axis=0)
            macro_iou, macro_f1, macro_prec, macro_rec = avg_macro
            global_iou, global_f1, global_prec, global_rec = calc_metrics_from_counts(global_tp, global_fp, global_fn)
            
            cl_iou = jaccard_score(trues_cl_all, preds_cl_all) if len(trues_cl_all) > 0 else 0.0
            
            print(f"✅ {model_name} 结果:")
            print(f"   [传统指标] Macro F1: {macro_f1:.4f} | Global F1: {global_f1:.4f} | Pre: {global_prec:.4f} | Rec: {global_rec:.4f}")
            print(f"   [结构指标] Adapted clIoU@4px: {cl_iou:.4f}")
            
            writer.writerow([model_name, f"{macro_f1:.4f}", f"{global_f1:.4f}", f"{global_prec:.4f}", f"{global_rec:.4f}", f"{cl_iou:.4f}"])

    print(f"\n🎉 评测结束！最终总表已保存至: {csv_file}")

if __name__ == '__main__':
    run_evaluation()