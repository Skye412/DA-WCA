import os
import cv2
import numpy as np
from tqdm import tqdm

# ================= 路径配置 =================
S2DS_DIR = "/home/skye/data/Skye/databases/s2ds5"
IMG_DIR = os.path.join(S2DS_DIR, "images")
GT_DIR = os.path.join(S2DS_DIR, "labs")

PRED_100_DIR = "./save/stage3_patch/predictions"
PRED_7030_DIR = "./save/stage3_patch_7030/predictions"
COMP_OUT_DIR = "./save/comparison_4in1"

LOG_FILE = "/home/skye/data/Skye/DA-WCA/log.txt"

os.makedirs(COMP_OUT_DIR, exist_ok=True)

# ================= 指标计算函数 =================
def calculate_metrics(pred_img, gt_img):
    # 只计算 0 和 255 的有效区域，忽略 255 以外的中间过渡区域
    valid_mask = (gt_img == 0) | (gt_img == 255)
    pred_valid = pred_img[valid_mask]
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

# ================= 绘图与文字标注函数 =================
def add_label(image, text):
    # 在图片左上角添加黑色半透明底色和白色文字
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (320, 50), (0, 0, 0), -1)
    image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)
    cv2.putText(image, text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
    return image

# ================= 主程序 =================
def main():
    # 打开日志文件准备写入
    with open(LOG_FILE, 'w', encoding='utf-8') as log_f:
        log_f.write("========== 100% Patch vs 70/30 Patch 预测对比评估 ==========\n\n")
        
        overall_100_f1, overall_7030_f1 = [], []

        # 遍历 5 个折
        for fold in range(1, 6):
            fold_100_dir = os.path.join(PRED_100_DIR, f"fold{fold}")
            fold_7030_dir = os.path.join(PRED_7030_DIR, f"fold{fold}")
            fold_out_dir = os.path.join(COMP_OUT_DIR, f"fold{fold}")
            
            # 如果对应折的预测结果不存在，跳过
            if not os.path.exists(fold_100_dir) or not os.path.exists(fold_7030_dir):
                print(f"⚠️ 找不到 Fold {fold} 的预测结果，跳过...")
                continue
                
            os.makedirs(fold_out_dir, exist_ok=True)
            log_f.write(f"\n[{'-'*15} Fold {fold} {'-'*15}]\n")
            
            filenames = os.listdir(fold_100_dir)
            fold_metrics_100 = []
            fold_metrics_7030 = []

            print(f"正在处理 Fold {fold} 的对比图与指标...")
            for fname in tqdm(filenames, desc=f"Fold {fold}", leave=False):
                # 1. 读取 4 张图片
                img_path = os.path.join(IMG_DIR, fname)
                gt_path = os.path.join(GT_DIR, fname)
                p100_path = os.path.join(fold_100_dir, fname)
                p7030_path = os.path.join(fold_7030_dir, fname)
                
                if not os.path.exists(p7030_path):
                    continue

                orig = cv2.imread(img_path)
                gt_bw = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                p100_bw = cv2.imread(p100_path, cv2.IMREAD_GRAYSCALE)
                p7030_bw = cv2.imread(p7030_path, cv2.IMREAD_GRAYSCALE)

                # 2. 计算指标 [IoU, F1, Precision, Recall]
                m100 = calculate_metrics(p100_bw, gt_bw)
                m7030 = calculate_metrics(p7030_bw, gt_bw)
                
                fold_metrics_100.append(m100)
                fold_metrics_7030.append(m7030)

                # 3. 制作 4合1 对比图
                # 统一转为 BGR 彩色图以便拼接
                gt_color = cv2.cvtColor(gt_bw, cv2.COLOR_GRAY2BGR)
                p100_color = cv2.cvtColor(p100_bw, cv2.COLOR_GRAY2BGR)
                p7030_color = cv2.cvtColor(p7030_bw, cv2.COLOR_GRAY2BGR)

                # 添加带底色的文字标签
                orig = add_label(orig, "Original")
                gt_color = add_label(gt_color, "Ground Truth")
                p100_color = add_label(p100_color, "100% Patch")
                p7030_color = add_label(p7030_color, "70/30 Patch")

                # 水平拼接并保存
                comp_img = np.hstack((orig, gt_color, p100_color, p7030_color))
                cv2.imwrite(os.path.join(fold_out_dir, fname), comp_img)

            # 4. 计算并写入单折平均指标
            avg_100 = np.mean(fold_metrics_100, axis=0)
            avg_7030 = np.mean(fold_metrics_7030, axis=0)
            
            overall_100_f1.append(avg_100[1])
            overall_7030_f1.append(avg_7030[1])

            log_f.write(f"100% Patch -> IoU: {avg_100[0]:.4f} | F1: {avg_100[1]:.4f} | Pre: {avg_100[2]:.4f} | Rec: {avg_100[3]:.4f}\n")
            log_f.write(f"70/30 Patch -> IoU: {avg_7030[0]:.4f} | F1: {avg_7030[1]:.4f} | Pre: {avg_7030[2]:.4f} | Rec: {avg_7030[3]:.4f}\n")

        # 5. 写入汇总指标
        if overall_100_f1:
            log_f.write(f"\n[{'='*15} 5-Fold 平均结果 {'='*15}]\n")
            log_f.write(f"100% Patch 5-Fold 平均 F1: {np.mean(overall_100_f1):.4f}\n")
            log_f.write(f"70/30 Patch 5-Fold 平均 F1: {np.mean(overall_7030_f1):.4f}\n")

    print(f"\n✅ 所有对比图已生成至: {COMP_OUT_DIR}")
    print(f"✅ 指标计算完成，日志已保存至: {LOG_FILE}")

if __name__ == '__main__':
    main()