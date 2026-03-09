import os
import cv2
import numpy as np
from tqdm import tqdm

def generate_bw_composites():
    base_dir = "./save/stage1_s2ds"
    pred_base = os.path.join(base_dir, "predictions")
    out_base = os.path.join(base_dir, "composites_bw")
    
    # 原始图像和标签的绝对路径
    img_dir = "/home/skye/data/Skye/databases/s2ds5/images"
    gt_dir = "/home/skye/data/Skye/databases/s2ds5/labs"

    os.makedirs(out_base, exist_ok=True)

    # 遍历 5 折的结果
    for fold in range(1, 6):
        pred_dir = os.path.join(pred_base, f"fold{fold}")
        if not os.path.exists(pred_dir):
            continue
            
        out_dir = os.path.join(out_base, f"fold{fold}")
        os.makedirs(out_dir, exist_ok=True)

        filenames = [f for f in os.listdir(pred_dir) if f.endswith('.png')]
        
        print(f"正在生成 Fold {fold} 的纯黑白三合一对比图...")
        for filename in tqdm(filenames):
            pred_path = os.path.join(pred_dir, filename)
            gt_path = os.path.join(gt_dir, filename)
            img_path = os.path.join(img_dir, filename)

            if not os.path.exists(gt_path) or not os.path.exists(img_path):
                continue

            # 1. 读取原始RGB图、真值灰度图、预测灰度图
            ori_img = cv2.imread(img_path)
            gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)

            H, W = gt.shape

            # 确保尺寸一致
            if pred.shape != gt.shape:
                pred = cv2.resize(pred, (W, H), interpolation=cv2.INTER_NEAREST)
            ori_img = cv2.resize(ori_img, (W, H))

            # 2. 严格二值化处理
            # 真值图：屏蔽忽略区(149/225等)，仅将真正的裂缝(255)设为白色，其余全黑
            gt_bw = np.where(gt == 255, 255, 0).astype(np.uint8)
            
            # 预测图：以128为阈值进行硬切分，转为纯黑白
            pred_bw = np.where(pred >= 128, 255, 0).astype(np.uint8)

            # 3. 将单通道黑白图转为三通道，以便与原图进行水平拼接
            gt_3c = cv2.cvtColor(gt_bw, cv2.COLOR_GRAY2BGR)
            pred_3c = cv2.cvtColor(pred_bw, cv2.COLOR_GRAY2BGR)

            # 4. 拼接：[原图 | 真值图(黑白) | 预测图(黑白)]
            composite = np.hstack((ori_img, gt_3c, pred_3c))

            # 5. 保存
            cv2.imwrite(os.path.join(out_dir, filename), composite)

if __name__ == '__main__':
    generate_bw_composites()