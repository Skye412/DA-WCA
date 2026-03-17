# -*- coding: utf-8 -*-
import os
import cv2
import torch
import numpy as np
from PIL import Image
from torch.utils import data
import torchvision.transforms as transforms
import albumentations as albu

class ImageFolder(data.Dataset):
    def __init__(self, list_path, image_root, gt_root, trainsize=512, is_train=False):
        self.training = is_train
        self.trainsize = trainsize

        with open(list_path, 'r') as f:
            filenames = [line.strip() for line in f.readlines() if line.strip()]

        self.images = [os.path.join(image_root, f) for f in filenames]
        self.gts = [os.path.join(gt_root, f) for f in filenames]

        if self.training:
            self.aug_transform = albu.Compose([
                # 【核心】：从原图 1:1 裁剪 512x512，且保证框内有裂缝 (p=1.0)
                albu.CropNonEmptyMaskIfExists(height=trainsize, width=trainsize, p=1.0),
                albu.HorizontalFlip(p=0.5),
                albu.RandomRotate90(p=0.5),
                albu.RandomBrightnessContrast(p=0.2),
            ])
        else:
            # 如果是 dataloader 里的验证，为了不出错使用中心裁剪
            self.aug_transform = albu.Compose([
                albu.CenterCrop(height=trainsize, width=trainsize, p=1.0),
            ])

        # 移除 Resize，只保留 ToTensor 和 Normalize
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        image = np.asarray(self.rgb_loader(self.images[index]))
        gt = np.asarray(self.binary_loader(self.gts[index]))

        # 1. 裁剪与增强 (保持物理分辨率)
        augmented = self.aug_transform(image=image, mask=gt)
        image = augmented['image']
        gt_aug = augmented['mask']

        # 2. 标签映射
        gt_np = np.asarray(gt_aug, dtype=np.float32)
        target = np.zeros_like(gt_np)
        target[gt_np == 255] = 1.0    
        target[(gt_np > 0) & (gt_np < 255)] = 255.0   
        gt_tensor = torch.from_numpy(target).unsqueeze(0) 

        # 3. 边缘计算
        gt_uint8 = np.where(gt_np == 255, 255, 0).astype(np.uint8) 
        edge = cv2.Canny(gt_uint8, 100, 200)
        kernel = np.ones((5, 5), np.uint8)
        edge = cv2.dilate(edge, kernel, iterations=1)
        edge_tensor = torch.from_numpy(edge).unsqueeze(0).float() / 255.0

        # 4. 图像转 Tensor
        image = self.img_transform(Image.fromarray(image))

        return {'image': image, 'label': gt_tensor, "edge": edge_tensor}

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            return Image.open(f).convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            return Image.open(f).convert('L')

    def __len__(self):
        return len(self.images)

def get_loader(list_path, image_root, gt_root, batchsize, trainsize, is_train=False, shuffle=True, num_workers=0, pin_memory=True):
    dataset = ImageFolder(list_path, image_root, gt_root, trainsize, is_train)
    data_loader = data.DataLoader(dataset=dataset, batch_size=batchsize, shuffle=shuffle, 
                                  num_workers=num_workers, pin_memory=pin_memory, drop_last=is_train)
    return data_loader