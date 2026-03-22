# -*- coding: utf-8 -*-
import os
import cv2
import torch
import numpy as np
import random
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

        self.base_aug = albu.Compose([
            albu.HorizontalFlip(p=0.5),
            albu.RandomRotate90(p=0.5),
            albu.RandomBrightnessContrast(p=0.2),
        ])
        
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        image = np.asarray(Image.open(self.images[index]).convert('RGB'))
        gt = np.asarray(Image.open(self.gts[index]).convert('L'))

        # 1. 裁剪与数据增强
        if self.training:
            if random.random() < 0.7:
                cropper = albu.CropNonEmptyMaskIfExists(height=self.trainsize, width=self.trainsize, p=1.0)
            else:
                cropper = albu.RandomCrop(height=self.trainsize, width=self.trainsize, p=1.0)
                
            aug = cropper(image=image, mask=gt)
            image_patch, gt_patch = aug['image'], aug['mask']
            
            final_aug = self.base_aug(image=image_patch, mask=gt_patch)
            image, gt_aug = final_aug['image'], final_aug['mask']
        else:
            aug = albu.CenterCrop(height=self.trainsize, width=self.trainsize, p=1.0)(image=image, mask=gt)
            image, gt_aug = aug['image'], aug['mask']

        # 2. 标签映射与容错环机制
        gt_np = np.asarray(gt_aug, dtype=np.float32)
        gt_bin = np.where(gt_np == 255, 255, 0).astype(np.uint8)

        if self.training:
            kernel = np.ones((3, 3), np.uint8)
            gt_closed = cv2.morphologyEx(gt_bin, cv2.MORPH_CLOSE, kernel, iterations=1)
            gt_dilated = cv2.dilate(gt_closed, kernel, iterations=1)

            target = np.zeros_like(gt_bin, dtype=np.float32)
            target[gt_closed == 255] = 1.0
            target[(gt_dilated == 255) & (gt_closed == 0)] = 255.0 
            edge_source = gt_closed
        else:
            target = np.zeros_like(gt_bin, dtype=np.float32)
            target[gt_bin == 255] = 1.0
            target[(gt_np > 0) & (gt_np < 255)] = 255.0 
            edge_source = gt_bin

        # 3. 边缘特征提取
        edge = cv2.Canny(edge_source, 100, 200)
        edge = cv2.dilate(edge, np.ones((5, 5), np.uint8), iterations=1)
        edge_tensor = torch.from_numpy(edge).unsqueeze(0).float() / 255.0

        # 返回纯净的字典 (只有 image, label, edge)
        return {
            'image': self.img_transform(Image.fromarray(image)), 
            'label': torch.from_numpy(target).unsqueeze(0), 
            "edge": edge_tensor
        }

    def __len__(self):
        return len(self.images)

def get_loader(list_path, image_root, gt_root, batchsize, trainsize, is_train=False, shuffle=True, num_workers=4, pin_memory=True):
    dataset = ImageFolder(list_path, image_root, gt_root, trainsize, is_train)
    return data.DataLoader(dataset=dataset, batch_size=batchsize, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, drop_last=is_train)