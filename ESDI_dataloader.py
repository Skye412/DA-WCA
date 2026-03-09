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
    def __init__(self, list_path, image_root, gt_root, trainsize=384, is_train=False):
        self.training = is_train
        self.trainsize = trainsize

        with open(list_path, 'r') as f:
            filenames = [line.strip() for line in f.readlines() if line.strip()]

        self.images = [os.path.join(image_root, f) for f in filenames]
        self.gts = [os.path.join(gt_root, f) for f in filenames]

        self.aug_transform = albu.Compose([
            albu.HorizontalFlip(p=0.5),
            albu.Rotate(limit=15, p=0.5),
            albu.RandomRotate90(p=0.5),
        ])

        self.img_transform = self.get_transform()
        self.trainsize_tuple = (self.trainsize, self.trainsize)

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])

        # 1. 数据增强
        image, gt = self.aug_transform(image=np.asarray(image), mask=np.asarray(gt)).values()

        # 2. 标签映射修正 (根据 check_pixels.py 的结果)
        gt_np = np.asarray(gt, dtype=np.float32)
        target = np.zeros_like(gt_np)
        
        # 【客观修正】：裂缝像素是 255，将其映射为 1.0
        target[gt_np == 255] = 1.0    
        # 其他病害（中间值如 149, 225）映射为 255.0，作为忽略区域
        target[(gt_np > 0) & (gt_np < 255)] = 255.0   
        
        # 3. Resize 并转为 Tensor
        target = cv2.resize(target, self.trainsize_tuple, interpolation=cv2.INTER_NEAREST)
        gt_tensor = torch.from_numpy(target).unsqueeze(0) 

        # 4. 边缘计算 (仅基于真正的裂缝)
        gt_uint8 = np.where(gt_np == 255, 255, 0).astype(np.uint8) 
        edge = cv2.Canny(gt_uint8, 100, 200)
        kernel = np.ones((5, 5), np.uint8)
        edge = cv2.dilate(edge, kernel, iterations=1)
        edge = cv2.resize(edge, self.trainsize_tuple, interpolation=cv2.INTER_NEAREST)
        edge_tensor = torch.from_numpy(edge).unsqueeze(0).float() / 255.0

        image = Image.fromarray(image)
        image = self.img_transform(image)

        return {'image': image, 'label': gt_tensor, "edge": edge_tensor}

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            return Image.open(f).convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            return Image.open(f).convert('L')

    def get_transform(self, mean=None, std=None):
        mean = [0.485, 0.456, 0.406] if mean is None else mean
        std = [0.229, 0.224, 0.225] if std is None else std
        return transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    def __len__(self):
        return len(self.images)

def get_loader(list_path, image_root, gt_root, batchsize, trainsize, is_train=False, shuffle=True, num_workers=0, pin_memory=True):
    dataset = ImageFolder(list_path, image_root, gt_root, trainsize, is_train)
    data_loader = data.DataLoader(dataset=dataset, batch_size=batchsize, shuffle=shuffle, 
                                  num_workers=num_workers, pin_memory=pin_memory, drop_last=is_train)
    return data_loader