# -*- coding: utf-8 -*-
# @Time    : 2020/7/22
# @Author  : Lart Pang
# @Email   : lartpang@163.com
# @File    : dataloader.py
# @Project : code
# @GitHub  : https://github.com/lartpang
import os
import random
from functools import partial

import torch
from PIL import Image

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from joint_transforms import (
    Compose,
    JointResize,
    RandomHorizontallyFlip,
    RandomRotate,
)
from misc import construct_print
import albumentations
from PIL import ImageEnhance
import numpy as np
def colorEnhance(image):
    bright_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity = random.randint(0, 20) / 10.0
    image = ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity = random.randint(0, 30) / 10.0
    image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image


def randomGaussian(image, mean=0.1, sigma=0.35):
    def gaussianNoisy(im, mean=mean, sigma=sigma):
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im

    img = np.asarray(image)
    width, height = img.shape
    img = gaussianNoisy(img[:].flatten(), mean, sigma)
    img = img.reshape([width, height])
    return Image.fromarray(np.uint8(img))


def randomPeper(img):
    img = np.array(img)
    noiseNum = int(0.0015 * img.shape[0] * img.shape[1])
    for i in range(noiseNum):

        randX = random.randint(0, img.shape[0] - 1)

        randY = random.randint(0, img.shape[1] - 1)

        if random.randint(0, 1) == 0:

            img[randX, randY] = 0

        else:

            img[randX, randY] = 255
    return Image.fromarray(img)

import cv2
import albumentations as albu
class ImageFolder(Dataset):
    def __init__(self, image_root, gt_root, trainsize=384, is_train=False):
        self.training = is_train
        self.trainsize = trainsize

        self.images = [image_root + f for f in os.listdir(image_root)]
        self.gts = [gt_root + p for p in os.listdir(gt_root)]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)

        self.joint_transform = Compose(
            [JointResize(trainsize), RandomHorizontallyFlip(), RandomRotate(10)]
        )
        self.aug_transform = albu.Compose([
            #  albu.RandomScale(scale_limit=0.25, p=0.5),
            albu.HorizontalFlip(p=0.5),
            # albu.VerticalFlip(p=0.5),
            albu.Rotate(limit=15, p=0.5),
            albu.RandomRotate90(p=0.5),
        ])

        self.img_transform = self.get_transform()
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        self.edge_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])


    # def __getitem__(self, index):
    #     image = self.rgb_loader(self.images[index])
    #     gt = self.binary_loader(self.gts[index])
    #     image_size = image.size
    #     # data augumentation
    #     #image, gt = self.joint_transform(image,gt)

    #     image, gt = self.aug_transform(image=np.asarray(image), mask=np.asarray(gt)).values()

    #     gt = np.asarray(gt)
    #     edge = cv2.Canny(gt, 100, 200)
    #     kernel = np.ones((5, 5), np.uint8)
    #     edge = cv2.dilate(edge, kernel, iterations=1)
    #     # opencv-->PIL
    #     # image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    #     image = Image.fromarray(image)
    #     gt = Image.fromarray(gt)
    #     edge = Image.fromarray(edge)

    #     image = self.img_transform(image)
    #     gt = self.gt_transform(gt)
    #     edge = self.edge_transform(edge)

    #     return {'image': image, 'label': gt,"edge":edge}
    
    # def __getitem__(self, index):
    #     image = self.rgb_loader(self.images[index])
    #     gt = self.binary_loader(self.gts[index])

    #     # 1. 数据增强
    #     image, gt = self.aug_transform(image=np.asarray(image), mask=np.asarray(gt)).values()

    #     # 2. S2DS 标签硬核映射
    #     gt_np = np.asarray(gt, dtype=np.float32)
    #     target = np.zeros_like(gt_np)
        
    #     # 【客观条件确认】：这里假设 S2DS 中“裂缝”的像素值为 1，其他病害为 2, 3, 4...
    #     # 若你的 S2DS 掩码像素定义不同（例如裂缝本身是 255），请务必修改这里的逻辑。
    #     target[gt_np == 1] = 1.0    # 正样本 (裂缝)：模型需要学习的目标
    #     target[gt_np > 1] = 255.0   # 忽略样本 (其他病害)：不产生梯度
        
    #     # 3. 手动 Resize 并转为 Tensor (避开 ToTensor() 自动除以 255 的机制)
    #     # 使用 INTER_NEAREST 防止插值产生 0~1 之间的小数或破坏 255 的整型属性
    #     target = cv2.resize(target, (self.trainsize, self.trainsize), interpolation=cv2.INTER_NEAREST)
    #     gt_tensor = torch.from_numpy(target).unsqueeze(0) # [1, H, W]

    #     # 4. 边缘计算 (训练中暂未直接作为 Loss，但保持数据结构不出错)
    #     edge = cv2.Canny(np.uint8(gt_np), 100, 200)
    #     kernel = np.ones((5, 5), np.uint8)
    #     edge = cv2.dilate(edge, kernel, iterations=1)
    #     edge = cv2.resize(edge, (self.trainsize, self.trainsize), interpolation=cv2.INTER_NEAREST)
    #     edge_tensor = torch.from_numpy(edge).unsqueeze(0).float() / 255.0

    #     # 5. 原图处理保持不变
    #     image = Image.fromarray(image)
    #     image = self.img_transform(image)

    #     return {'image': image, 'label': gt_tensor, "edge": edge_tensor}
    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])

        # 1. 数据增强
        image, gt = self.aug_transform(image=np.asarray(image), mask=np.asarray(gt)).values()

        # 2. S2DS 标签硬核映射
        gt_np = np.asarray(gt, dtype=np.float32)
        target = np.zeros_like(gt_np)
        
        # 【客观条件确认】：这里假设 S2DS 中“裂缝”的像素值为 1，其他病害为 2, 3, 4...
        # 若你的 S2DS 掩码像素定义不同（例如裂缝本身是 255），请务必修改这里的逻辑。
        target[gt_np == 1] = 1.0    # 正样本 (裂缝)：模型需要学习的目标
        target[gt_np > 1] = 255.0   # 忽略样本 (其他病害)：不产生梯度
        
        # 3. 手动 Resize 并转为 Tensor (避开 ToTensor() 自动除以 255 的机制)
        # 使用 INTER_NEAREST 防止插值产生 0~1 之间的小数或破坏 255 的整型属性
        target = cv2.resize(target, (self.trainsize, self.trainsize), interpolation=cv2.INTER_NEAREST)
        gt_tensor = torch.from_numpy(target).unsqueeze(0) # [1, H, W]

        # 4. 边缘计算 (训练中暂未直接作为 Loss，但保持数据结构不出错)
        edge = cv2.Canny(np.uint8(gt_np), 100, 200)
        kernel = np.ones((5, 5), np.uint8)
        edge = cv2.dilate(edge, kernel, iterations=1)
        edge = cv2.resize(edge, (self.trainsize, self.trainsize), interpolation=cv2.INTER_NEAREST)
        edge_tensor = torch.from_numpy(edge).unsqueeze(0).float() / 255.0

        # 5. 原图处理保持不变
        image = Image.fromarray(image)
        image = self.img_transform(image)

        return {'image': image, 'label': gt_tensor, "edge": edge_tensor}


    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def get_transform(self, mean=None, std=None):
        mean = [0.485, 0.456, 0.406] if mean is None else mean
        std = [0.229, 0.224, 0.225] if std is None else std
        transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])
        return transform

    def __len__(self):
        return len(self.images)


import torch.utils.data as data
def get_loader(image_root, gt_root, batchsize, trainsize, is_train=False, shuffle=True, num_workers=0, pin_memory=True):
    dataset = ImageFolder(image_root, gt_root, trainsize, is_train)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory, drop_last=True)



    return data_loader

