import os
import random
from pathlib import Path

import albumentations
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader
from collections import Counter

# 确保在文件开头定义TransformTwice类
class TransformTwice:
    def __init__(self, weak_transform, strong_transform):
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform

    def __call__(self, inp):
        weak_out = self.weak_transform(image=inp)["image"]
        strong_out = self.strong_transform(image=inp)["image"]
        return weak_out, strong_out

class Isic2019Raw(torch.utils.data.Dataset):
    def __init__(
        self,
        X_dtype=torch.float32,  # 输入数据（图像）类型
        y_dtype=torch.int64,  # 输出数据（标签）类型
        augmentations=None,  # 数据增强
        data_path='/dataset/ISIC_2019',  # 图片和csv根路径
        csvpath = '/home/zhangyuchao/tmpdir/Federated_Learning/FedCD/dataset/ISIC_2019/train_test_split.csv'  # csv文件路径
    ):
        self.dic = {
            "input_preprocessed": "/home/zhangyuchao/tmpdir/Federated_Learning/FedCD/dataset/ISIC_2019/ISIC_2019_Training_Input_preprocessed",  # 图像路径
            "train_test_split":  "/home/zhangyuchao/tmpdir/Federated_Learning/FedCD/dataset/ISIC_2019/train_test_split.csv",  # csv文件路径
        } 
        self.X_dtype = X_dtype  # 输入数据（图像）类型
        self.y_dtype = y_dtype  # 输出数据（标签）类型
        df2 = pd.read_csv(self.dic["train_test_split"])

        images = df2.image.tolist()  # df2.image 表示访问 DataFrame df2 中名为 image 的列，tolist将其SERIES转换列表 （就是里面全是图像名字）
        self.image_paths = [  # 生成图像的完整路径列表
            os.path.join(self.dic["input_preprocessed"], image_name + ".jpg")  # image_name + ".jpg"是一体，和前面的用系统自带属性分隔符分开
            for image_name in images
        ]
        self.targets = df2.target  # df2.target 表示访问 DataFrame df2 中名为 target 的列。
        self.augmentations = augmentations
        self.centers = df2.center
        self.n_classes = 8  # 类别数

    def __len__(self):
        return len(self.image_paths)  # 一共多少个图像，self.imaeg_path是一个列表

    def __getitem__(self, idx):  # 根据索引 idx 获取数据集中的一个样本（图像和标签）
        image_path = self.image_paths[idx]  # image_path是一个列表
        image = Image.open(image_path)  # PIL类型
        
        # 修改这里：将target转为int且添加断言检查
        target = int(self.targets[idx])
        assert 0 <= target <= 7, f"错误标签: {target} @ idx={idx} 路径={self.image_paths[idx]}"

        # 处理双视图情况
        if isinstance(self.augmentations, TransformTwice):
            image1 = np.array(image)
            image2 = np.array(image.copy())
            
            # 直接使用 __call__ 方法获取两个视图
            image1, image2 = self.augmentations(image1)
            
            # 添加NaN和全零检查
            if np.isnan(image1).any() or np.isnan(image2).any():
                raise ValueError(f"增强后有NaN @ idx={idx} 路径={self.image_paths[idx]}")
            if np.abs(image1).max() < 1e-5 or np.abs(image2).max() < 1e-5:
                print(f"警告：增强后接近全零！@ idx={idx} 路径={self.image_paths[idx]}")
            
            image1 = np.transpose(image1, (2, 0, 1)).astype(np.float32)
            image2 = np.transpose(image2, (2, 0, 1)).astype(np.float32)
            return (
                (torch.tensor(image1, dtype=self.X_dtype), 
                 torch.tensor(image2, dtype=self.X_dtype)),
                torch.tensor(target, dtype=self.y_dtype),
            )
        # 单视图情况
        else:
            image = np.array(image)
            if self.augmentations is not None:
                augmented = self.augmentations(image=image)
                image = augmented["image"]
            
            # 添加NaN和全零检查
            if np.isnan(image).any():
                raise ValueError(f"增强后有NaN @ idx={idx} 路径={self.image_paths[idx]}")
            if np.abs(image).max() < 1e-5:
                print(f"警告：增强后接近全零！@ idx={idx} 路径={self.image_paths[idx]}")
                
            image = np.transpose(image, (2, 0, 1)).astype(np.float32)
            return (
                torch.tensor(image, dtype=self.X_dtype),
                torch.tensor(target, dtype=self.y_dtype),
            )

class FedIsic2019(Isic2019Raw):  # 会继承 Isic2019Raw 的所有属性和方法
    def __init__(
        self,
        center: int=0,  # 默认client0
        train: bool = True,  # 默认训练集
        csvpath: str= None,  # csv文件路径
        pooled: bool = False,  # 默认不进行汇总的数据集
        debug: bool = False,
        X_dtype: torch.dtype = torch.float32,
        y_dtype: torch.dtype = torch.int64,
        data_path: str = None,  # 图像路径
        use_twice_aug: bool = False  # 新增参数控制是否使用双视图
    ):
        sz = 200  # 图像大小
        if train:
               # 弱增强（用于教师模型）
            weak_augmentations = albumentations.Compose([
                albumentations.HorizontalFlip(p=0.5),
                albumentations.RandomBrightnessContrast(0.1, 0.1),
                albumentations.RandomCrop(sz, sz),
                albumentations.Normalize(p=1.0),
            ])
            
            # 强增强（用于学生模型）
            strong_augmentations = albumentations.Compose([
                albumentations.RandomScale(0.07),
                albumentations.Rotate(50),
                albumentations.RandomBrightnessContrast(0.15, 0.1),
                albumentations.HorizontalFlip(p=0.5),
                albumentations.Affine(shear=0.1),
                albumentations.RandomCrop(sz, sz),
                albumentations.CoarseDropout(max_holes=8, min_holes=1, max_height=16, max_width=16, min_height=16, min_width=16, p=0.5),
                albumentations.Normalize(p=1.0),
            ])
            # 如果需要双视图且是训练集
            if use_twice_aug:
                augmentations = TransformTwice(weak_augmentations, strong_augmentations)
            else:
                augmentations = weak_augmentations
        else:
            augmentations = albumentations.Compose(
                [
                    albumentations.CenterCrop(sz, sz),
                    albumentations.Normalize(p=1.0),
                ]
            )
        super().__init__(
            X_dtype=X_dtype,
            y_dtype=y_dtype,
            augmentations=augmentations,
            data_path=data_path,
            csvpath=csvpath
        )

        self.center = center
        self.train_test = "train" if train else "test"
        self.pooled = pooled
        self.key = self.train_test + "_" + str(self.center)
        df = pd.read_csv(self.dic["train_test_split"])

        if self.pooled:  # 汇聚
            df2 = df.query("fold == '" + self.train_test + "' ").reset_index(drop=True)  # query 方法根据条件筛选数据   reset_index(drop=True)：重置索引

        if not self.pooled:  # 不汇聚
            assert center in range(6)  # 断言，如果center不在0-5之间，则抛出异常
            df2 = df.query("fold2 == '" + self.key + "' ").reset_index(drop=True)    

        images = df2.image.tolist()  # 将df2.image转换为列表
        self.image_paths = [
            os.path.join(self.dic["input_preprocessed"], image_name + ".jpg")
            for image_name in images
        ]
        self.targets = df2.target  # 获取标签
        self.centers = df2.center  # 获取中心
        
        # 添加标签分布检查代码
        target_counter = Counter(self.targets)
        if len(self.targets) > 0 and (len(target_counter) == 1 or len(self.targets) < 10):
            print(f"警告：Client-{center} {'训练' if train else '测试'} 标签分布: {dict(sorted(target_counter.items()))}, 样本数: {len(self.targets)}")
        
        # 检查是否存在空客户端
        if len(self.targets) == 0:
            print(f"警告：Client-{center} 没有数据！")


