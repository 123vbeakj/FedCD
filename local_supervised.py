# local_supervised.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from networks.model import FedCDModel
from dataset.dataset_isic2019 import FedIsic2019
import albumentations as A
import numpy as np
import os


class LocalSupervisedClient:
    def __init__(self, client_id, args, device=None):
        self.client_id = client_id
        self.args = args
        # 优先使用 args 中定义的 device
        self.device = torch.device(args.device) if hasattr(args, 'device') and args.device else \
                     (device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        
        # 打印使用的设备信息，便于排查
        if 'cuda' in str(self.device):
            print(f"有监督 Client {client_id} 使用 GPU: {self.device} - {torch.cuda.get_device_name(self.device)}")
        else:
            print(f"有监督 Client {client_id} 使用 CPU - 警告：这会使训练很慢")
        
        self.model = FedCDModel(out_size=8, args=args)
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=args.supervised_lr, weight_decay=args.weight_decay)

        # 数据预处理
        self.transform = A.Compose([
            A.Resize(240, 240),
            A.RandomCrop(224, 224),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(0.1, 0.1),
            A.Normalize(p=1.0),
            ])

        # 加载数据集
        self.dataset = FedIsic2019(
            center=client_id,
            train=True,
            csvpath=args.csvpath,
            data_path=args.data_path,
            use_twice_aug=False
        )
        
        # 获取合适的 worker 数量
        num_workers = args.num_workers if hasattr(args, 'num_workers') else 4
        
        # 启用 pin_memory 和多线程数据加载
        self.dataloader = DataLoader(
            self.dataset, 
            batch_size=args.batch_size, 
            shuffle=True,
            num_workers=num_workers,
            pin_memory='cuda' in str(self.device),
            drop_last=False
        )

    def train(self, global_model=None, current_round=None):
        # 确保模型在训练模式并在正确的设备上
        self.model.train()
        self.model = self.model.to(self.device)
        
        # 使用CUDA事件来测量时间
        if 'cuda' in str(self.device):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            
        total_loss = 0.0
        batch_count = 0
        
        for batch_idx, (data, targets) in enumerate(self.dataloader):
            # 将数据移到GPU，使用non_blocking=True加速数据传输
            data = data.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            
            # 清除梯度
            self.optimizer.zero_grad()
            
            # 前向传播
            _, outputs = self.model(data)
            loss = self.criterion(outputs, targets)
            
            # 反向传播和优化
            loss.backward()
            self.optimizer.step()
            
            # 使用 item() 避免保留计算图
            total_loss += loss.item()
            batch_count += 1
            
            # 定期清除缓存
            if 'cuda' in str(self.device) and (batch_idx + 1) % 10 == 0:
                torch.cuda.empty_cache()
        
        # 测量结束时间
        if 'cuda' in str(self.device):
            end_event.record()
            torch.cuda.synchronize()
            elapsed_time = start_event.elapsed_time(end_event) / 1000.0  # 转换为秒
            print(f"有监督 Client {self.client_id} 训练耗时: {elapsed_time:.2f} 秒")
        
        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        return avg_loss

    def test(self):
        # 确保模型在评估模式并在正确的设备上
        self.model.eval()
        self.model = self.model.to(self.device)
        
        test_loss = 0.0
        correct = 0
        with torch.no_grad():
            for data, targets in self.dataloader:
                # 使用non_blocking=True加速数据传输
                data = data.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                
                # 前向传播
                _, outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                test_loss += loss.item()
                
                # 计算准确率
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == targets).sum().item()
        
        # 清除GPU缓存
        if 'cuda' in str(self.device):
            torch.cuda.empty_cache()
            
        test_loss /= len(self.dataloader)
        accuracy = correct / len(self.dataset)
        return test_loss, accuracy

    def get_model_params(self):
        return self.model.state_dict()

    def set_model_params(self, model_params):
        self.model.load_state_dict(model_params)
        self.model = self.model.to(self.device)  # 确保模型在GPU上