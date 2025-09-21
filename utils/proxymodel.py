# proxymodel.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataset.dataset_isic2019 import FedIsic2019


class ProxyModel(nn.Module):
    def __init__(self):
        super(ProxyModel, self).__init__()
        self.confident_threshold = 0.4
        self.low_rank_threshold = 5
        self.high_rank_threshold = 6

    def forward(self, global_model, local_model, dataloader):
        # 确保所有张量都在同一个设备上
        device = next(local_model.parameters()).device
        
        # 处理 global_model 可能是 DataParallel 实例的情况
        if isinstance(global_model, nn.DataParallel):
            global_model = global_model.module
        
        # 确保 global_model 在正确的设备上
        global_model = global_model.to(device)
        global_model.eval()
        
        # 获取全局模型的预测
        confident_classes = []
        with torch.no_grad():
            for data, _ in dataloader:
                # 直接使用数据，不再解包
                inputs = data.to(device, non_blocking=True)
                outputs = global_model(inputs)

                # 如果 outputs 是一个元组，提取 logits
                if isinstance(outputs, tuple):
                    outputs = outputs[1]  # 假设 logits 是元组的第二个元素

                probabilities = F.softmax(outputs, dim=1)
                confident_classes.extend(torch.max(probabilities, dim=1)[1].cpu().numpy())

        # 统计每个类别的累积概率
        class_counts = np.bincount(confident_classes, minlength=8)
        class_confidences = class_counts / len(confident_classes)

        # 动态计算 confident_class_indices
        confident_class_indices = np.where(class_confidences > self.confident_threshold)[0]
        confident_class_indices = torch.tensor(confident_class_indices).to(device)

        # 确定不可靠样本
        unreliable_samples = []
        local_model.eval()
        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(dataloader):
                # 直接使用数据，不再解包
                inputs = data.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                outputs = local_model(inputs)

                # 如果 outputs 是一个元组，提取 logits
                if isinstance(outputs, tuple):
                    outputs = outputs[1]  # 假设 logits 是元组的第二个元素

                probabilities = F.softmax(outputs, dim=1)
                sorted_indices = torch.argsort(probabilities, dim=1)
                for i in range(len(sorted_indices)):
                    top_classes = sorted_indices[i, -self.high_rank_threshold:self.low_rank_threshold]
                    if not any(cls in confident_class_indices.cpu().numpy() for cls in top_classes.cpu().numpy()):
                        unreliable_samples.append(batch_idx * len(data) + i)  # 记录不可靠样本的索引

        # 计算权重因子
        num_unreliable = len(unreliable_samples)
        num_total = len(dataloader.dataset)
        w = torch.ones(num_total, device=device)
        
        # 修复张量大小不匹配问题
        if num_unreliable > 0:
            # 为不可靠样本和可靠样本批量设置权重
            unreliable_mask = torch.zeros(num_total, dtype=torch.bool, device=device)
            
            for idx in unreliable_samples:
                if idx < num_total:  # 确保索引在有效范围内
                    unreliable_mask[idx] = True
            
            # 为不可靠样本设置权重
            w[unreliable_mask] = 1.0 / max(1, num_unreliable)
            
            # 为可靠样本设置权重
            if (num_total - num_unreliable) > 0:
                w[~unreliable_mask] = 1.0 / (num_total - num_unreliable)
                
        # 如果在GPU上，清空缓存
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        return w
