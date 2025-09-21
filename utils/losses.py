# loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class ReweightLoss(nn.Module):
    def __init__(self):
        super(ReweightLoss, self).__init__()
        self.alpha = 0.01  
        self.alpha_n = 0.1  
       
    def forward(self, loss, w, current_round, total_rounds):
        device = loss.device
        
        # 当使用标量损失时，确保张量化
        if not isinstance(loss, torch.Tensor):
            loss = torch.tensor(loss, device=device)
            
        # 确保权重在同一设备上
        w = w.to(device)
        
        # 计算当前轮次的 alpha 值
        alpha_t = self.alpha + (self.alpha_n - self.alpha) * (current_round - 1) / (total_rounds - 1)
        alpha_t = torch.tensor(alpha_t, device=device)
        
        # 应用重加权并计算平均损失
        reweight_loss = (alpha_t * w * loss).mean()  # 添加mean()操作

        return reweight_loss