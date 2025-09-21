import torch
import torch.nn as nn
import torch.nn.functional as F
import albumentations as A
import numpy as np
from PIL import Image


class DualTeacher(nn.Module):
    def __init__(self, student, global_teacher, local_teacher, ema_decay=0.999):
        super(DualTeacher, self).__init__()
        self.student = student
        self.global_teacher = global_teacher
        self.local_teacher = local_teacher
        self.temperature = 0.5#论文中0.5
        self.ema_decay = ema_decay  # EMA更新系数
        
        # 获取设备
        self.device = next(student.parameters()).device
        
        # 确保所有模型都在相同设备上
        self.global_teacher.to(self.device)
        self.local_teacher.to(self.device)
        
        # 停止教师模型的梯度更新
        for param in self.global_teacher.parameters():
            param.requires_grad = False
        for param in self.local_teacher.parameters():
            param.requires_grad = False
            
        # 调试标志
        self.debug = False
        
        # 定义弱增强 - 只包含扰动操作，不改变图像尺寸
        self.weak_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(0.1, 0.1),
            A.Normalize(p=1.0),
        ])

    def apply_transform(self, data, transform):
        """应用数据增强，保持原始尺寸"""
        # 将tensor转换为numpy数组
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        
        # 确保数据形状正确 [B, C, H, W] -> [B, H, W, C]
        if data.shape[1] == 3:
            data = np.transpose(data, (0, 2, 3, 1))
        
        # 对每个样本应用变换
        transformed_data = []
        for img in data:
            # 确保像素值在0-255范围内
            img = (img * 255).astype(np.uint8)
            transformed = transform(image=img)['image']
            transformed_data.append(transformed)
        
        # 转换为tensor [B, H, W, C] -> [B, C, H, W]
        transformed_data = np.array(transformed_data)
        transformed_data = np.transpose(transformed_data, (0, 3, 1, 2))
        transformed_data = torch.from_numpy(transformed_data).float().to(self.device)
        
        return transformed_data

    def forward(self, data, _):
        # 对输入数据进行三次相同的弱增强（每次都会产生不同的扰动）
        student_data = self.apply_transform(data, self.weak_transform)
        global_teacher_data = self.apply_transform(data, self.weak_transform)
        local_teacher_data = self.apply_transform(data, self.weak_transform)
        
        print(f"学生模型输入形状: {student_data.shape}")
        print(f"全局教师模型输入形状: {global_teacher_data.shape}")
        print(f"本地教师模型输入形状: {local_teacher_data.shape}")
        
        # 学生模型前向传播
        features_student, logits_student = self.student(student_data)
        
        # 全局教师模型前向传播
        with torch.no_grad():
            features_global, logits_global = self.global_teacher(global_teacher_data)
            global_teacher_out = F.softmax(logits_global / self.temperature, dim=1)

        # 本地教师模型前向传播
        with torch.no_grad():
            features_local, logits_local = self.local_teacher(local_teacher_data)
            local_teacher_out = F.softmax(logits_local / self.temperature, dim=1)

        # 计算学生模型的输出
        student_weak_out = F.softmax(logits_student / self.temperature, dim=1)

        # 锐化操作
        student_weak_out_sharpened = student_weak_out ** (1 / self.temperature)
        student_weak_out_sharpened = student_weak_out_sharpened / student_weak_out_sharpened.sum(dim=1, keepdim=True)

        global_teacher_out_sharpened = global_teacher_out ** (1 / self.temperature)
        global_teacher_out_sharpened = global_teacher_out_sharpened / global_teacher_out_sharpened.sum(dim=1, keepdim=True)

        local_teacher_out_sharpened = local_teacher_out ** (1 / self.temperature)
        local_teacher_out_sharpened = local_teacher_out_sharpened / local_teacher_out_sharpened.sum(dim=1, keepdim=True)

        # 计算损失
        mse_loss = nn.MSELoss()
        loss_mse_global = mse_loss(student_weak_out_sharpened, global_teacher_out_sharpened)
        loss_mse_local = mse_loss(student_weak_out_sharpened, local_teacher_out_sharpened)
        
        # 计算KL散度
        v_local = F.kl_div(F.log_softmax(student_weak_out, dim=1), local_teacher_out, reduction='batchmean')
        v_global = F.kl_div(F.log_softmax(student_weak_out, dim=1), global_teacher_out, reduction='batchmean')

        # 调整损失系数
        loss_local = torch.exp(-v_local) * loss_mse_local
        loss_global = torch.exp(-v_global) * loss_mse_global
        lamba1 = 0.02  # 论文中0.02
        lamba2 = 0.02
        
        # 总损失
        loss = lamba1 * loss_local + lamba2 * loss_global
        
        # 确保损失不为0
        if loss.item() == 0:
            print("[DualTeacher] 警告: 损失为0，使用默认损失值")
            loss = torch.tensor(0.1, device=self.device, requires_grad=True)
        
        # 调试输出
        if self.debug:
            print(f"[DualTeacher] MSE损失(Global): {loss_mse_global.item():.6f}")
            print(f"[DualTeacher] MSE损失(Local): {loss_mse_local.item():.6f}")
            print(f"[DualTeacher] KL散度(Local): {v_local.item():.6f}")
            print(f"[DualTeacher] KL散度(Global): {v_global.item():.6f}")
            print(f"[DualTeacher] 最终损失: {loss.item():.6f}")

        return loss
    
    def update_local_teacher(self):
        """
        使用EMA更新本地教师模型
        在每个训练轮次结束后调用此方法
        """
        with torch.no_grad():
            for student_param, local_teacher_param in zip(self.student.parameters(), self.local_teacher.parameters()):
                local_teacher_param.data = self.ema_decay * local_teacher_param.data + (1 - self.ema_decay) * student_param.data
                
            # 调试输出
            if self.debug:
                print("[DualTeacher] 已更新本地教师模型 (EMA)")
                
    def set_global_teacher(self, global_model):
        """
        设置全局教师模型
        在每轮联邦学习结束后，服务器聚合后调用此方法
        """
        # 确保全局模型与全局教师在同一设备上
        for name, param in global_model.state_dict().items():
            self.global_teacher.state_dict()[name].copy_(param.to(self.device))
            
        # 确保全局教师模型处于评估模式
        self.global_teacher.eval()
        
        # 调试输出
        if self.debug:
            print("[DualTeacher] 已更新全局教师模型")