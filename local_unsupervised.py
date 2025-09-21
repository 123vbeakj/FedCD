# local_unsupervised.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from networks.model import FedCDModel
from dataset.dataset_isic2019 import FedIsic2019
import albumentations as A
import numpy as np
import copy
from utils.DualTeacher import DualTeacher
from utils.proxymodel import ProxyModel
from utils.losses import ReweightLoss

class LocalUnsupervisedClient:
    def __init__(self, client_id, args, device=None):
        self.client_id = client_id
        self.args = args
        # 优先使用 args 中定义的 device
        self.device = torch.device(args.device) if hasattr(args, 'device') and args.device else \
                     (device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        
        # 打印使用的设备信息，便于排查
        if 'cuda' in str(self.device):
            print(f"Client {client_id} 使用 GPU: {self.device} - {torch.cuda.get_device_name(self.device)}")
        else:
            print(f"Client {client_id} 使用 CPU - 警告：这会使训练很慢")
        
        # 创建学生模型
        self.model = FedCDModel(out_size=8, args=args)       
        self.model.to(self.device)
        # 创建教师模型的独立副本 (使用 deepcopy)
        global_teacher_model = copy.deepcopy(self.model)
        local_teacher_model = copy.deepcopy(self.model)
        # 使用独立实例初始化 DualTeacher
        self.dual_teacher = DualTeacher(self.model, global_teacher_model, local_teacher_model)
        self.proxy_model = ProxyModel()
        self.reweight_loss = ReweightLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        # 添加交叉熵损失函数
        self.criterion = nn.CrossEntropyLoss().to(self.device)

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

    def train(self, global_model, current_round):
        # 确保模型在训练模式并在正确的设备上
        self.model.train()
        self.model = self.model.to(self.device)
        
        # 如果提供了全局模型，确保它也在正确的设备上
        if global_model is not None:
            global_model = global_model.to(self.device)
            
        total_loss = 0.0
        batch_count = 0
        
        # 使用CUDA事件来测量时间
        if 'cuda' in str(self.device):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        
        print(f"\n开始无监督客户端 {self.client_id} 的训练...")
        print(f"当前轮次: {current_round}")
        
        # 计算整个数据集的权重
        w = self.proxy_model(global_model, self.model, self.dataloader)
        print(f"代理模型权重形状: {w.shape}")
        print(f"代理模型权重统计: 最小值={w.min().item():.4f}, 最大值={w.max().item():.4f}, 平均值={w.mean().item():.4f}")
        
        for batch_idx, (data, targets) in enumerate(self.dataloader):
            # 将数据移到GPU，使用non_blocking=True加速数据传输
            data = data.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            
            print(f"\n批次 {batch_idx + 1}:")
            print(f"数据形状: {data.shape}")
            print(f"目标形状: {targets.shape}")
            
            self.optimizer.zero_grad()
            
            # 计算无监督损失 Lu
            loss_u = self.dual_teacher(data, None)  # 第二个参数不再使用
            
            print(f"无监督损失 (loss_u): {loss_u.item():.4f}")
            
            # 获取当前批次的权重
            batch_start_idx = batch_idx * self.args.batch_size
            batch_end_idx = min((batch_idx + 1) * self.args.batch_size, len(w))
            batch_weights = w[batch_start_idx:batch_end_idx]
            
            # 重加权无监督损失
            loss = self.reweight_loss(loss_u, batch_weights, current_round, self.args.fl_epochs)
            print(f"重加权后的损失: {loss.item():.4f}")
            
            # 确保损失不为0
            if torch.isnan(loss) or torch.isinf(loss) or loss.item() == 0:
                print(f"警告：损失值为 {loss.item()}，使用默认权重")
                loss = loss_u  # 直接使用原始损失，不进行mean操作
                print(f"使用原始损失: {loss.item():.4f}")
            
            # 反向传播和优化
            loss.backward()
            self.optimizer.step()
            
            # 使用 item() 避免保留计算图
            total_loss += loss.item()
            batch_count += 1
            
            # 定期清除缓存
            if 'cuda' in str(self.device) and (batch_idx + 1) % 10 == 0:
                torch.cuda.empty_cache()
                print(f"已清除GPU缓存，当前批次: {batch_idx + 1}")
        
        # 测量结束时间
        if 'cuda' in str(self.device):
            end_event.record()
            torch.cuda.synchronize()
            elapsed_time = start_event.elapsed_time(end_event) / 1000.0  # 转换为秒
            print(f"Client {self.client_id} 训练耗时: {elapsed_time:.2f} 秒")
            
        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        print(f"无监督客户端 {self.client_id} 平均损失: {avg_loss:.4f}")
        return avg_loss

    def test(self):
        # 确保模型在评估模式并在正确的设备上
        self.model.eval()
        self.model = self.model.to(self.device)
        
        test_loss = 0.0
        all_preds = []
        all_labels = []
        # 确保在测试时有损失函数
        if not hasattr(self, 'criterion'):
             self.criterion = nn.CrossEntropyLoss().to(self.device)

        with torch.no_grad():
            for data, targets in self.dataloader:
                # 注意：dataloader返回的是列表 [weak_data, strong_data], targets
                # 测试时通常只需要一个视图的数据
                if isinstance(data, list):
                    input_data = data[0] # 使用弱增强数据进行测试
                else:
                    input_data = data # 如果不是列表，直接使用

                # 使用non_blocking=True加速数据传输
                input_data = input_data.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                
                _, outputs = self.model(input_data)

                # 计算测试损失 (例如交叉熵损失)
                loss = self.criterion(outputs, targets)
                test_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                all_preds.append(predicted.cpu())
                all_labels.append(targets.cpu())

        # 清除GPU缓存
        if 'cuda' in str(self.device):
            torch.cuda.empty_cache()
            
        # 确保dataloader不为空
        if len(self.dataloader) > 0:
             test_loss /= len(self.dataloader) # 计算平均损失
        else:
             test_loss = 0.0 # 或者其他合适的默认值

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        num_classes = self.args.num_classes # 使用args中的类别数

        # 确保 num_classes > 0
        if num_classes <= 0:
             print(f"警告: num_classes 为 {num_classes}，无法计算 BACC。")
             return test_loss, 0.0, 0.0, 0.0, 0.0 # 返回默认BACC和其它指标

        # BACC
        recalls = []
        for c in range(num_classes):
            idx = (all_labels == c)
            class_total = idx.sum().item()
            if class_total == 0:
                recalls.append(0.0)
            else:
                correct = (all_preds[idx] == c).sum().item()
                recalls.append(float(correct) / class_total)
        bacc = sum(recalls) / float(num_classes)

        # Accuracy
        total = all_labels.size(0)
        correct = (all_preds == all_labels).sum().item()
        acc = float(correct) / total if total > 0 else 0.0

        # Precision (macro)
        precision = 0.0
        for c in range(num_classes):
            pred_c = (all_preds == c)
            tp = ((all_labels == c) & pred_c).sum().item()
            pred_total = pred_c.sum().item()
            if pred_total > 0:
                precision += tp / pred_total
        precision = precision / num_classes if num_classes > 0 else 0.0

        # AUC (macro, one-vs-rest)
        try:
            from sklearn.metrics import roc_auc_score
            # 需要概率输出
            all_probs = []
            with torch.no_grad():
                for data, targets in self.dataloader:
                    if isinstance(data, list):
                        input_data = data[0]
                    else:
                        input_data = data
                    input_data = input_data.to(self.device, non_blocking=True)
                    _, outputs = self.model(input_data)
                    probs = torch.softmax(outputs, dim=1)
                    all_probs.append(probs.cpu())
            all_probs = torch.cat(all_probs)
            # 转为one-hot
            all_labels_onehot = torch.zeros((all_labels.size(0), num_classes))
            all_labels_onehot.scatter_(1, all_labels.unsqueeze(1), 1)
            auc = roc_auc_score(all_labels_onehot.numpy(), all_probs.numpy(), average='macro', multi_class='ovr')
        except Exception as e:
            auc = 0.0

        return test_loss, bacc, acc, auc, precision
    
    def get_model_params(self):
        return self.model.state_dict()

    def set_model_params(self, model_params):
        self.model.load_state_dict(model_params)
        self.model = self.model.to(self.device)  # 确保模型在GPU上