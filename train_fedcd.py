# train_fedcd.py
import torch
from torch.utils.data import DataLoader
from networks.model import FedCDModel
from local_supervised import LocalSupervisedClient
from local_unsupervised import LocalUnsupervisedClient
from options import parse_args
import numpy as np
import copy
import os
def main():
    args = parse_args()
    
    # 强制使用CUDA，使用args.gpu指定的GPU ID
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        torch.cuda.set_device(args.gpu)
        print(f"使用GPU {args.gpu}：{torch.cuda.get_device_name(device)}")
    else:
        device = torch.device('cpu')
        print("警告：使用CPU进行训练，这可能会很慢")
    
    # 将设备信息添加到args
    args.device = str(device)
    
    num_clients = 6
    supervised_client_id = 0
    clients = []
    for client_id in range(num_clients):
        if client_id == supervised_client_id:
            clients.append(LocalSupervisedClient(client_id, args, device))
        else:
            clients.append(LocalUnsupervisedClient(client_id, args, device))
    
    global_model = FedCDModel(out_size=8, args=args)
    global_model.to(device)
    print("Preheating supervised client...")
    
    # 预热阶段，只训练有监督客户端（不传递global_model和current_round参数）
    for epoch in range(args.warm_epochs):
        supervised_loss = clients[supervised_client_id].train()  # 预热阶段不需要传递额外参数
        print(f"Epoch {epoch + 1}/{args.warm_epochs}, Supervised Client Loss: {supervised_loss:.4f}")   
    
    # 获取预热后的模型参数
    preheat_params = clients[supervised_client_id].get_model_params()
    global_model.load_state_dict(preheat_params)
    
    # 开始联邦学习阶段
    print("Starting federated learning...")
    
    # 在联邦学习开始时为每个无标签客户端的ReweightLoss设置alpha和alpha_n值
    for client_id, client in enumerate(clients):
        if client_id != supervised_client_id:
            client.reweight_loss.alpha = 0.01  # 第1轮的alpha值
            client.reweight_loss.alpha_n = 0.1  # 最后一轮的alpha值
    
    lu_total = 0.0  # 所有无监督客户端的损失总和
    lce_total = 0.0  # 有监督客户端的损失
    
    # 计算总样本数
    total_samples = sum(len(client.dataset) for client in clients)
    
    for epoch in range(args.fl_epochs):
        local_models = []
        client_losses = []
        
        # 客户端训练阶段
        for client in clients:
            current_round = epoch + 1
            
            # 联邦学习阶段，对每个客户端都传递global_model和current_round参数
            loss = client.train(global_model, current_round)
            client_losses.append(loss)
            
            # 计算客户端权重（样本数/总样本数）
            client_weight = len(client.dataset) / total_samples
            
            # 累计有监督和无监督的损失（带权重）
            if client.client_id == supervised_client_id:
                lce_total += loss * client_weight  # 有监督客户端的损失
            else:
                lu_total += loss * client_weight  # 无监督客户端的损失
                
            print(f"Epoch {epoch + 1}/{args.fl_epochs}, Client {client.client_id} Loss: {loss:.4f}")
            local_models.append(client.get_model_params())
        
        # 计算并输出平均损失
        avg_round_loss = sum(client_losses) / len(client_losses)
        print(f"Epoch {epoch + 1}/{args.fl_epochs}, Average Loss: {avg_round_loss:.4f}")
        
        # 计算联邦学习的总损失
        total_fl_loss = lu_total + lce_total
        print(f"Epoch {epoch + 1}/{args.fl_epochs}, 联邦学习总损失 (Lu + Lce): {total_fl_loss:.4f}")
        
        # 服务器聚合模型参数
        client_samples = [len(client.dataset) for client in clients]  # 获取每个客户端的样本数
        global_model_params = fedavg(local_models, device, client_samples)
        global_model.load_state_dict(global_model_params)
        
        # 分发模型参数给所有客户端
        for client in clients:
            client.set_model_params(global_model_params)
        
        # 测试阶段
        client_metrics = []
        for client in clients:
            # 期望 test() 返回 test_loss, bacc, acc, auc, precision
            test_result = client.test()
            if isinstance(test_result, tuple) and len(test_result) == 5:
                test_loss, bacc, acc, auc, precision = test_result
            else:
                # 兼容旧接口
                test_loss, bacc = test_result
                acc = auc = precision = 0.0
            client_metrics.append((bacc, acc, auc, precision))
            print(f"Epoch {epoch + 1}/{args.fl_epochs}, Client {client.client_id} Test Loss: {test_loss:.4f}, BACC: {bacc:.4f}, Acc: {acc:.4f}, AUC: {auc:.4f}, Precision: {precision:.4f}")
        
        # 计算并输出平均指标
        avg_bacc = sum([m[0] for m in client_metrics]) / len(client_metrics)
        avg_acc = sum([m[1] for m in client_metrics]) / len(client_metrics)
        avg_auc = sum([m[2] for m in client_metrics]) / len(client_metrics)
        avg_precision = sum([m[3] for m in client_metrics]) / len(client_metrics)
        print(f"Epoch {epoch + 1}/{args.fl_epochs}, Average BACC: {avg_bacc:.4f}, Average Acc: {avg_acc:.4f}, Average AUC: {avg_auc:.4f}, Average Precision: {avg_precision:.4f}")
        
        # 重置每轮的损失累计
        lu_total = 0.0
        lce_total = 0.0
        
    # 保存最终模型
    torch.save(global_model.state_dict(), 'fedcd_model.pth')
    print("训练完成，模型已保存为 fedcd_model.pth")

def fedavg(local_models, device=None, client_samples=None):
    """
    使用加权平均进行模型聚合
    Args:
        local_models: 本地模型参数列表
        device: 设备
        client_samples: 每个客户端的样本数列表
    """
    avg_model = copy.deepcopy(local_models[0])
    num_models = len(local_models)
    
    # 如果未指定设备，则获取第一个参数的设备
    if device is None and len(avg_model) > 0:
        first_param = next(iter(avg_model.values()))
        device = first_param.device
    
    # 计算总样本数
    total_samples = sum(client_samples) if client_samples else num_models
    
    for key in avg_model.keys():
        # 记录原始数据类型
        original_dtype = avg_model[key].dtype
        
        # 使用浮点数进行累加，确保在GPU上
        sum_tensor = torch.zeros_like(avg_model[key], dtype=torch.float, device=device)
        for i in range(num_models):
            # 计算权重
            weight = client_samples[i] / total_samples if client_samples else 1.0 / num_models
            # 将每个模型的参数移到GPU并转换为浮点数，乘以权重
            sum_tensor += local_models[i][key].to(device=device, dtype=torch.float) * weight
            
        # 将加权平均值强制转换回原始数据类型，并确保在指定设备上
        avg_model[key] = sum_tensor.to(device=device, dtype=original_dtype)

    return avg_model

if __name__ == '__main__':
    main()