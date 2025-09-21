import torch.nn as nn
# from networks.resnet import resnet18
import torchvision.models as models

class FedCDModel(nn.Module):
    def __init__(self, out_size, args):
        super().__init__()
        # 使用预训练的ResNet18，但不使用其最后的分类层
        self.backbone = models.resnet18(weights='IMAGENET1K_V1')
        # 移除最后的全连接层
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # 添加自定义的分类头
        self.mlp1 = nn.Linear(512, args.proto_dim)  # ResNet18的特征维度是512
        self.mlp2 = nn.Linear(args.proto_dim, out_size)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 确保输入是4D张量 [batch_size, channels, height, width]
        if x.dim() == 3:
            x = x.unsqueeze(0)  # 添加batch维度
        
        # 获取特征
        features = self.backbone(x)
        features = features.view(features.size(0), -1)  # 展平特征
        
        # 通过MLP层
        proto = self.relu(self.mlp1(features))
        out = self.mlp2(proto)
        
        return proto, out
# import torch.nn as nn
# from networks.resnet import resnet18

# class FedCDModel(nn.Module):
#     def __init__(self, out_size, args):
#         super().__init__()
#         self.backbone = resnet18(args, pretrained=True, num_classes=out_size)
#         self.mlp1 = nn.Linear(args.proto_dim, args.proto_dim)
#         self.mlp2 = nn.Linear(args.proto_dim, out_size)
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, x):
#         proto, _ = self.backbone(x)
#         proto = self.relu(self.mlp1(proto))
#         out = self.mlp2(proto)
#         return proto, out