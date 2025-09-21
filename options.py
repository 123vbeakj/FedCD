import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--proto_dim', type=int, default=512, help='原型特征维度')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate for unsupervised clients')
    parser.add_argument('--supervised_lr', type=float, default=0.02, help='Learning rate for supervised client')
    parser.add_argument('--fl_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--warm_epochs', type=int, default=2, help='Number of warmup')
    parser.add_argument('--data_path', type=str, default='dataset/ISIC_2019/ISIC_2019_Training_Input_preprocessed',help='Path to dataset')
    parser.add_argument('--csvpath', type=str, default='dataset/ISIC_2019/train_test_split.csv', help='Path to CSV file')
    parser.add_argument('--num_classes', type=int, default=8, help='Number of classes')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use (-1 for CPU)')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='cuda:0', help='设备 (例如: cuda:0, cpu)')
    args = parser.parse_args()
    return args