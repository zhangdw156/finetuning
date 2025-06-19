import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
from accelerate import Accelerator
accelerator = Accelerator(mixed_precision="bf16")
device = accelerator.device

# 设置随机种子与设备
def setup_env(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    device = accelerator.device
    print(f"使用设备: {device}, 种子: {seed}")
    return device

# 数据加载与预处理（针对CIFAR-10优化）
def get_cifar10_loaders(batch_size=128, val_ratio=0.1):
    # 数据增强（针对32x32图像优化）
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    # 加载完整训练集
    train_dataset = datasets.CIFAR10(
        root='../dataset/', train=True, download=False, transform=train_transform
    )
    
    # 划分训练集与验证集
    num_train = len(train_dataset)
    indices = list(range(num_train))
    val_size = int(np.floor(val_ratio * num_train))
    
    np.random.seed(42)
    np.random.shuffle(indices)
    
    train_idx, val_idx = indices[val_size:], indices[:val_size]
    print(f"训练集索引数: {len(train_idx)} ({len(train_idx)/num_train:.1%})")
    print(f"验证集索引数: {len(val_idx)} ({len(val_idx)/num_train:.1%})")
    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_idx)
    
    # 加载测试集
    test_dataset = datasets.CIFAR10(
        root='../dataset/', train=False, download=False, transform=test_transform
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=4, pin_memory=True
    )
    
    val_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=val_sampler,
        num_workers=4, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, 10  # 10个类别

# 定义轻量级模型（针对CIFAR-10优化的ResNet）
class CIFARResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(CIFARResNet, self).__init__()
        base_model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        
        # 关键修复：将 base_model.conv1 赋值给 self.conv1
        self.conv1 = base_model.conv1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        base_model.maxpool = nn.Identity()  # 移除最大池化层
        
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4
        self.avgpool = base_model.avgpool
        self.fc = nn.Linear(base_model.fc.in_features, num_classes)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = self.layer1(self.layer1(self.conv1(x)))  # 双重前向增强特征
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# 训练函数（含混合精度与余弦退火学习率）
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                num_epochs=20, device='cuda', save_path='cifar10_model.pth'):
    model = model.to(device)
    best_val_acc = 0.0
    train_metrics = {'loss': [], 'acc': []}
    val_metrics = {'loss': [], 'acc': []}
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        correct, total = 0, 0
        
        with tqdm(train_loader, unit="batch") as tepoch:
            for inputs, labels in tepoch:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                
                with accelerator.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                
                accelerator.backward(loss)  # 自动处理混合精度和分布式训练
                optimizer.step()
                optimizer.zero_grad()
                
                running_loss += loss.item() * inputs.size(0)
                total += labels.size(0)
                correct += torch.sum(preds == labels.data)
                
                tepoch.set_postfix(loss=loss.item(), acc=correct/total)
            scheduler.step()
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct.double() / total
        train_metrics['loss'].append(epoch_loss)
        train_metrics['acc'].append(epoch_acc.item())
        
        # 验证阶段
        model.eval()
        running_loss = 0.0
        correct, total = 0, 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                
                running_loss += loss.item() * inputs.size(0)
                total += labels.size(0)
                correct += torch.sum(preds == labels.data)
        
        epoch_loss = running_loss / len(val_loader.dataset)
        epoch_acc = correct.double() / total
        val_metrics['loss'].append(epoch_loss)
        val_metrics['acc'].append(epoch_acc.item())
        
        # 更新学习率
        scheduler.step()
        
        # 打印进度
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} - "
              f"Val Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
        
        # 保存最佳模型
        if epoch_acc > best_val_acc:
            best_val_acc = epoch_acc
            torch.save(model.state_dict(), save_path)
            print(f"最佳模型已保存至 {save_path} (Acc: {best_val_acc:.4f})")
    
    print(f"训练完成，最佳验证准确率: {best_val_acc:.4f}")
    return model, train_metrics, val_metrics

# 评估模型
def evaluate_model(model, data_loader, device='cuda'):
    model.eval()
    correct, total = 0, 0
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += torch.sum(preds == labels.data)
    
    accuracy = correct.double() / total
    print(f"测试准确率: {accuracy:.4f} ({correct}/{total})")
    return accuracy

# 可视化训练过程
def plot_metrics(train_metrics, val_metrics):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 损失曲线
    ax1.plot(train_metrics['loss'], label='Train Loss')
    ax1.plot(val_metrics['loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.set_title('CIFAR-10 tran & val loss')
    
    # 准确率曲线
    ax2.plot(train_metrics['acc'], label='Train Acc')
    ax2.plot(val_metrics['acc'], label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.set_title('CIFAR-10 train & val acc')
    
    plt.tight_layout()
    plt.savefig('cifar10_metrics.png')
    plt.show()

# 主函数
def main():
    device = setup_env()
    
    # 加载CIFAR-10数据
    print("加载CIFAR-10数据集...")
    train_loader, val_loader, test_loader, num_classes = get_cifar10_loaders(
        batch_size=64, val_ratio=0.1
    )
    print(f"训练集大小: {len(train_loader.dataset)}, 验证集大小: {len(val_loader.dataset)}")
    
    # 创建模型
    print("创建CIFAR-10分类模型...")
    model = CIFARResNet(num_classes=num_classes)
    
    # 定义损失函数与优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-4)
    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )
    # 训练模型
    print("开始训练...")
    model, train_metrics, val_metrics = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        num_epochs=20, device=device, save_path='cifar10_model.pth'
    )
    
    # 评估模型
    print("评估模型...")
    test_acc = evaluate_model(model, test_loader, device)
    
    # 可视化
    print("可视化训练指标...")
    plot_metrics(train_metrics, val_metrics)

if __name__ == "__main__":
    main()