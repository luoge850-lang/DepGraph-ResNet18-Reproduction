import torch
import torchvision
import torchvision.transforms as transforms
import torch_pruning as tp
import torch.nn as nn
import torch.optim as optim

print("========================================")
print("[系统级配置] 初始化微调验证环境...")

# 1. 硬件资源初始化
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. 数据流水线构建
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
subset_indices = list(range(1000))
trainset_subset = torch.utils.data.Subset(trainset, subset_indices)
trainloader = torch.utils.data.DataLoader(trainset_subset, batch_size=32, shuffle=True, num_workers=0)

# 3. 模型构造与结构化剪枝 (复现 50% 剪枝率)
print("[模型构造] 加载 ResNet-18 并执行结构化剪枝 (Ratio: 0.5)...")
model = torchvision.models.resnet18(weights='DEFAULT')
model.fc = nn.Linear(model.fc.in_features, 10)
model.eval()

example_inputs = torch.randn(1, 3, 224, 224)
imp = tp.importance.MagnitudeImportance(p=1)
ignored_layers = [m for m in model.modules() if isinstance(m, torch.nn.Linear)]

pruner = tp.pruner.MagnitudePruner(
    model, example_inputs, importance=imp, pruning_ratio=0.5, ignored_layers=ignored_layers
)
pruner.step()
model.to(device)

# 4. 优化器策略配置
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)

# 5. 模型微调训练循环 (纯净日志版)
print("========================================")
print("[微调训练] 启动前向/反向传播循环 (Epoch 1)...")
model.train()

running_loss = 0.0
total_batches = len(trainloader)

for i, (inputs, labels) in enumerate(trainloader):
    inputs, labels = inputs.to(device), labels.to(device)

    optimizer.zero_grad()             
    outputs = model(inputs)           
    loss = criterion(outputs, labels) 
    loss.backward()                   
    optimizer.step()                  

    running_loss += loss.item()
    
    # 核心优化：取消进度条，改为每 10 个 Batch 打印一次纯净日志
    if (i + 1) % 10 == 0 or (i + 1) == total_batches:
        print(f"  -> [Training Log] Batch {i+1:>2}/{total_batches} | 实时 Loss: {loss.item():.4f}")

avg_loss = running_loss / total_batches
print("========================================")
print(f"✅ [实验结果] 极速验证完成。当前平均 Loss: {avg_loss:.4f}")
print("========================================")