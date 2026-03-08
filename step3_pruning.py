import torch
import torchvision
import torchvision.transforms as transforms
import torch_pruning as tp
import torch.nn as nn
import torch.optim as optim
import os

print("========================================")
print("[系统配置] 正在探测云端 GPU 算力...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type != 'cuda':
    raise SystemError("❌ 没有检测到 GPU！请按照【动作一】重新设置 T4 GPU。")
print(f"✅ 成功挂载云端算力核心: {torch.cuda.get_device_name(0)}")

# 1. 数据管道 (Data Pipeline)
# 训练集和测试集分离：训练集用来长肌肉，测试集用来测力量
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("[数据管道] 拉取 CIFAR-10 全量数据集 (60,000 张图)...")
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2) # Batch 调大，加速训练

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

# 2. 模型重构与剪枝 (Pruning 50%)
print("[网络结构] 正在执行 50% 结构化剪枝...")
model = torchvision.models.resnet18(weights='DEFAULT')
model.fc = nn.Linear(model.fc.in_features, 10)
model.eval()

example_inputs = torch.randn(1, 3, 224, 224)
imp = tp.importance.MagnitudeImportance(p=1)
ignored_layers = [m for m in model.modules() if isinstance(m, torch.nn.Linear)]

pruner = tp.pruner.MagnitudePruner(model, example_inputs, importance=imp, pruning_ratio=0.5, ignored_layers=ignored_layers)
pruner.step()
model = model.to(device)

# 3. 康复训练策略 (Fine-Tuning)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4) # 稍微调大了一点学习率，加速恢复
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20) # 引入学术界标配：余弦退火学习率

# 4. 挂机主循环 (Epochs)
EPOCHS = 20 # 为了让你今晚能拿到结果，我们先跑 20 轮（学术界通常 50-100 轮）
best_acc = 0.0
save_path = "best_pruned_resnet18.pth"

print("========================================")
print(f"🔥 开始全量康复训练 (目标: {EPOCHS} Epochs)...")

for epoch in range(EPOCHS):
    # --- 训练阶段 ---
    model.train()
    running_loss = 0.0
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    scheduler.step() # 更新学习率
    
    # --- 测试阶段 (验证准确率) ---
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad(): # 测试时不需要计算梯度，省显存
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    acc = 100 * correct / total
    avg_loss = running_loss / len(trainloader)
    
    # --- 学术日志打印与模型保存 ---
    print(f"[Epoch {epoch+1:02d}/{EPOCHS}] Train Loss: {avg_loss:.4f} | Test Accuracy: {acc:.2f}%")
    
    if acc > best_acc:
        print(f"   🌟 准确率创新高 ({best_acc:.2f}% -> {acc:.2f}%)，正在保存模型权重...")
        torch.save(model.state_dict(), save_path)
        best_acc = acc

print("========================================")
print(f"🎉 训练完美收官！最高准确率定格在: {best_acc:.2f}%")
print(f"💾 模型权重已保存至: {os.path.abspath(save_path)}")