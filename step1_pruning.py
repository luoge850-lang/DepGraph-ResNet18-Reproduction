import torch
import torchvision
import torch_pruning as tp

print("========================================")
# 1. 实验设置：使用最新规范加载 ResNet-18 (消除 UserWarning)
model = torchvision.models.resnet18(weights='DEFAULT')
model.eval()
example_inputs = torch.randn(1, 3, 224, 224)

# 【学术指标提取】获取术前 Baseline 数据 (修正 API)
macs_before, params_before = tp.utils.count_ops_and_params(model, example_inputs)
print(f"📊 [Baseline] MACs: {macs_before / 1e9:.3f} G, Params: {params_before / 1e6:.3f} M")

# 2. 剪枝策略：设定 50% 剪枝率
imp = tp.importance.MagnitudeImportance(p=1)
ignored_layers = [m for m in model.modules() if isinstance(m, torch.nn.Linear)]
pruner = tp.pruner.MagnitudePruner(
    model,
    example_inputs,
    importance=imp,
    pruning_ratio=0.5, 
    ignored_layers=ignored_layers,
)

# 3. 执行剪枝
pruner.step()

# 【学术指标提取】获取术后剪枝数据 (修正 API)
macs_after, params_after = tp.utils.count_ops_and_params(model, example_inputs)
print("========================================")
print(f"📉 [Pruned] MACs: {macs_after / 1e9:.3f} G, Params: {params_after / 1e6:.3f} M")
print(f"🚀 [Reduction] MACs 减少了: {(1 - macs_after/macs_before)*100:.2f}%")
print(f"🚀 [Reduction] Params 减少了: {(1 - params_after/params_before)*100:.2f}%")