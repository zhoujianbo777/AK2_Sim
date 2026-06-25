import torch, sklearn, imblearn, mlflow, onnx, dvc
import torch.nn as nn

print("=== AK2 模型开发环境验证 ===")
print(f"PyTorch:          {torch.__version__}")
print(f"CUDA可用:          {torch.cuda.is_available()}")
print(f"GPU:              {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
print(f"scikit-learn:     {sklearn.__version__}")
print(f"imbalanced-learn: {imblearn.__version__}")
print(f"MLflow:           {mlflow.__version__}")
print(f"ONNX:             {onnx.__version__}")
print(f"DVC:              {dvc.__version__}")
print()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# M1 MLP 前向测试
mlp = nn.Sequential(nn.Linear(240, 128), nn.ReLU(), nn.Linear(128, 12), nn.Sigmoid()).to(device)
x = torch.randn(4, 240).to(device)
out = mlp(x)
print(f"M1 MLP  前向测试: input{list(x.shape)} -> output{list(out.shape)} [OK]")

# M2 CNN 前向测试
cnn = nn.Sequential(
    nn.Conv1d(1, 16, 7, padding=3), nn.ReLU(), nn.MaxPool1d(2),
    nn.Conv1d(16, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
    nn.Flatten(), nn.Linear(64, 9)
).to(device)
x2 = torch.randn(4, 1, 256).to(device)
out2 = cnn(x2)
print(f"M2 CNN  前向测试: input{list(x2.shape)} -> output{list(out2.shape)} [OK]")

print()
print("=== 全部验证通过 ===")
