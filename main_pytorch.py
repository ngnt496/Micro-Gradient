import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_circles

torch.manual_seed(42)
np.random.seed(42)

X, y = make_circles(n_samples=30, factor=0.4, noise=0.08, random_state=42)
xs = X.tolist()
ys = [1.0 if label == 1 else -1.0 for label in y]

X_tensor = torch.tensor(xs, dtype=torch.float32)
y_tensor = torch.tensor(ys, dtype=torch.float32).unsqueeze(1)


class PyTorchMLP(nn.Module):

  def __init__(self):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(2, 8),
        nn.Tanh(),
        nn.Linear(8, 8),
        nn.Tanh(),
        nn.Linear(8, 1),
        nn.Tanh(),
    )

  def forward(self, x):
    return self.net(x)


model = PyTorchMLP()

total_params = sum(p.numel() for p in model.parameters())
print(
    f"Tổng số tham số (weights + biases) của mạng PyTorch: {total_params}\n"
)

learning_rate = 0.15
epochs = 200
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

for epoch in range(epochs):
  optimizer.zero_grad()
  ypred = model(X_tensor)
  loss = criterion(ypred, y_tensor)

  loss.backward()
  optimizer.step()

  if epoch % 20 == 0 or epoch == epochs - 1:
    with torch.no_grad():
      preds = (ypred > 0).float()
      targets = (y_tensor > 0).float()
      correct = (preds == targets).sum().item()
      acc = (correct / len(ys)) * 100
    print(
        f"Epoch {epoch:3d} | Mean Loss: {loss.item():.6f} | Accuracy:"
        f" {correct}/{len(ys)} ({acc:.1f}%)"
    )

print("\n--- KẾT QUẢ SAU KHI HUẤN LUYỆN ---")
model.eval()
with torch.no_grad():
  ypred_final = model(X_tensor)
  for i in range(5):
    print(
        f"Mẫu {i+1}: Tọa độ ({xs[i][0]:+.2f}, {xs[i][1]:+.2f}) -> Nhãn thật:"
        f" {ys[i]:+.1f} | Dự đoán: {ypred_final[i].item():+.4f}"
    )

grid_size = 35
xx, yy = np.meshgrid(
    np.linspace(-1.4, 1.4, grid_size), np.linspace(-1.4, 1.4, grid_size)
)
grid_points = np.c_[xx.ravel(), yy.ravel()]
grid_tensor = torch.tensor(grid_points, dtype=torch.float32)

with torch.no_grad():
  Z_pred = model(grid_tensor).numpy().reshape(xx.shape)

plt.figure(figsize=(6, 6))
plt.contourf(
    xx,
    yy,
    Z_pred,
    levels=[-1.5, 0.0, 1.5],
    colors=['#ffcccc', '#ccffcc'],
    alpha=0.8,
)
plt.contour(
    xx, yy, Z_pred, levels=[0.0], colors='black', linewidths=2, linestyles='--'
)

for (x1, x2), label in zip(xs, ys):
  color = 'green' if label == 1.0 else 'red'
  marker = 'o' if label == 1.0 else 'X'
  plt.scatter(x1, x2, color=color, s=100, edgecolors='black', marker=marker)

plt.title(f"PyTorch - Concentric Circles (Loss: {loss.item():.4f})")
plt.xlabel("x1")
plt.ylabel("x2")
plt.grid(True, alpha=0.3)
plt.savefig("decision_boundary_pytorch.png", dpi=120, bbox_inches='tight')
print("\nĐã lưu biểu đồ trực quan vào file 'decision_boundary_pytorch.png'!")