import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from engine import Value
from nn import MLP

random.seed(42)
np.random.seed(42)

X, y = make_circles(n_samples=30, factor=0.4, noise=0.08, random_state=42)
xs = X.tolist()
ys = [1.0 if label == 1 else -1.0 for label in y]

model = MLP(2, [8, 8, 1])
print(f"Tổng số tham số (weights + biases) của mạng: {len(model.parameters())}\n")

learning_rate = 0.15
epochs = 200

for epoch in range(epochs):
    ypred = [model(x) for x in xs]
    
    loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred)) * (1.0 / len(ys))

    for p in model.parameters():
        p.grad = 0.0

    loss.backward()

    for p in model.parameters():
        p.data += -learning_rate * p.grad
        
    if epoch % 20 == 0 or epoch == epochs - 1:
        correct = sum(1 for ygt, yout in zip(ys, ypred) if (yout.data > 0) == (ygt > 0))
        acc = (correct / len(ys)) * 100
        print(f"Epoch {epoch:3d} | Mean Loss: {loss.data:.6f} | Accuracy: {correct}/{len(ys)} ({acc:.1f}%)")

print("\n--- KẾT QUẢ SAU KHI HUẤN LUYỆN ---")
for i in range(5):
    print(f"Mẫu {i+1}: Tọa độ ({xs[i][0]:+.2f}, {xs[i][1]:+.2f}) -> Nhãn thật: {ys[i]:+.1f} | Dự đoán: {ypred[i].data:+.4f}")

grid_size = 35
xx, yy = np.meshgrid(np.linspace(-1.4, 1.4, grid_size), np.linspace(-1.4, 1.4, grid_size))
Z = np.zeros_like(xx)

for i in range(grid_size):
    for j in range(grid_size):
        Z[i, j] = model([xx[i, j], yy[i, j]]).data

plt.figure(figsize=(6, 6))
plt.contourf(xx, yy, Z, levels=[-1.5, 0.0, 1.5], colors=['#ffcccc', '#ccffcc'], alpha=0.8)
plt.contour(xx, yy, Z, levels=[0.0], colors='black', linewidths=2, linestyles='--')

for (x1, x2), label in zip(xs, ys):
    color = 'green' if label == 1.0 else 'red'
    marker = 'o' if label == 1.0 else 'X'
    plt.scatter(x1, x2, color=color, s=100, edgecolors='black', marker=marker)

plt.title(f"Micrograd - Concentric Circles (Loss: {loss.data:.4f})")
plt.xlabel("x1")
plt.ylabel("x2")
plt.grid(True, alpha=0.3)
plt.savefig("decision_boundary.png", dpi=120, bbox_inches='tight')
print("\nĐã lưu biểu đồ trực quan vào file 'decision_boundary.png'!")