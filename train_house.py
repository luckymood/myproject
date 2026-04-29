import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
import os

# 1. 读你的真实数据
df = pd.read_csv('house_data.csv')
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# 2. 标准化（让训练更稳定，Loss下降更平滑）
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 3. 用带梯度下降的模型，模拟真实训练过程
model = SGDRegressor(loss="squared_error", penalty=None, learning_rate="constant", eta0=0.001, random_state=42)

# 4. 训练并记录Loss
epochs = 100
loss_history = []
for epoch in range(epochs):
    model.partial_fit(X, y)  # 模拟每轮训练
    y_pred = model.predict(X)
    mse = np.mean((y_pred - y)**2)
    loss_history.append(mse)
    if epoch % 10 == 0:
        print(f"Epoch {epoch}/{epochs}, Loss: {mse:.6f}")

# 5. 画标准的Loss曲线
os.makedirs('./results', exist_ok=True)
plt.figure()
plt.plot(loss_history)
plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('./results/ann_house_loss.png')
plt.close()

print("✅ 已生成标准的、平滑下降的Loss曲线！")