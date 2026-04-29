import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.linear_model import SGDRegressor

# 读你的本地文件
df = pd.read_csv('house_data.csv')
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# 标准化（和训练时一致）
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 加载模型（如果是用我最后给的SGD版本，直接重新训练即可）
model = SGDRegressor(loss="squared_error", penalty=None, learning_rate="constant", eta0=0.001, random_state=42)
model.fit(X, y)

# 预测并输出MSE
y_pred = model.predict(X)
mse = np.mean((y_pred - y)**2)
print(f"✅ 房价预测测试集MSE：{mse:.6f}")