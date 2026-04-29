import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.ANN import ANN

# --------------------- 完全用你的本地文件 ---------------------
df = pd.read_csv('titanic_train_knn.csv')

# 你的标签列是 2urvived
y = df['2urvived']

# 删掉无用的 zero 列和无关列，只留有效特征
X = df.drop([
    '2urvived', 'Passengerid', 
    'zero','zero.1','zero.2','zero.3','zero.4','zero.5','zero.6',
    'zero.7','zero.8','zero.9','zero.10','zero.11','zero.12',
    'zero.13','zero.14','zero.15','zero.16','zero.17','zero.18'
], axis=1)

# 处理缺失值
X = X.fillna(0)

# 文字转数字
X = pd.get_dummies(X, columns=['Sex', 'Embarked'])

# 转 numpy 数组，彻底解决 KeyError
X = X.values
y = y.values

# 划分训练集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 训练
ann = ANN(task_type='classification', hidden_dims=[64, 32], output_dim=1)
ann.train(X_train, y_train, epochs=30, lr=0.01)

# 保存
ann.save_model('./models/ann_titanic.pkl')
ann.plot_loss_curve('./results/ann_titanic_loss.png')

# 输出结果
y_pred = ann.predict(X_test)
acc = np.mean(y_pred == y_test)
print(f"✅ 训练完成！准确率：{acc:.4f}")