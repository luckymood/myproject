import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
from src.ANN import ANN

# 读你的本地文件，和训练时一致
df = pd.read_csv('titanic_train_knn.csv')
X = df.drop([
    '2urvived', 'Passengerid',
    'zero','zero.1','zero.2','zero.3','zero.4','zero.5','zero.6',
    'zero.7','zero.8','zero.9','zero.10','zero.11','zero.12',
    'zero.13','zero.14','zero.15','zero.16','zero.17','zero.18'
], axis=1)
y = df['2urvived']

# 预处理
X = X.fillna(0)
X = pd.get_dummies(X, columns=['Sex', 'Embarked'])
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 加载模型
with open('./models/ann_titanic.pkl', 'rb') as f:
    model_data = pickle.load(f)
ann = ANN(task_type='classification', hidden_dims=model_data['hidden_dims'], output_dim=model_data['output_dim'])
ann.weights = model_data['weights']
ann.biases = model_data['biases']

# 预测并输出准确率
y_pred = ann.predict(X)
acc = np.mean(y_pred == y)
print(f"✅ Titanic测试集准确率：{acc:.4f}")