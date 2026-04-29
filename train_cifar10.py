import numpy as np
from tensorflow.keras.datasets import cifar10
from sklearn.preprocessing import StandardScaler
from src.ANN import ANN

# 加载CIFAR10数据
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = X_train.reshape(X_train.shape[0], -1) / 255.0
X_test = X_test.reshape(X_test.shape[0], -1) / 255.0

# 标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 训练
ann = ANN(task_type='classification', hidden_dims=[256, 128, 64], output_dim=10)
ann.train(X_train, y_train.flatten(), epochs=20, lr=0.01)

# 保存模型和曲线
ann.save_model('./models/ann_cifar10.pkl')
ann.plot_loss_curve('./results/ann_cifar10_loss.png')
ann.plot_accuracy_curve('./results/ann_cifar10_acc.png')

print("✅ CIFAR10训练完成！")