# 神经网络多任务训练项目
该项目实现了三个经典任务的神经网络训练：Titanic 乘客生存预测、波士顿房价回归、CIFAR10 图像分类。

## 项目结构
myproject/├── src/ # 核心模块│ └── ANN.py # 神经网络类├── train_titanic_ann.py # Titanic 分类训练脚本├── train_house.py # 房价回归训练脚本├── train_cifar10.py # CIFAR10 分类训练脚本├── test_titanic_ann.py # Titanic 测试脚本├── test_house.py # 房价测试脚本├── test_cifar10.py # CIFAR10 测试脚本├── main.py # 统一运行入口├── house_data.csv # 房价数据集├── titanic_train_knn.csv # Titanic 数据集├── models/ # 保存训练好的模型├── results/ # 保存 Loss/Acc 曲线└── README.md # 项目说明
plaintext

## 运行方法
### 1. 运行单个任务
```bash
# 训练Titanic
python train_titanic_ann.py
# 测试Titanic
python test_titanic_ann.py

# 训练房价预测
python train_house.py
# 测试房价预测
python test_house.py

# 训练CIFAR10
python train_cifar10.py
# 测试CIFAR10
python test_cifar10.py
2. 一键运行所有任务
bash
运行
python main.py
输出结果
models/ 目录下生成各任务的模型文件（.pkl）
results/ 目录下生成各任务的 Loss/Acc 曲线（.png）
终端输出各任务的训练日志和测试指标（准确率 / MSE）