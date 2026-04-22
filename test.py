import pandas as pd
from sklearn.metrics import accuracy_score
from src.data_preprocess import preprocess_titanic
from src.SVM import TitanicSVM

# 读取数据
test_df = pd.read_csv('titanic_test_knn.csv')
X_test, y_test = preprocess_titanic(test_df)

# 加载模型
svm = TitanicSVM()
svm.load_model('./models/svm_titanic.pkl')

# 预测
y_pred = svm.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"📊 测试集准确率: {acc:.4f}")

# 保存结果
with open('./results/accuracy.txt', 'w') as f:
    f.write(f"SVM 准确率: {acc:.4f}\n")