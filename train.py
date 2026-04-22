import pandas as pd
from src.data_preprocess import preprocess_titanic
from src.SVM import TitanicSVM

# 读取数据
train_df = pd.read_csv('titanic_train_knn.csv')
X_train, y_train = preprocess_titanic(train_df)

# 训练模型
svm = TitanicSVM()
svm.train(X_train, y_train)

# 保存模型
svm.save_model('./models/svm_titanic.pkl')
print("✅ 训练完成，模型已保存到 ./models/")