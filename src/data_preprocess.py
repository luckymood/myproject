import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_titanic(df, is_train=True):
    # 只保留数值列，自动填充缺失值
    df = df.select_dtypes(include=['number'])
    df = df.fillna(df.median())

    # 标准化，强制转为float
    scaler = StandardScaler()
    df[df.columns] = scaler.fit_transform(df).astype(float)

    if is_train:
        # 自动识别标签列
        if 'Survived' in df.columns:
            y = df['Survived'].astype(int)
            X = df.drop('Survived', axis=1)
        elif '2urvived' in df.columns:
            y = df['2urvived'].astype(int)
            X = df.drop('2urvived', axis=1)
        else:
            raise ValueError("找不到标签列！")
        return X, y
    else:
        return df