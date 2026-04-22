import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, required=True, help='svm / knn / logistic')
    parser.add_argument('--data', type=str, required=True, help='titanic')
    parser.add_argument('--mode', type=str, default='train')
    args = parser.parse_args()

    print("运行配置：")
    print(f"算法：{args.algo}")
    print(f"数据集：{args.data}")
    print(f"模式：{args.mode}")