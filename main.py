import os
import subprocess

def run_task(task_name):
    """运行指定任务的训练+测试"""
    print(f"\n===== 开始运行 {task_name} 任务 =====")
    # 运行训练脚本
    train_script = f"train_{task_name}.py"
    if os.path.exists(train_script):
        subprocess.run(["python", train_script], check=True)
    # 运行测试脚本
    test_script = f"test_{task_name}.py"
    if os.path.exists(test_script):
        subprocess.run(["python", test_script], check=True)
    print(f"===== {task_name} 任务运行完成 =====\n")

if __name__ == "__main__":
    # 依次运行三个任务
    run_task("titanic_ann")
    run_task("house")
    run_task("cifar10")
    print("🎉 所有任务运行完成！")