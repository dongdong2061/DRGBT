import os
import shutil

# set path
txt_file_path = "evaluate_DRGBT603/test_set.txt"  # 存放序列名称的 txt 文件
dataset_root = "/data/wuyingjie/datasets"  # 数据集文件夹的根目录
target_folder = "evaluate_DRGBT603/gt_files"  # 存放复制后的 init.txt 文件的目标目录

# 读取测试集文件中的序列名称
with open(txt_file_path, "r") as file:
    sequence_names = [line.strip() for line in file]

# 确保目标文件夹存在
os.makedirs(target_folder, exist_ok=True)

# 遍历数据集的文件夹，寻找匹配的序列文件夹
for seq_name in sequence_names:
    seq_path = os.path.join(dataset_root, seq_name)  # 序列文件夹路径
    init_file_path = os.path.join(seq_path, "init.txt")  # 需要复制的文件路径

    # 检查该序列文件夹是否存在，并且其中的 init.txt 文件是否存在
    if os.path.exists(seq_path) and os.path.isfile(init_file_path):
        # 目标文件路径：重命名为文件夹名称
        new_file_path = os.path.join(target_folder, f"{seq_name}.txt")
        
        # 复制并重命名文件
        shutil.copy(init_file_path, new_file_path)
        print(f"已复制并重命名: {init_file_path} -> {new_file_path}")
    else:
        print(f"跳过: {seq_name} (文件夹或 init.txt 不存在)")

print("所有文件处理完成！")
