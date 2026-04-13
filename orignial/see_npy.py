import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 解决OpenMP重复初始化问题
import numpy as np

# 替换为你的npy文件路径（选其中一个即可，比如affine1.png.npy）
npy_path = r"E:\dataset\osdataset\512gt\test\affine1.png.npy"

# 读取并打印关键信息
data = np.load(npy_path, allow_pickle=True)
print("1. .npy文件存储的数据类型：", type(data))
print("2. 数据形状（若为数组）：", data.shape if isinstance(data, np.ndarray) else "非数组")
print("3. 数据总元素数：", np.size(data))
print("4. 元素（预览）：", data[:])

# 若数据是元组/列表，打印每个元素的信息
if isinstance(data, (tuple, list)):
    for i, item in enumerate(data):
        print(f"\n第{i+1}个元素：")
        print("   类型：", type(item))
        print("   形状：", item.shape if isinstance(item, np.ndarray) else "非数组")
        print("   元素数：", np.size(item))