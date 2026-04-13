import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 解决OpenMP重复初始化问题
import cv2
import numpy as np
from glob import glob

# ----------------------
# 配置参数
# ----------------------
npy_dir = "E:\dataset\osdataset\\512gt\\val"  # 测试文件夹路径
output_dir = "E:\dataset\osdataset\\512_warped1\\val\\ref"  # 变换后图像的保存路径
src_img_dir = "E:\dataset\osdataset\\512_original\\val\\ref"

# 校验源图像文件夹是否存在
assert os.path.exists(src_img_dir), f"源图像文件夹不存在：{src_img_dir}"

# ----------------------
# 读取并排序.npy文件
# ----------------------
npy_pattern = os.path.join(npy_dir, "affine*.png.npy")
npy_paths = sorted(
    glob(npy_pattern),
    key=lambda x: int(os.path.basename(x).split("affine")[1].split(".png.npy")[0])
)
# 确保npy文件数量为424
assert len(npy_paths) == 238, f"test文件夹中.npy文件数量为{len(npy_paths)}，需为424"

# ----------------------
# 批量处理：仿射变换
# ----------------------
for npy_path in npy_paths:
    # 1. 解析序号，匹配源图像
    basename = os.path.basename(npy_path)
    idx = int(basename.split("affine")[1].split(".png.npy")[0])
    src_img_name = f"opt{idx}.png"  # 源图像命名规则（需匹配实际）
    src_img_path = os.path.join(src_img_dir, src_img_name)

    # 2. 校验并读取源图像
    if not os.path.exists(src_img_path):
        print(f"警告：源图像不存在 → {src_img_path}，跳过")
        continue
    src_img = cv2.imread(src_img_path)
    if src_img is None:
        print(f"警告：无法读取源图像 → {src_img_path}，跳过")
        continue
    h, w = src_img.shape[:2]  # 源图像尺寸（仿射变换默认用原尺寸）

    # 3. 读取仿射变换矩阵（2×3）
    try:
        affine_matrix = np.load(npy_path).astype(np.float32)  # 转float32（cv2要求）
        assert affine_matrix.shape == (2, 3), f"{npy_path} 矩阵形状错误，应为(2,3)，实际{affine_matrix.shape}"
    except Exception as e:
        print(f"警告：读取/解析{npy_path}失败 → {e}，跳过")
        continue

    # 4. 执行仿射变换（核心修改：用warpAffine替代warpPerspective）
    # 参数：源图像、仿射矩阵、输出尺寸（宽, 高）
    warped_img = cv2.warpAffine(src_img, affine_matrix, (512, 512))

    # 5. 保存变换结果
    output_path = os.path.join(output_dir, f"opt{idx}.png")
    cv2.imwrite(output_path, warped_img)
    print(f"已处理 {idx + 1}/2012 → 保存至：{output_path}")

print("\n所有有效图像处理完成！")