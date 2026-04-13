import cv2
import numpy as np
import os

# 创建测试图像目录
test_dir = './test'
ref_dir = os.path.join(test_dir, 'ref')
sen_dir = os.path.join(test_dir, 'sen')

if not os.path.exists(ref_dir):
    os.makedirs(ref_dir)
if not os.path.exists(sen_dir):
    os.makedirs(sen_dir)

# 生成简单的测试图像
def create_test_image(size=(256, 256), type='ref'):
    img = np.zeros(size, dtype=np.uint8)
    
    if type == 'ref':
        # 创建一个带有矩形和十字线的图像作为OPT图像
        cv2.rectangle(img, (50, 50), (200, 200), 255, 2)
        cv2.line(img, (128, 50), (128, 200), 255, 1)
        cv2.line(img, (50, 128), (200, 128), 255, 1)
    else:
        # 创建一个带有圆形和网格的图像作为SAR图像
        cv2.circle(img, (128, 128), 75, 255, 2)
        for i in range(0, 256, 20):
            cv2.line(img, (i, 0), (i, 255), 50, 1)
            cv2.line(img, (0, i), (255, i), 50, 1)
    
    return img

# 生成10对测试图像
for i in range(10):
    # 创建OPT图像
    opt_img = create_test_image(type='ref')
    opt_path = os.path.join(ref_dir, f'test_opt_{i}.png')
    cv2.imwrite(opt_path, opt_img)
    
    # 创建SAR图像
    sar_img = create_test_image(type='sen')
    sar_path = os.path.join(sen_dir, f'test_sar_{i}.png')
    cv2.imwrite(sar_path, sar_img)

print(f'Created 10 test image pairs in {test_dir}')
