import os
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
import kornia.utils as KU
import glob
import math
from utils.utils import affine


class TrainData(Dataset):                                      

    def __init__(self, opt):
        super(TrainData, self).__init__()

        self.files_rgb = sorted(glob.glob("%s/ref/*" % opt.train_data_path))
        self.files_sar = sorted(glob.glob("%s/sen/*" % opt.train_data_path))
       
        self.opt = opt
        self.affine = affine

    def __getitem__(self, index):

        item_rgb = imread(self.files_rgb[index % len(self.files_rgb)])
        item_sar = imread(self.files_sar[index % len(self.files_sar)])

        random_numbers = torch.rand(8).numpy() * 2 - 1
        item_rgb_warp, gt_tp, flow = self.affine(random_numbers=random_numbers, imgs=[item_rgb], padding_modes=['zeros'], opt=self.opt)
        item_sar_warp, gt_tp, flow = self.affine(random_numbers=random_numbers, imgs=[item_sar], padding_modes=['zeros'], opt=self.opt)

        item_sar = item_sar.squeeze(0)
        item_rgb = item_rgb.squeeze(0)
           
        return item_rgb, item_sar, item_rgb_warp, item_sar_warp, gt_tp, flow

    def __len__(self):
        return len(self.files_rgb)


class TestData(Dataset):
    def __init__(self, opt, is_validation=False):
    
        self.files_rgb = sorted(glob.glob("%s/ref/*" % opt.test_data_path))
        self.files_sar = sorted(glob.glob("%s/s/*" % opt.test_data_path))

        self.opt = opt
        self.is_validation = is_validation
        self.affine = affine

    def __getitem__(self, index):
  
        # 获取文件名
        rgb_filename = os.path.basename(self.files_rgb[index % len(self.files_rgb)])
        sar_filename = os.path.basename(self.files_sar[index % len(self.files_sar)])
        
        # 读取图像 - 光学图像是原始图像，SAR图像是已经变换的图像
        item_rgb = imread(self.files_rgb[index % len(self.files_rgb)])
        item_sar = imread(self.files_sar[index % len(self.files_sar)])

        if self.is_validation:
            # 训练验证阶段需要应用随机变换
            random_numbers = torch.rand(8).numpy() * 2 - 1
            item_rgb_warp, gt_tp, flow = self.affine(random_numbers=random_numbers, imgs=[item_rgb], padding_modes=['zeros'], opt=self.opt)
            item_sar_warp, gt_tp, flow = self.affine(random_numbers=random_numbers, imgs=[item_sar], padding_modes=['zeros'], opt=self.opt)
        else:
            # 单独测试阶段不需要再应用随机变换，直接返回原始图像对
            item_rgb_warp = item_rgb.clone()
            item_sar_warp = item_sar.clone()
            
            # 创建占位符gt_tp和flow，它们不会被使用
            gt_tp = torch.zeros(2, 3)
            flow = torch.zeros(1, item_rgb.shape[2], item_rgb.shape[3], 2)

        # 确保所有四个图像张量的维度一致
        # item_rgb和item_sar是[1, 1, H, W]格式，需要squeeze(0)变成[1, H, W]
        item_sar = item_sar.squeeze(0)
        item_rgb = item_rgb.squeeze(0)
        
        # 在训练验证阶段，item_rgb_warp和item_sar_warp已经是[1, H, W]格式，不需要再squeeze
        # 在单独测试阶段，item_rgb_warp和item_sar_warp是[1, 1, H, W]格式，需要squeeze(0)变成[1, H, W]
        if not self.is_validation:
            item_rgb_warp = item_rgb_warp.squeeze(0)
            item_sar_warp = item_sar_warp.squeeze(0)
           
        # 根据调用上下文返回不同数量的值
        # 在训练过程中使用时，返回与TrainData类相同的6个值
        # 在测试过程中使用时，返回包含文件名的7个值
        # 可以通过检查调用栈或使用一个标志来确定上下文
        # 这里我们使用一个简单的方法：如果在train.py中调用，会期望6个返回值
        # 我们可以通过检查是否有8个返回值的使用来确定
        # 但为了兼容，我们修改为始终返回6个值，并在test.py中单独处理文件名
        return item_rgb, item_sar, item_rgb_warp, item_sar_warp, gt_tp, flow

    def __len__(self):
        return len(self.files_rgb)



def imread(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    im_ts = (KU.image_to_tensor(img)/255.).float()
    im_ts = im_ts.unsqueeze(0)
    return im_ts


def img_save(img, filename):
    img = img.squeeze().cpu()
    img = KU.tensor_to_image(img*255)
    cv2.imwrite(filename, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))