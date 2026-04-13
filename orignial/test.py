import torch
from torch.utils.data import DataLoader
from options import TrainOptions
from dataset import TestData, img_save
from model import ADRNet
from utils.saver import Saver
from time import time
from tqdm import tqdm
from modules.losses import *
from build_table import *
import torch.nn as nn
import os
import torch.nn.functional as F
import numpy as np
import glob
import cv2
# from thop import profile


def generate_checkerboard(opt_img, sar_reg_img, num_checkers=8):
    """
    生成棋盘格图，用于直观展示配准效果
    opt_img: 原始光学图像 (1, 1, H, W) 或 (H, W)
    sar_reg_img: 配准后的SAR图像 (1, 1, H, W) 或 (H, W)
    num_checkers: 每行每列的棋盘格数量
    """
    # 确保输入是单通道2D图像
    if len(opt_img.shape) > 2:
        opt_img = opt_img.squeeze()
    if len(sar_reg_img.shape) > 2:
        sar_reg_img = sar_reg_img.squeeze()
    
    H, W = opt_img.shape
    
    # 计算棋盘格大小，确保每行每列有num_checkers个棋盘格
    checker_size_h = H // num_checkers
    checker_size_w = W // num_checkers
    checker_size = min(checker_size_h, checker_size_w)  # 使用较小的尺寸以保持正方形棋盘格
    
    # 创建棋盘格掩码
    checkerboard = np.zeros((H, W, 3), dtype=np.float32)
    
    # 光学图像显示区域 (白色棋盘格)
    for i in range(0, H, checker_size):
        for j in range(0, W, checker_size):
            if (i // checker_size + j // checker_size) % 2 == 0:
                # 确保不超出图像边界
                h_end = min(i + checker_size, H)
                w_end = min(j + checker_size, W)
                checkerboard[i:h_end, j:w_end, 0] = opt_img[i:h_end, j:w_end].cpu().numpy()  # R通道
                checkerboard[i:h_end, j:w_end, 1] = opt_img[i:h_end, j:w_end].cpu().numpy()  # G通道
                checkerboard[i:h_end, j:w_end, 2] = opt_img[i:h_end, j:w_end].cpu().numpy()  # B通道
    
    # SAR图像显示区域 (灰度显示，保持原始颜色)
    for i in range(0, H, checker_size):
        for j in range(0, W, checker_size):
            if (i // checker_size + j // checker_size) % 2 == 1:
                # 确保不超出图像边界
                h_end = min(i + checker_size, H)
                w_end = min(j + checker_size, W)
                checkerboard[i:h_end, j:w_end, 0] = sar_reg_img[i:h_end, j:w_end].cpu().numpy()  # R通道
                checkerboard[i:h_end, j:w_end, 1] = sar_reg_img[i:h_end, j:w_end].cpu().numpy()  # G通道
                checkerboard[i:h_end, j:w_end, 2] = sar_reg_img[i:h_end, j:w_end].cpu().numpy()  # B通道
    
    return checkerboard


def test(opts):

    model = ADRNet(opts)
    model.cuda()
    # 使用本地的90轮次模型权重
    model_weights = torch.load('./model_save3/osdata/osdataing_140.pth')
    model.RES.load_state_dict(model_weights['RES'])
    model.UN.load_state_dict(model_weights['UN'])


    # 使用TestData而不是InferData
    test_dataset = TestData(opts)
    loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    p_bar = tqdm(enumerate(loader), total=len(loader))
    model.eval() 
    start = time()
    total_loss = 0
    l1loss = nn.L1Loss()
    
    # 创建结果保存目录
    results_dir = './results/test_reg'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # 获取测试数据集中的文件名列表
    files_rgb = sorted(glob.glob("%s/ref/*" % opts.test_data_path))
    files_sar = sorted(glob.glob("%s/sen/*" % opts.test_data_path))
    
    for idx, [opt, sar, opt_warp, sar_warp, gt_tp, gt_disp] in p_bar:
        # 手动获取当前批次的文件名
        rgb_filename = os.path.basename(files_rgb[idx % len(files_rgb)])
        sar_filename = os.path.basename(files_sar[idx % len(files_sar)])
     
        vi_tensor = opt.cuda()
        ir_tensor = sar.cuda()
        vi_warp_tensor = opt_warp.cuda()
        ir_warp_tensor = sar_warp.cuda()
        
        # 加载用户提供的真实变换参数
        if isinstance(sar_filename, tuple):
            sar_filename = sar_filename[0]
        # 将sarX.png转换为affineX.png.npy格式
        gt_param_file = os.path.join('./test_gt', sar_filename.replace('sar', 'affine').replace('.png', '.png.npy'))
        # 加载npy文件并转换为torch tensor
        gt_param = np.load(gt_param_file)
        gt_param = torch.tensor(gt_param, dtype=torch.float32).unsqueeze(0).cuda()  # (1, 2, 3)
        
        # 使用真实变换参数生成真实位移场
        b, c, h, w = vi_tensor.shape
        
        # 归一化平移分量（每行最后一个数）
        # 仿射变换矩阵结构：[[a, b, tx], [c, d, ty]]
        # 其中tx和ty是基于像素坐标的平移，需要转换到[-1, 1]范围
        scale_factor_x = 2.0 / (w - 1)
        scale_factor_y = 2.0 / (h - 1)
        
        # 只归一化平移分量
        normalized_gt_param = gt_param.clone()
        normalized_gt_param[:, 0, 2] = gt_param[:, 0, 2] * scale_factor_x
        normalized_gt_param[:, 1, 2] = gt_param[:, 1, 2] * scale_factor_y
        
        # 使用归一化的变换参数生成位移场
        gt_disp = F.affine_grid(normalized_gt_param, (b, c, h, w), align_corners=True)

        b,c,h,w = vi_tensor.shape
    
        with torch.no_grad():
            
            # 执行测试前向传播
            sar_reg_aff = model.forward(vi_tensor, ir_tensor, vi_warp_tensor, ir_warp_tensor)
            
            # 计算位移场损失
            # 获取RES网络预测的变换参数
            sw2o = model.sw2r
            # 将6个元素的向量转换为2x3的矩阵
            sw2o = sw2o.view(b, 2, 3)
            # 将变换参数转换为位移场
            sw2o_disp = F.affine_grid(sw2o, (b, c, h, w), align_corners=True)
            
            # 获取UNet预测的位移场
            sar2opt_disp = F.interpolate(model.disp['sar2rgb'], size=(h, w), mode='bilinear', align_corners=False)
            # 计算最终位移场
            pre_disp_sw2r = sw2o_disp + sar2opt_disp.permute(0,2,3,1)
            
            # 使用用户提供的真实变换参数计算损失
            loss = torch.sum(abs(pre_disp_sw2r - gt_disp).pow(2))/(h*w)
            total_loss = total_loss + loss
            
            # 保存配准结果 - 使用原始SAR文件名
            # 处理可能的元组类型
            if isinstance(sar_filename, tuple):
                sar_filename = sar_filename[0]
            save_filename = sar_filename.replace('.png', '_reg.png')
            img_save(sar_reg_aff, os.path.join(results_dir, save_filename))
            
            # 生成并保存棋盘格图 (每行每列6个棋盘格)
            checkerboard = generate_checkerboard(vi_tensor, sar_reg_aff, num_checkers=6)
            checkerboard_filename = sar_filename.replace('.png', '_checkerboard.png')
            checkerboard_path = os.path.join(results_dir, checkerboard_filename)
            # 转换为0-255范围并保存
            cv2.imwrite(checkerboard_path, (checkerboard * 255).astype(np.uint8))
      
    print(f"Average RMSE: {(total_loss/len(loader)).sqrt()}")
   



if __name__ == '__main__':
    parser = TrainOptions()   # 加载选项信息
    opt = parser.parse()
    # 设置测试集路径
    opt.test_data_path = './test'
    # 设置批大小为1以处理512*512图片
    opt.batch_size = 1
    print('\n--- options load success ---')
    test(opts=opt)
