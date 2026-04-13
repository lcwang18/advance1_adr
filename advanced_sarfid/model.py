from modules.losses import *
import kornia
import kornia.utils as KU
from modules.modules import resnet, unet, get_scheduler, gaussian_weights_init, SpatialTransformer
import cv2


def normalize_image(img):
    """将图像归一化到[0, 1]范围
    
    Args:
        img: 输入图像张量，形状为(batch, channel, height, width)
    
    Returns:
        normalized_img: 归一化后的图像张量，像素值范围为[0, 1]
    """
    # 计算图像的最小值和最大值
    img_min = img.min()
    img_max = img.max()
    
    # 避免除零错误
    if img_max - img_min < 1e-6:
        return img * 0 + 0.5  # 如果图像几乎没有变化，返回灰色图像
    
    # 归一化到[0, 1]范围
    normalized_img = (img - img_min) / (img_max - img_min)
    
    return normalized_img
def four_point_RMSE_loss(tp_pre, tp_gt, image_size=512):
    """计算四个角点的RMSE损失
    
    Args:
        tp_pre: 预测的变换参数 (2, 3) 或 (batch_size, 6)
        tp_gt: 真实的变换参数 (2, 3) 或 (batch_size, 6)
        image_size: 图像尺寸，默认为512（适配当前数据）
    
    Returns:
        rmse: 四个角点的平均RMSE误差
    """
    # 确保输入是(2, 3)形状
    tp_pre = np.array(tp_pre.cpu()).reshape(2, 3)
    tp_gt = np.array(tp_gt.cpu()).reshape(2, 3)

    # 定义四个角点坐标
    max_coord = image_size - 1
    corners = np.array([
        [0, 0, 1],
        [0, max_coord, 1],
        [max_coord, 0, 1],
        [max_coord, max_coord, 1]
    ])  # (4, 3)

    # 应用仿射变换计算变换后的角点
    # 仿射变换: new_x = a*x + b*y + c, new_y = d*x + e*y + f
    # 其中tp = [[a, b, c], [d, e, f]]
    corners_pre = np.dot(tp_pre, corners.T).T  # (4, 2)
    corners_gt = np.dot(tp_gt, corners.T).T  # (4, 2)

    # 计算每个角点的欧氏距离
    distances = np.sqrt(np.sum((corners_pre - corners_gt) ** 2, axis=1))

    # 计算平均RMSE
    rmse = np.mean(distances)

    return rmse


def generate_mask(img):
    mask = torch.gt(img, 1)
    mask = torch.tensor(mask, dtype=torch.float32)
    return mask

def affine_to_flow(tp, b, h=512, w=512):
    """将仿射变换参数转换为光流场
    
    Args:
        tp: 仿射变换参数 (batch_size, 6) 或 (batch_size, 2, 3)
        b: batch size
        h: 图像高度，默认512
        w: 图像宽度，默认512
    
    Returns:
        flow: 变换后的点坐标
        flow_grid: 相对于网格的流动
    """
    tp = tp.reshape(-1, 2, 3)
    a = torch.Tensor([[[0, 0, 1]]]).cuda().repeat(b, 1, 1)
    tp = torch.cat((tp, a), dim=1)
    grid = KU.create_meshgrid(h, w).cuda().repeat(b, 1, 1, 1)  # 使用动态尺寸创建网格
    flow = kornia.geometry.linalg.transform_points(tp, grid)
    return flow, flow-grid


def normalize_image(x):
    return x[:, 0:1, :, :]


def border_suppression(img, mask):
    return (img * (1 - mask)).mean()


def STN(img, pre_tps):
    # 确保输入是Float类型
    img = img.float()
    pre_tps = pre_tps.float()
    aff_mat = pre_tps.reshape(-1, 2, 3)
    img_grid = F.affine_grid(aff_mat, img.size())
    img_reg = F.grid_sample(img, img_grid)
    return img_reg


def displacement_cmr(error_disp, h, w, thresholds=(1.0, 3.0, 5.0)):
    """统计位移误差在像素单位下低于多个阈值的比例。"""
    scale = error_disp.new_tensor([(w - 1) / 2.0, (h - 1) / 2.0]).view(1, 1, 1, 2)
    error_in_pixels = error_disp * scale
    error_norm = torch.norm(error_in_pixels, dim=-1)
    return tuple((error_norm < thr).float().mean() for thr in thresholds)


def masked_l1_mean(pred, target, mask, scale_factor=1.0, eps=1e-6):
    pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
    target = torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
    mask = torch.nan_to_num(mask, nan=0.0, posinf=0.0, neginf=0.0)
    diff = torch.abs(pred - target) * mask
    valid = mask.sum()
    if valid.item() <= eps:
        return diff.new_tensor(0.0), diff.new_tensor(0.0)
    return diff.sum() * scale_factor / (valid + eps), valid / mask.numel()



class ADRNet(nn.Module):
    def __init__(self, config=None):
        super(ADRNet, self).__init__()

        lr = 0.0001

        self.RES = resnet()
        self.ST = SpatialTransformer(256, 256, True)
        self.UN = unet()

        self.RES_opt = torch.optim.Adam(self.RES.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.00001)
        self.UN_opt = torch.optim.Adam(self.UN.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.00001)

        self.gradient_loss = gradient_loss()
        self.ncc_loss = ncc_loss()
        self.mi_loss = mi_loss()
        self.l1_loss = nn.L1Loss()
        self.deformation_1 = {}
        self.deformation_2 = {}
        self.border_mask = torch.zeros([1, 1, 256, 256])
        self.border_mask[:, :, 10:-10, 10:-10] = 1
        self.AP = nn.AvgPool2d(5, stride=1, padding=2)
        self.initialize()

    def initialize(self):
        self.RES.apply(gaussian_weights_init)
        self.UN.apply(gaussian_weights_init)

    def set_scheduler(self, opts, now_ep=0):
        self.RES_sch = get_scheduler(self.RES_opt, opts, now_ep)
        self.UN_sch = get_scheduler(self.UN_opt, opts, now_ep)





    def forward(self, vi_tensor, ir_tensor, vi_warp_tensor, ir_warp_tensor):
        b, c, h, w = vi_tensor.shape
        sar_stack = torch.cat([ir_warp_tensor, ir_tensor])
        opt_stack = torch.cat([vi_tensor, vi_warp_tensor])
        sw2o, _, _, ow2s = self.RES(sar_stack, opt_stack)

        # 获取当前图像尺寸
        _, _, h, w = ir_warp_tensor.shape
        _, sw2o_disp = affine_to_flow(sw2o, b, h, w)
        _, ow2s_disp = affine_to_flow(ow2s, b, h, w)

        opt_reg_aff = STN(vi_warp_tensor, ow2s)
        sar_reg_aff = STN(ir_warp_tensor, sw2o)

        # 保存用于显示的变量
        self.image_rgb = vi_tensor
        self.image_sar = ir_tensor
        self.image_rgb_warp = vi_warp_tensor
        self.image_sar_warp = ir_warp_tensor
        self.rgb_reg_aff = opt_reg_aff
        self.sar_reg_aff = sar_reg_aff
        self.sw2r = sw2o
        self.rw2s = ow2s

        sar_stack_ = torch.cat([sar_reg_aff, ir_tensor])
        opt_stack_ = torch.cat([vi_tensor, opt_reg_aff])
        _, _, disp, _ = self.UN(sar_stack_, opt_stack_)

        # 调整UNet输出的尺寸以匹配位移场尺寸
        sar2opt_disp = F.interpolate(disp['sar2rgb'], size=sw2o_disp.shape[1:3], mode='bilinear', align_corners=False)
        opt2sar_disp = F.interpolate(disp['rgb2sar'], size=ow2s_disp.shape[1:3], mode='bilinear', align_corners=False)

        pre_disp1 = sw2o_disp + sar2opt_disp.permute(0,2,3,1)
        pre_disp2 = ow2s_disp + opt2sar_disp.permute(0,2,3,1)

        img_stack = torch.cat([ir_warp_tensor, vi_warp_tensor])
        disp_stack = torch.cat([pre_disp1, pre_disp2])
        img_reg, _ = self.ST(img_stack, disp_stack)

        image_sar_reg, image_opt_reg = torch.split(img_reg, b, dim=0)
        
        # 保存用于显示的最终配准结果
        self.image_sar_reg = image_sar_reg
        self.image_rgb_reg = image_opt_reg
        self.disp = disp
        self.u, self.v = None, None  # 这些在UNet中可能没有使用

        return image_sar_reg

    def test_forward(self, image_rgb, image_sar, image_rgb_warp, image_sar_warp, gt_tp, gt_disp):
        # 保存输入参数
        self.image_rgb = image_rgb
        self.image_sar = image_sar
        self.image_rgb_warp = image_rgb_warp
        self.image_sar_warp = image_sar_warp
        self.gt_tp = gt_tp
        self.gt_disp = gt_disp
        
        b, c, h, w = image_sar.shape
        
        # 计算真实配准结果
        sar_gt_reg = STN(image_sar_warp, gt_tp)
        
        # 执行前向传播
        image_sar_reg = self.forward(image_rgb, image_sar, image_rgb_warp, image_sar_warp)
        
        # 计算各种损失
        # 1. 变换参数RMSE损失（四点RMSE）
        loss_pts = four_point_RMSE_loss(self.sw2r, gt_tp)  # 每个角的位移差
        
        # 2. 位移场RMSE损失
        grid = KU.create_meshgrid(h, w).cuda()
        
        # 获取RES网络的位移场
        _, sw2r_disp = affine_to_flow(self.sw2r, b, h, w)
        
        # 调整UNet输出的尺寸以匹配位移场尺寸
        sar2opt_disp = F.interpolate(self.disp['sar2rgb'], size=sw2r_disp.shape[1:3], mode='bilinear', align_corners=False)
        
        # 计算最终位移场
        pre_disp_sw2r = sw2r_disp + sar2opt_disp.permute(0,2,3,1)
        
        # 确保gt_disp和pre_disp_sw2r尺寸匹配
        if self.gt_disp.shape != pre_disp_sw2r.shape:
            gt_disp_adjusted = F.interpolate(self.gt_disp.permute(0,3,1,2), size=pre_disp_sw2r.shape[1:3], mode='bilinear', align_corners=False).permute(0,2,3,1)
            grid_adjusted = F.interpolate(grid.permute(0,3,1,2), size=pre_disp_sw2r.shape[1:3], mode='bilinear', align_corners=False).permute(0,2,3,1)
        else:
            gt_disp_adjusted = self.gt_disp
            grid_adjusted = grid
        
        disp_error = pre_disp_sw2r - (gt_disp_adjusted - grid_adjusted)
        loss_disp = torch.sum(abs(disp_error).pow(2))/(h*w)
        cmr_1, cmr_3, cmr_5 = displacement_cmr(disp_error, h, w)
        
        # 3. SAR图像自损失（与原始代码保持一致）
        # 确保图像尺寸匹配
        if sar_gt_reg.shape != image_sar_reg.shape:
            sar_gt_reg_aligned = F.interpolate(sar_gt_reg, size=image_sar_reg.shape[2:], mode='bilinear', align_corners=False)
        else:
            sar_gt_reg_aligned = sar_gt_reg

        ones_mask = torch.ones_like(image_sar_warp)
        gt_reg_mask = STN(ones_mask, gt_tp)
        pred_reg_mask, _ = self.ST(ones_mask, pre_disp_sw2r)
        gt_reg_mask = (gt_reg_mask > 0.5).float()
        pred_reg_mask = (pred_reg_mask > 0.5).float()
        valid_mask = gt_reg_mask * pred_reg_mask

        sarf_self_loss = self.l1_loss(sar_gt_reg_aligned * 512, image_sar_reg * 512)
        sarf_self_loss_masked, valid_ratio = masked_l1_mean(
            image_sar_reg,
            sar_gt_reg_aligned,
            valid_mask,
            scale_factor=512.0,
        )

        return image_sar_reg, loss_pts, loss_disp, sarf_self_loss, sarf_self_loss_masked, valid_ratio, cmr_1, cmr_3, cmr_5, valid_mask

    def train_forward(self):
        b = self.image_sar_warp.shape[0]

        sar_stack = torch.cat([self.image_sar_warp, self.image_sar])
        rgb_stack = torch.cat([self.image_rgb, self.image_rgb_warp])
        
        self.sw2r, self.s2rw, self.r2sw, self.rw2s = self.RES(sar_stack, rgb_stack)

        # 获取当前图像尺寸
        b, c, h, w = self.image_sar_warp.shape
        
        # 传递正确的图像尺寸参数给affine_to_flow函数
        _, self.sw2r_disp = affine_to_flow(self.sw2r, b, h, w)
        _, self.rw2s_disp = affine_to_flow(self.rw2s, b, h, w)    # 使用当前图像尺寸

        # 直接在归一化图像上生成mask，移除255缩放因子以适应512输入尺寸
        self.mask = generate_mask(self.image_sar_warp)
        self.mask_true = STN(self.mask, self.gt_tp)


        self.rgb_reg_aff = STN(self.image_rgb_warp, self.rw2s)
        self.sar_reg_aff = STN(self.image_sar_warp, self.sw2r)
     
        sar_stack_ = torch.cat([self.sar_reg_aff, self.image_sar])
        rgb_stack_ = torch.cat([self.image_rgb, self.rgb_reg_aff])

        self.u, self.v, self.disp, self.disp1 = self.UN(sar_stack_, rgb_stack_)  # [8,2,256,256]

        # 调整UNet输出的尺寸以匹配sw2r_disp和rw2s_disp的尺寸
        sar2rgb_disp = F.interpolate(self.disp['sar2rgb'], size=self.sw2r_disp.shape[1:3], mode='bilinear', align_corners=False)
        rgb2sar_disp = F.interpolate(self.disp['rgb2sar'], size=self.rw2s_disp.shape[1:3], mode='bilinear', align_corners=False)
        
        self.pre_disp_sw2r = self.sw2r_disp + sar2rgb_disp.permute(0,2,3,1)
        self.pre_disp_rw2s = self.rw2s_disp + rgb2sar_disp.permute(0,2,3,1)

        img_stack = torch.cat([self.image_sar_warp, self.image_rgb_warp])
        disp_stack = torch.cat([self.pre_disp_sw2r, self.pre_disp_rw2s])
        # 只取SpatialTransformer返回的第一个元素（变换后的图像）
        img_reg_stack, _ = self.ST(img_stack, disp_stack)

        self.image_sar_reg, self.image_rgb_reg = torch.split(img_reg_stack, b, dim=0)


    def update_RF(self, image_rgb, image_sar, image_rgb_warp, image_sar_warp, gt_tp, gt_disp):
        self.image_rgb = image_rgb
        self.image_sar = image_sar
        self.image_rgb_warp = image_rgb_warp
        self.image_sar_warp = image_sar_warp
        self.gt_tp = gt_tp
        self.gt_disp = gt_disp

        self.RES_opt.zero_grad()
        self.UN_opt.zero_grad()

        self.train_forward()

        self.backward_RF()

        nn.utils.clip_grad_norm_(self.RES.parameters(), 5)
        nn.utils.clip_grad_norm_(self.UN.parameters(), 5)

        self.RES_opt.step()
        self.UN_opt.step()

    def img_loss(self, src, tgt, mask=1, weights=None):
        if weights is None:
            weights = [0.1, 0.9]
        
        # 确保tgt尺寸与src匹配
        if tgt.shape != src.shape:
            tgt = F.interpolate(tgt, size=src.shape[2:], mode='bilinear', align_corners=False)
            
        return weights[0] * (l1loss(src, tgt, mask) + l2loss(src, tgt, mask)) + weights[1] * self.gradient_loss(src, tgt,
                                                                                                               mask)

    def weight_filed_loss(self, ref, tgt, disp, disp_gt):
        ref = (ref - ref.mean(dim=[-1, -2], keepdim=True)) / (ref.std(dim=[-1, -2], keepdim=True) + 1e-5)
        tgt = (tgt - tgt.mean(dim=[-1, -2], keepdim=True)) / (tgt.std(dim=[-1, -2], keepdim=True) + 1e-5)
        g_ref = KF.spatial_gradient(ref, order=2).mean(dim=1).abs().sum(dim=1).detach().unsqueeze(1)
        g_tgt = KF.spatial_gradient(tgt, order=2).mean(dim=1).abs().sum(dim=1).detach().unsqueeze(1)
        w = ((g_ref + g_tgt) * 2 + 1) * self.border_mask.to(device=0)
        return (w * (1000 * (disp.permute(0,3,1,2) - disp_gt.permute(0,3,1,2)).abs().clamp(min=1e-2).pow(2))).mean()


    def sym_loss(self, disp1, disp2):
        # 确保disp2尺寸与disp1匹配
        if disp2.shape != disp1.shape:
            # 调整disp2的尺寸以匹配disp1
            disp2 = F.interpolate(disp2.permute(0,3,1,2), size=disp1.shape[1:3], 
                                 mode='bilinear', align_corners=False).permute(0,2,3,1)
        
        # 根据disp1的尺寸动态创建网格
        h, w = disp1.shape[1], disp1.shape[2]
        grid = KU.create_meshgrid(h, w).cuda()
        flow1 = grid + disp1
        flow2 = grid + disp2
        a = F.grid_sample(flow1.permute(0,3,1,2), flow2).permute(0,2,3,1)
        mask = torch.eq(a, 0)
        mask = torch.tensor(~mask, dtype=torch.float32)
        num = mask.permute(0,3,1,2).view(mask.permute(0,3,1,2).size(0), mask.permute(0,3,1,2).size(1) , -1).sum(dim=-1).sum(dim=-1)/2
        grid1 = grid*mask

        loss = (torch.sum(abs(a-grid1).pow(2), dim=[-1,-2,-3])/num).mean().sqrt()
        return loss


    def backward_RF(self):
        b, c, h, w = self.image_rgb.shape
        grid = KU.create_meshgrid(h,w).cuda()
        idx3 = torch.Tensor([[[0,0,1]]]).cuda().repeat(b, 1, 1)
        unit = torch.Tensor([[[1,0,0],[0,1,0],[0,0,1]]]).cuda().repeat(b, 1, 1)

        sw2r = self.sw2r.reshape(-1, 2, 3)
        sw2r_mat = torch.cat((sw2r, idx3), dim=1)
        r2sw = self.r2sw.reshape(-1, 2, 3)
        r2sw_mat = torch.cat((r2sw, idx3), dim=1)  
        e1 = torch.matmul(sw2r_mat, r2sw_mat)

        rw2s = self.rw2s.reshape(-1, 2, 3)
        rw2s_mat = torch.cat((rw2s, idx3), dim=1)    
        s2rw = self.s2rw.reshape(-1, 2, 3)
        s2rw_mat = torch.cat((s2rw, idx3), dim=1)
        e2 = torch.matmul(rw2s_mat, s2rw_mat)
                
        dc1 = torch.sum(abs(e1-unit), dim=[-2,-1]).mean() + torch.sum(abs(e2-unit),dim=[-2,-1]).mean()
  
        ld_loss = self.img_loss(self.image_rgb, self.image_rgb_reg, self.mask_true) + \
                  self.img_loss(self.image_sar, self.image_sar_reg, self.mask_true)
 
        loss_tp = torch.sum(abs(self.sw2r-self.gt_tp.reshape(b, -1))).mean() + torch.sum(abs(self.rw2s-self.gt_tp.reshape(b, -1))).mean()
 
        loss_mi1 = self.mi_loss(self.image_rgb*self.mask_true, self.rgb_reg_aff) + self.mi_loss(self.image_sar*self.mask_true, self.sar_reg_aff)

        loss_reg = 10*loss_tp + loss_mi1 + dc1

    
        # 确保disp和disp1中的位移场尺寸匹配后再计算对称性损失
        # 调整disp1中的张量尺寸以匹配disp中的对应张量
        rgb2sar_disp1 = F.interpolate(self.disp1['rgb2sar'], size=self.disp['sar2rgb'].shape[2:], 
                                    mode='bilinear', align_corners=False)
        sar2rgb_disp1 = F.interpolate(self.disp1['sar2rgb'], size=self.disp['rgb2sar'].shape[2:], 
                                    mode='bilinear', align_corners=False)
        
        dc2 = self.sym_loss(self.disp['sar2rgb'].permute(0,2,3,1), rgb2sar_disp1.permute(0,2,3,1)) + \
              self.sym_loss(self.disp['rgb2sar'].permute(0,2,3,1), sar2rgb_disp1.permute(0,2,3,1))
 
        # 调整grid尺寸以匹配pre_disp_sw2r和pre_disp_rw2s
        grid_adjusted = F.interpolate(grid.permute(0,3,1,2), size=self.pre_disp_sw2r.shape[1:3], mode='bilinear', align_corners=False).permute(0,2,3,1)
        
        # 确保gt_disp和grid_adjusted尺寸匹配
        if self.gt_disp.shape != grid_adjusted.shape:
            gt_disp_adjusted = F.interpolate(self.gt_disp.permute(0,3,1,2), size=grid_adjusted.shape[1:3], mode='bilinear', align_corners=False).permute(0,2,3,1)
        else:
            gt_disp_adjusted = self.gt_disp
        
        loss_disp1 = (torch.sum(abs(self.pre_disp_sw2r - (gt_disp_adjusted-grid_adjusted)).pow(2))/(h*w)).sqrt()
        loss_disp2 = (torch.sum(abs(self.pre_disp_rw2s - (gt_disp_adjusted-grid_adjusted)).pow(2))/(h*w)).sqrt()
        loss_disp = loss_disp1 + loss_disp2

        # 确保图像尺寸匹配后再计算互信息损失
        rgb_reg_aligned = F.interpolate(self.image_rgb_reg, size=self.image_rgb.shape[2:], mode='bilinear', align_corners=False)
        sar_reg_aligned = F.interpolate(self.image_sar_reg, size=self.image_sar.shape[2:], mode='bilinear', align_corners=False)
        
        loss_mi2 = self.mi_loss(self.image_rgb*self.mask_true, rgb_reg_aligned) + self.mi_loss(self.image_sar*self.mask_true, sar_reg_aligned) 
        
        loss_smooth_down2 = smoothloss(self.u)
        loss_smooth_down4 = smoothloss(self.v)
        loss_smooth = loss_smooth_down2 + loss_smooth_down4

        # 确保图像尺寸与mask匹配后再进行边界抑制计算
        mask_true_aligned = F.interpolate(self.mask_true, size=self.image_sar_reg.shape[2:], mode='bilinear', align_corners=False)
        
        # 确保image_rgb_reg与mask_true_aligned尺寸匹配
        if self.image_rgb_reg.shape != mask_true_aligned.shape:
            rgb_reg_aligned_for_re = F.interpolate(self.image_rgb_reg, size=mask_true_aligned.shape[2:], mode='bilinear', align_corners=False)
        else:
            rgb_reg_aligned_for_re = self.image_rgb_reg
        
        loss_re = border_suppression(self.image_sar_reg, mask_true_aligned) + border_suppression(rgb_reg_aligned_for_re, mask_true_aligned) 
           
        loss_d = 10*ld_loss + 10*loss_disp + loss_smooth + loss_re + loss_mi2 + dc2

        loss_total = loss_reg + loss_d

        loss_total.backward()

        self.loss_tp = loss_tp/2
        self.loss_disp = loss_disp/2


    def update_lr(self):

        self.RES_sch.step()
        self.UN_sch.step()

    def save(self, filename, epoch=None):
        state = {

            'RES': self.RES.state_dict(),
            'UN': self.UN.state_dict(),
            'RES_opt': self.RES_opt.state_dict(),
            'UN_opt': self.UN_opt.state_dict(),

        }
        if epoch is not None:
            state['epoch'] = int(epoch)
        torch.save(state, filename)
        return

    def load(self, filename, map_location='cpu', strict=True, load_optimizer=True):
        checkpoint = torch.load(filename, map_location=map_location)
        self.RES.load_state_dict(checkpoint['RES'], strict=strict)
        self.UN.load_state_dict(checkpoint['UN'], strict=strict)

        if load_optimizer:
            if 'RES_opt' in checkpoint:
                self.RES_opt.load_state_dict(checkpoint['RES_opt'])
            if 'UN_opt' in checkpoint:
                self.UN_opt.load_state_dict(checkpoint['UN_opt'])
        return checkpoint

    def assemble_outputs(self):
        # 定义归一化函数（确保正确归一化）
        def normalize_image(img):
            # 确保图像在[0,1]范围内
            img_min = img.min()
            img_max = img.max()
            if img_max - img_min > 1e-6:
                return (img - img_min) / (img_max - img_min)
            else:
                return img
        
        # 获取所有需要显示的图像
        images_ir = normalize_image(self.image_sar).detach()  # 原始SAR图像
        images_vi = normalize_image(self.image_rgb).detach()  # 原始RGB图像
        images_ir_warp = normalize_image(self.image_sar_warp).detach()  # 变换后的SAR图像
        images_vi_warp = normalize_image(self.image_rgb_warp).detach()  # 变换后的RGB图像
        images_ir_Reg = normalize_image(self.sar_reg_aff).detach()  # 仿射配准后的SAR图像
        images_vi_Reg = normalize_image(self.rgb_reg_aff).detach()  # 仿射配准后的RGB图像
        images_ir_fake = normalize_image(self.image_sar_reg).detach()  # 最终配准后的SAR图像
        images_vi_fake = normalize_image(self.image_rgb_reg).detach()  # 最终配准后的RGB图像
        
        # 只使用batch中的第一张图像
        images_ir = images_ir[0:1]
        images_vi = images_vi[0:1]
        images_ir_warp = images_ir_warp[0:1]
        images_vi_warp = images_vi_warp[0:1]
        images_ir_Reg = images_ir_Reg[0:1]
        images_vi_Reg = images_vi_Reg[0:1]
        images_ir_fake = images_ir_fake[0:1]
        images_vi_fake = images_vi_fake[0:1]
        
        # 确定统一的显示尺寸（使用原始图像尺寸）
        h, w = images_ir.shape[2], images_ir.shape[3]
        
        # 调整所有图像尺寸以匹配统一尺寸
        def resize(img):
            return F.interpolate(img, size=(h, w), mode='bilinear', align_corners=False)
        
        # 调整所有图像尺寸
        images = [
            resize(images_ir),       # 1. 原始SAR
            resize(images_ir_warp),  # 2. 变换后的SAR
            resize(images_ir_Reg),   # 3. 仿射配准后的SAR
            resize(images_ir_fake),  # 4. 最终配准后的SAR
            resize(images_vi),       # 5. 原始RGB
            resize(images_vi_warp),  # 6. 变换后的RGB
            resize(images_vi_Reg),   # 7. 仿射配准后的RGB
            resize(images_vi_fake)   # 8. 最终配准后的RGB
        ]
        
        # 检查每张图像的有效性（非全黑）
        valid_images = []
        for i, img in enumerate(images):
            if img.sum() > 0:  # 确保图像不是全黑
                valid_images.append(img)
            else:
                # 如果图像全黑，使用原始图像替代
                valid_images.append(images_ir if i < 4 else images_vi)
        
        # 将所有有效图像在水平方向拼接
        # 分成两行：SAR相关和RGB相关
        if len(valid_images) >= 4:
            row1 = torch.cat(valid_images[0:4], dim=3)  # SAR相关图像
            row2 = torch.cat(valid_images[4:8], dim=3)  # RGB相关图像
            result = torch.cat([row1, row2], dim=2)     # 两行拼接
        else:
            # 如果有效图像不足，只拼接可用图像
            result = torch.cat(valid_images, dim=3)
        
        return result
