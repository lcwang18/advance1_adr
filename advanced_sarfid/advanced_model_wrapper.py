import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia.utils as KU

from advanced_adrnet import Advanced_ADRNet, LossManager
from modules.modules import get_scheduler
from modules.losses import mi_loss


def _masked_l1_mean(pred, target, mask, scale_factor=1.0, eps=1e-6):
    pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
    target = torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
    mask = torch.nan_to_num(mask, nan=0.0, posinf=0.0, neginf=0.0)
    diff = torch.abs(pred - target) * mask
    valid = mask.sum()
    if valid.item() <= eps:
        return diff.new_tensor(0.0), diff.new_tensor(0.0)
    return diff.sum() * scale_factor / (valid + eps), valid / mask.numel()


def _displacement_cmr(error_disp, h, w, thresholds=(1.0, 3.0, 5.0)):
    scale = error_disp.new_tensor([(w - 1) / 2.0, (h - 1) / 2.0]).view(1, 1, 1, 2)
    error_in_pixels = error_disp * scale
    error_norm = torch.norm(error_in_pixels, dim=-1)
    return tuple((error_norm < thr).float().mean() for thr in thresholds)


def _local_ncc_score(x, y, win=9, eps=1e-6):
    b, c, _, _ = x.shape
    pad = win // 2
    weight = torch.ones((c, 1, win, win), device=x.device, dtype=x.dtype)

    x2 = x * x
    y2 = y * y
    xy = x * y

    x_sum = F.conv2d(x, weight, padding=pad, groups=c)
    y_sum = F.conv2d(y, weight, padding=pad, groups=c)
    x2_sum = F.conv2d(x2, weight, padding=pad, groups=c)
    y2_sum = F.conv2d(y2, weight, padding=pad, groups=c)
    xy_sum = F.conv2d(xy, weight, padding=pad, groups=c)

    win_size = float(win * win)
    x_mean = x_sum / win_size
    y_mean = y_sum / win_size

    cross = xy_sum - y_mean * x_sum - x_mean * y_sum + x_mean * y_mean * win_size
    x_var = x2_sum - 2.0 * x_mean * x_sum + x_mean * x_mean * win_size
    y_var = y2_sum - 2.0 * y_mean * y_sum + y_mean * y_mean * win_size

    ncc = (cross * cross) / (x_var * y_var + eps)
    return ncc.mean()


def _mutual_information_score(x, y, bins=64, eps=1e-8):
    """
    Histogram-based MI for evaluation (no gradient needed).
    x, y: [B, C, H, W], expected in [0, 1].
    """
    b = x.shape[0]
    x = x.detach().clamp(0.0, 1.0)
    y = y.detach().clamp(0.0, 1.0)
    mi_values = []
    edges = torch.linspace(0.0, 1.0, steps=bins + 1, device=x.device, dtype=x.dtype)

    for bi in range(b):
        xa = x[bi].reshape(-1)
        ya = y[bi].reshape(-1)

        x_bin = torch.bucketize(xa, edges, right=False) - 1
        y_bin = torch.bucketize(ya, edges, right=False) - 1
        x_bin = x_bin.clamp(0, bins - 1)
        y_bin = y_bin.clamp(0, bins - 1)

        joint_idx = x_bin * bins + y_bin
        joint_hist = torch.bincount(joint_idx, minlength=bins * bins).float().reshape(bins, bins)
        joint_prob = joint_hist / (joint_hist.sum() + eps)

        px = joint_prob.sum(dim=1, keepdim=True)
        py = joint_prob.sum(dim=0, keepdim=True)
        denom = px * py
        mi = torch.sum(joint_prob * torch.log((joint_prob + eps) / (denom + eps)))
        mi_values.append(mi)

    return torch.stack(mi_values).mean()


class AdvancedADRNetWrapper(nn.Module):
    """Adapter that exposes the same training/test interface as ADRNet."""

    def __init__(self, config=None):
        super().__init__()
        self.config = config

        self.net = Advanced_ADRNet(
            in_channels=1,
            loftr_feat_dim=getattr(config, "adv_loftr_feat_dim", 128),
            loftr_blocks=getattr(config, "adv_loftr_blocks", 2),
            loftr_heads=getattr(config, "adv_loftr_heads", 8),
            unet_base_channels=getattr(config, "adv_unet_base_channels", 32),
        )
        self.loss_manager = LossManager(
            # SAR-fidelity-oriented default:
            # Local NCC is monitored but does not drive optimization unless explicitly enabled.
            w_sim=getattr(config, "adv_w_sim", 0.0),
            w_evidential=getattr(config, "adv_w_evidential", 0.1),
            w_gcl=getattr(config, "adv_w_gcl", 0.05),
            ncc_window=getattr(config, "adv_ncc_window", 9),
        )
        self.l1_loss = nn.L1Loss()
        self.mi_loss = mi_loss()
        self.eps = 1e-6

        lr = float(getattr(config, "lr", 1e-4))
        self.ADV_opt = torch.optim.Adam(self.net.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=1e-5)
        # Compatibility with existing training logger that reads model.RES_opt.
        self.RES_opt = self.ADV_opt
        self.UN_opt = self.ADV_opt
        self.re_scale = float(getattr(config, "re_scale", 255.0))
        self.present_metric = str(getattr(config, "present_metric", "ncc"))

        self.ADV_sch = None
        self.loss_tp = torch.tensor(0.0)
        self.loss_disp = torch.tensor(0.0)
        self.last_loss_components = {}
        self.image_sar_reg_nointerp = None

    def set_scheduler(self, opts, now_ep=0):
        self.ADV_sch = get_scheduler(self.ADV_opt, opts, now_ep)

    def update_lr(self):
        if self.ADV_sch is not None:
            self.ADV_sch.step()

    @staticmethod
    def _normalize_for_vis(img):
        img_min = img.min()
        img_max = img.max()
        if (img_max - img_min) > 1e-6:
            return (img - img_min) / (img_max - img_min)
        return img

    @staticmethod
    def _stn_affine(img, theta, padding_mode="border", mode="bilinear"):
        grid = F.affine_grid(theta, img.size(), align_corners=True)
        return F.grid_sample(img, grid, mode=mode, padding_mode=padding_mode, align_corners=True)

    @staticmethod
    def _warp_dvf(img, dvf, padding_mode="border", mode="bilinear"):
        b, _, h, w = img.shape
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h, device=img.device, dtype=img.dtype),
            torch.linspace(-1.0, 1.0, w, device=img.device, dtype=img.dtype),
            indexing="ij",
        )
        base_grid = torch.stack([xx, yy], dim=-1).unsqueeze(0).repeat(b, 1, 1, 1)
        final_grid = base_grid + dvf.permute(0, 2, 3, 1)
        return F.grid_sample(
            img,
            final_grid,
            mode=mode,
            padding_mode=padding_mode,
            align_corners=True,
        )

    @staticmethod
    def _resize_gt_map(gt_disp, target_h, target_w):
        if gt_disp.shape[1] == target_h and gt_disp.shape[2] == target_w:
            return gt_disp
        return F.interpolate(
            gt_disp.permute(0, 3, 1, 2), size=(target_h, target_w), mode="bilinear", align_corners=True
        ).permute(0, 2, 3, 1)

    @staticmethod
    def _batch_four_point_rmse(tp_pre, tp_gt, image_size):
        tp_gt = tp_gt.to(device=tp_pre.device, dtype=tp_pre.dtype)
        # Corners in normalized coordinates used by affine_grid.
        corners = tp_pre.new_tensor(
            [[-1.0, -1.0, 1.0], [-1.0, 1.0, 1.0], [1.0, -1.0, 1.0], [1.0, 1.0, 1.0]]
        )  # [4,3]
        corners = corners.t().unsqueeze(0).repeat(tp_pre.shape[0], 1, 1)  # [B,3,4]
        pre_xy = torch.bmm(tp_pre, corners).transpose(1, 2)  # [B,4,2]
        gt_xy = torch.bmm(tp_gt, corners).transpose(1, 2)    # [B,4,2]
        pixel_scale = (float(image_size) - 1.0) / 2.0
        dist = torch.norm((pre_xy - gt_xy) * pixel_scale, dim=-1)  # [B,4]
        return torch.sqrt(torch.mean(dist.pow(2)) + 1e-12)

    @staticmethod
    def _compose_sampling_map(theta, dvf, h, w):
        b = theta.shape[0]
        base_grid = KU.create_meshgrid(h, w, device=theta.device, dtype=theta.dtype).repeat(b, 1, 1, 1)
        affine_grid = F.affine_grid(theta, torch.Size([b, 1, h, w]), align_corners=True)  # [B,H,W,2]
        final_grid = base_grid + dvf.permute(0, 2, 3, 1)                                   # [B,H,W,2]
        composed = F.grid_sample(
            affine_grid.permute(0, 3, 1, 2),
            final_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        ).permute(0, 2, 3, 1)
        return composed

    def _sar_fidelity_loss(self, pred, target, mask, eps=1e-6):
        # Old-style equivalent: 0.1 * (L1 + L2) + 0.9 * gradient consistency, all masked.
        pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
        target = torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
        mask = torch.nan_to_num(mask, nan=0.0, posinf=0.0, neginf=0.0)

        valid = mask.sum()
        if valid.item() <= eps:
            return pred.new_tensor(0.0)

        l1 = (torch.abs(pred - target) * mask).sum() / (valid + eps)
        l2 = ((pred - target).pow(2) * mask).sum() / (valid + eps)

        dx_pred = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        dx_tgt = target[:, :, :, 1:] - target[:, :, :, :-1]
        dx_mask = mask[:, :, :, 1:] * mask[:, :, :, :-1]
        dx_valid = dx_mask.sum()
        if dx_valid.item() > eps:
            grad_x = (torch.abs(dx_pred - dx_tgt) * dx_mask).sum() / (dx_valid + eps)
        else:
            grad_x = pred.new_tensor(0.0)

        dy_pred = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        dy_tgt = target[:, :, 1:, :] - target[:, :, :-1, :]
        dy_mask = mask[:, :, 1:, :] * mask[:, :, :-1, :]
        dy_valid = dy_mask.sum()
        if dy_valid.item() > eps:
            grad_y = (torch.abs(dy_pred - dy_tgt) * dy_mask).sum() / (dy_valid + eps)
        else:
            grad_y = pred.new_tensor(0.0)

        grad_loss = 0.5 * (grad_x + grad_y)
        return 0.1 * (l1 + l2) + 0.9 * grad_loss

    @staticmethod
    def _border_suppression(pred, valid_mask):
        invalid = (1.0 - valid_mask).clamp(0.0, 1.0)
        return (pred * invalid).mean()

    def _forward_once(self, image_rgb, image_sar, image_sar_warp):
        outputs = self.net(image_src=image_sar_warp, image_tgt=image_rgb)
        self.theta = outputs["theta"]
        self.sar_reg_aff = outputs["warped_affine"]
        self.image_sar_reg = outputs["warped_final"]
        self.image_rgb_reg = image_rgb
        self.rgb_reg_aff = image_rgb
        return outputs

    def update_RF(self, image_rgb, image_sar, image_rgb_warp, image_sar_warp, gt_tp, gt_disp):
        self.image_rgb = image_rgb
        self.image_sar = image_sar
        self.image_rgb_warp = image_rgb_warp
        self.image_sar_warp = image_sar_warp
        self.gt_tp = gt_tp.reshape(-1, 2, 3).to(device=image_rgb.device, dtype=image_rgb.dtype)
        self.gt_disp = gt_disp.to(device=image_rgb.device, dtype=image_rgb.dtype)

        self.ADV_opt.zero_grad()
        outputs = self._forward_once(image_rgb, image_sar, image_sar_warp)
        _, _, h, w = image_sar_warp.shape

        pred_map = self._compose_sampling_map(self.theta, outputs["dvf"], h, w)
        gt_map = self._resize_gt_map(self.gt_disp, h, w)
        error_map = pred_map - gt_map

        loss_disp = torch.sqrt(torch.mean(error_map.pow(2)) + 1e-12)
        loss_tp = self._batch_four_point_rmse(self.theta, self.gt_tp, image_size=max(h, w))
        loss_pack = self.loss_manager(outputs, image_rgb, pseudo_gt_dvf=None)

        ones = torch.ones_like(image_sar_warp)
        gt_reg_mask = (self._stn_affine(ones, self.gt_tp, padding_mode="zeros", mode="nearest") > 0.5).float()
        pred_aff_mask = self._stn_affine(ones, self.theta, padding_mode="zeros", mode="nearest")
        pred_reg_mask = (self._warp_dvf(pred_aff_mask, outputs["dvf"], padding_mode="zeros", mode="nearest") > 0.5).float()
        valid_mask = gt_reg_mask * pred_reg_mask

        loss_sar_fid = self._sar_fidelity_loss(self.image_sar_reg, image_sar, valid_mask, eps=self.eps)
        loss_sar_border = self._border_suppression(self.image_sar_reg, valid_mask)
        if valid_mask.sum().item() > 16.0:
            loss_sar_mi = self.mi_loss(
                (self.image_sar_reg * valid_mask).clamp(0.0, 1.0),
                (image_sar * valid_mask).clamp(0.0, 1.0),
            )
        else:
            loss_sar_mi = self.image_sar_reg.new_tensor(0.0)

        total_loss = (
            loss_pack["total_loss"]
            + float(getattr(self.config, "adv_w_tp", 1.0)) * loss_tp
            + float(getattr(self.config, "adv_w_disp", 1.0)) * loss_disp
            + float(getattr(self.config, "adv_w_sar_fid", 10.0)) * loss_sar_fid
            + float(getattr(self.config, "adv_w_sar_border", 1.0)) * loss_sar_border
            + float(getattr(self.config, "adv_w_sar_mi", 0.0)) * loss_sar_mi
        )
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
        self.ADV_opt.step()

        self.loss_tp = loss_tp.detach()
        self.loss_disp = loss_disp.detach()
        self.last_loss_components = {
            "L_total": float(total_loss.detach().cpu().item()),
            "L_sim": float(loss_pack["L_sim"].detach().cpu().item()),
            "L_evidential": float(loss_pack["L_evidential"].detach().cpu().item()),
            "L_GCL": float(loss_pack["L_GCL"].detach().cpu().item()),
            "L_tp": float(loss_tp.detach().cpu().item()),
            "L_disp": float(loss_disp.detach().cpu().item()),
            "L_sar_fid": float(loss_sar_fid.detach().cpu().item()),
            "L_sar_border": float(loss_sar_border.detach().cpu().item()),
            "L_sar_mi": float(loss_sar_mi.detach().cpu().item()),
        }

    def test_forward(self, image_rgb, image_sar, image_rgb_warp, image_sar_warp, gt_tp, gt_disp):
        self.image_rgb = image_rgb
        self.image_sar = image_sar
        self.image_rgb_warp = image_rgb_warp
        self.image_sar_warp = image_sar_warp
        self.gt_tp = gt_tp.reshape(-1, 2, 3).to(device=image_rgb.device, dtype=image_rgb.dtype)
        self.gt_disp = gt_disp.to(device=image_rgb.device, dtype=image_rgb.dtype)

        outputs = self._forward_once(image_rgb, image_sar, image_sar_warp)
        _, _, h, w = image_sar_warp.shape

        pred_map = self._compose_sampling_map(self.theta, outputs["dvf"], h, w)
        gt_map = self._resize_gt_map(self.gt_disp, h, w)
        error_map = pred_map - gt_map

        cor_rmse = self._batch_four_point_rmse(self.theta, self.gt_tp, image_size=max(h, w))
        disp_rmse = torch.sqrt(torch.mean(error_map.pow(2)) + 1e-12)

        sar_re = self.l1_loss(self.image_sar_reg * self.re_scale, image_sar * self.re_scale)

        ones = torch.ones_like(image_sar_warp)
        gt_reg_mask = (self._stn_affine(ones, self.gt_tp, padding_mode="zeros", mode="nearest") > 0.5).float()
        pred_aff_mask = self._stn_affine(ones, self.theta, padding_mode="zeros", mode="nearest")
        pred_reg_mask = (self._warp_dvf(pred_aff_mask, outputs["dvf"], padding_mode="zeros", mode="nearest") > 0.5).float()
        valid_mask = gt_reg_mask * pred_reg_mask
        sar_re_masked, valid_ratio = _masked_l1_mean(
            self.image_sar_reg, image_sar, valid_mask, scale_factor=self.re_scale
        )

        cmr_1, cmr_3, cmr_5 = _displacement_cmr(error_map, h, w)

        # no-interpolation (nearest) result for val single-image visualization
        sar_aff_nointerp = self._stn_affine(image_sar_warp, self.theta, padding_mode="zeros", mode="nearest")
        self.image_sar_reg_nointerp = self._warp_dvf(
            sar_aff_nointerp, outputs["dvf"], padding_mode="zeros", mode="nearest"
        )

        if self.present_metric == "ncc":
            present_ncc = _local_ncc_score(self.image_sar_reg, image_rgb, win=int(getattr(self.config, "adv_ncc_window", 9)))
        else:
            present_ncc = _local_ncc_score(self.image_sar_reg, image_rgb, win=int(getattr(self.config, "adv_ncc_window", 9)))
        present_mi = _mutual_information_score(
            self.image_sar_reg,
            image_rgb,
            bins=int(getattr(self.config, "present_mi_bins", 64)),
        )
        return (
            self.image_sar_reg,
            cor_rmse,
            disp_rmse,
            sar_re,
            sar_re_masked,
            valid_ratio,
            cmr_1,
            cmr_3,
            cmr_5,
            valid_mask,
            present_ncc,
            present_mi,
        )

    def save(self, filename, epoch=None):
        state = {
            "ADV_NET": self.net.state_dict(),
            "ADV_OPT": self.ADV_opt.state_dict(),
            "config": vars(self.config) if self.config is not None else {},
        }
        if epoch is not None:
            state["epoch"] = int(epoch)
        torch.save(state, filename)

    def load(self, filename, map_location="cpu", strict=True, load_optimizer=True):
        checkpoint = torch.load(filename, map_location=map_location)
        self.net.load_state_dict(checkpoint["ADV_NET"], strict=strict)
        if load_optimizer and "ADV_OPT" in checkpoint:
            self.ADV_opt.load_state_dict(checkpoint["ADV_OPT"])
        return checkpoint

    def assemble_outputs(self):
        images = [
            self._normalize_for_vis(self.image_sar)[0:1],
            self._normalize_for_vis(self.image_sar_warp)[0:1],
            self._normalize_for_vis(self.sar_reg_aff)[0:1],
            self._normalize_for_vis(self.image_sar_reg)[0:1],
            self._normalize_for_vis(self.image_rgb)[0:1],
            self._normalize_for_vis(self.image_rgb_warp)[0:1],
            self._normalize_for_vis(self.rgb_reg_aff)[0:1],
            self._normalize_for_vis(self.image_rgb_reg)[0:1],
        ]
        h, w = images[0].shape[2], images[0].shape[3]
        resized = [F.interpolate(img, size=(h, w), mode="bilinear", align_corners=False) for img in images]
        row1 = torch.cat(resized[0:4], dim=3)
        row2 = torch.cat(resized[4:8], dim=3)
        return torch.cat([row1, row2], dim=2)
