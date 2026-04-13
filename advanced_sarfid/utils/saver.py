import os
import cv2
import numpy as np
import torch
import torchvision


class Saver():
    def __init__(self, opts):
        self.model_dir = os.path.join(opts.model_dir, opts.data_name)
        self.image_dir = os.path.join(opts.train_img_dir, opts.data_name, 'train', 'image')
        self.val_image_dir = os.path.join(opts.train_img_dir, opts.data_name, 'val', 'image')
        self.val_checkerboard_dir = os.path.join(opts.train_img_dir, opts.data_name, 'val', 'checkerboard')
        self.val_checkerboard_coarse_dir = os.path.join(opts.train_img_dir, opts.data_name, 'val', 'checkerboard_coarse')
        self.val_fine_sar_raw_dir = os.path.join(opts.train_img_dir, opts.data_name, 'val', 'fine_sar_raw')
        self.val_fine_sar_interp_dir = os.path.join(opts.train_img_dir, opts.data_name, 'val', 'fine_sar_interp')

        # make directory
        for path in [self.model_dir, self.image_dir, self.val_image_dir, self.val_checkerboard_dir, self.val_checkerboard_coarse_dir, self.val_fine_sar_raw_dir, self.val_fine_sar_interp_dir]:
            if not os.path.exists(path):
                os.makedirs(path)

    # save result images
    def write_img(self, ep, model):
        assembled_images = model.assemble_outputs()
        img_filename = '%s/train_%05d.jpg' % (self.image_dir, ep)
        torchvision.utils.save_image(assembled_images, img_filename, nrow=1)
        if ep == -1:
            assembled_images = model.assemble_outputs()
            img_filename = '%s/train_last.jpg' % (self.image_dir, ep)
            torchvision.utils.save_image(assembled_images, img_filename, nrow=1)

    def write_val_img(self, ep, idx, model):
        assembled_images = model.assemble_outputs()
        img_filename = '%s/val_ep%05d_idx%03d.jpg' % (self.val_image_dir, ep, idx)
        torchvision.utils.save_image(assembled_images, img_filename, nrow=1)

    def write_val_checkerboard(self, ep, idx, opt_img, sar_reg_img, num_checkers=6):
        checkerboard = generate_checkerboard(opt_img, sar_reg_img, num_checkers=num_checkers)
        img_filename = '%s/val_ep%05d_idx%03d_checkerboard.png' % (self.val_checkerboard_dir, ep, idx)
        cv2.imwrite(img_filename, (checkerboard * 255).astype(np.uint8))

    def write_val_checkerboard_coarse(self, ep, idx, opt_img, sar_reg_img, num_checkers=6):
        checkerboard = generate_checkerboard(opt_img, sar_reg_img, num_checkers=num_checkers)
        img_filename = '%s/val_ep%05d_idx%03d_checkerboard_coarse.png' % (self.val_checkerboard_coarse_dir, ep, idx)
        cv2.imwrite(img_filename, (checkerboard * 255).astype(np.uint8))

    def write_val_fine_sar_raw(self, ep, idx, sar_reg_img):
        sar_gray_uint8 = _to_gray_uint8(sar_reg_img)
        img_filename = '%s/val_ep%05d_idx%03d_sar_fine_raw.png' % (self.val_fine_sar_raw_dir, ep, idx)
        cv2.imwrite(img_filename, sar_gray_uint8)

    def write_val_fine_sar_interp(self, ep, idx, sar_reg_img):
        sar_gray_uint8 = _to_gray_uint8(sar_reg_img)
        img_filename = '%s/val_ep%05d_idx%03d_sar_fine_interp.png' % (self.val_fine_sar_interp_dir, ep, idx)
        cv2.imwrite(img_filename, sar_gray_uint8)

    # save model
    def save_model(self, opts, ep, model, mode):
        if mode == 'ing':
            mode_save_path = '%s/%s%s_%d.pth' % (self.model_dir, opts.data_name, mode, ep)
            model.save(mode_save_path)
        if mode == 'last':
            mode_save_path = '%s/%s%s_%d.pth' % (self.model_dir, opts.data_name, mode, ep)
            model.save(mode_save_path)

    def save_named_model(self, filename, model, epoch=None):
        save_path = os.path.join(self.model_dir, filename)
        try:
            model.save(save_path, epoch=epoch)
        except TypeError:
            model.save(save_path)
        return save_path

    def remove_model(self, filename):
        save_path = os.path.join(self.model_dir, filename)
        if os.path.exists(save_path):
            os.remove(save_path)


def _to_color_float01_opt(opt_img):
    if isinstance(opt_img, torch.Tensor):
        tensor = torch.nan_to_num(opt_img.detach(), nan=0.0, posinf=0.0, neginf=0.0).cpu()
        if tensor.dim() == 3 and tensor.shape[0] in (1, 3):
            tensor = tensor.permute(1, 2, 0)
        elif tensor.dim() == 2:
            tensor = tensor.unsqueeze(-1)
        arr = tensor.numpy().astype(np.float32)
        if arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
        arr = np.clip(arr, 0.0, 1.0)
        return arr

    arr = np.asarray(opt_img)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    arr = arr.astype(np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def _to_color_float01_sar(sar_reg_img, target_hw):
    tensor = torch.nan_to_num(sar_reg_img.detach(), nan=0.0, posinf=0.0, neginf=0.0).cpu()
    if tensor.dim() == 3 and tensor.shape[0] in (1, 3):
        tensor = tensor.permute(1, 2, 0)
    elif tensor.dim() == 2:
        tensor = tensor.unsqueeze(-1)
    arr = tensor.numpy().astype(np.float32)
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    arr = np.clip(arr, 0.0, 1.0)

    target_h, target_w = target_hw
    if arr.shape[0] != target_h or arr.shape[1] != target_w:
        arr = cv2.resize(arr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    return arr


def _to_gray_uint8(img):
    if isinstance(img, torch.Tensor):
        tensor = torch.nan_to_num(img.detach(), nan=0.0, posinf=0.0, neginf=0.0).cpu()
        if tensor.dim() == 3 and tensor.shape[0] in (1, 3):
            tensor = tensor.permute(1, 2, 0)
        elif tensor.dim() == 2:
            tensor = tensor.unsqueeze(-1)
        arr = tensor.numpy().astype(np.float32)
    else:
        arr = np.asarray(img).astype(np.float32)
        if arr.ndim == 2:
            arr = arr[:, :, None]

    if arr.ndim == 3 and arr.shape[2] >= 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)[:, :, None]
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    if arr.max() > 1.0:
        arr = arr / 255.0
    arr = np.clip(arr, 0.0, 1.0)
    return (arr * 255.0).round().astype(np.uint8)


def generate_checkerboard(opt_img, sar_reg_img, num_checkers=8):
    opt_color = _to_color_float01_opt(opt_img)
    h, w = opt_color.shape[:2]
    sar_color = _to_color_float01_sar(sar_reg_img, (h, w))

    checker_size_h = max(h // num_checkers, 1)
    checker_size_w = max(w // num_checkers, 1)
    checker_size = min(checker_size_h, checker_size_w)

    checkerboard = np.zeros((h, w, 3), dtype=np.float32)
    for i in range(0, h, checker_size):
        for j in range(0, w, checker_size):
            h_end = min(i + checker_size, h)
            w_end = min(j + checker_size, w)
            source = opt_color if (i // checker_size + j // checker_size) % 2 == 0 else sar_color
            checkerboard[i:h_end, j:w_end] = source[i:h_end, j:w_end]

    return checkerboard
