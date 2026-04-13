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

        # make directory
        for path in [self.model_dir, self.image_dir, self.val_image_dir, self.val_checkerboard_dir]:
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

    # save model
    def save_model(self, opts, ep, model, mode):
        if mode == 'ing':
            mode_save_path = '%s/%s%s_%d.pth' % (self.model_dir, opts.data_name, mode, ep)
            model.save(mode_save_path)
        if mode == 'last':
            mode_save_path = '%s/%s%s_%d.pth' % (self.model_dir, opts.data_name, mode, ep)
            model.save(mode_save_path)

    def save_named_model(self, filename, model):
        save_path = os.path.join(self.model_dir, filename)
        model.save(save_path)
        return save_path

    def remove_model(self, filename):
        save_path = os.path.join(self.model_dir, filename)
        if os.path.exists(save_path):
            os.remove(save_path)


def generate_checkerboard(opt_img, sar_reg_img, num_checkers=8):
    if len(opt_img.shape) > 2:
        opt_img = opt_img.squeeze()
    if len(sar_reg_img.shape) > 2:
        sar_reg_img = sar_reg_img.squeeze()

    H, W = opt_img.shape
    checker_size_h = max(H // num_checkers, 1)
    checker_size_w = max(W // num_checkers, 1)
    checker_size = min(checker_size_h, checker_size_w)

    checkerboard = np.zeros((H, W, 3), dtype=np.float32)
    opt_img = torch.nan_to_num(opt_img.detach(), nan=0.0, posinf=0.0, neginf=0.0).cpu()
    sar_reg_img = torch.nan_to_num(sar_reg_img.detach(), nan=0.0, posinf=0.0, neginf=0.0).cpu()

    for i in range(0, H, checker_size):
        for j in range(0, W, checker_size):
            h_end = min(i + checker_size, H)
            w_end = min(j + checker_size, W)
            source = opt_img if (i // checker_size + j // checker_size) % 2 == 0 else sar_reg_img
            checkerboard[i:h_end, j:w_end, 0] = source[i:h_end, j:w_end].numpy()
            checkerboard[i:h_end, j:w_end, 1] = source[i:h_end, j:w_end].numpy()
            checkerboard[i:h_end, j:w_end, 2] = source[i:h_end, j:w_end].numpy()

    return checkerboard
