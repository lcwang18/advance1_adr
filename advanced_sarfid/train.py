#!/usr/bin/python3
import os
import gc
import re
import cv2
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 解决OpenMP重复初始化问题
from torch.utils.data import DataLoader
from dataset import TrainData, TestData, img_save
from options import TrainOptions
from model import ADRNet
from advanced_model_wrapper import AdvancedADRNetWrapper
from utils.saver import Saver
from time import time
from tqdm import tqdm
from modules.losses import *
from build_table import *
from torch.utils.data import DataLoader
import torch


def tensor_stats(tensor, zero_eps=1e-8):
    detached = tensor.detach()
    nan_ratio = torch.isnan(detached).float().mean().item()
    finite_tensor = torch.nan_to_num(detached, nan=0.0, posinf=0.0, neginf=0.0)
    zero_ratio = (finite_tensor.abs() <= zero_eps).float().mean().item()
    min_value = finite_tensor.min().item()
    max_value = finite_tensor.max().item()
    return {
        'nan_ratio': nan_ratio,
        'zero_ratio': zero_ratio,
        'min': min_value,
        'max': max_value,
    }


def _to_float(x):
    if hasattr(x, 'detach'):
        return float(x.detach().cpu().item())
    return float(x)


def _build_fixed_vis_indices(num_samples, k=5):
    if num_samples <= 0:
        return []
    if num_samples <= k:
        return list(range(num_samples))

    raw = [int(round(i * (num_samples - 1) / (k - 1))) for i in range(k)]
    indices = []
    for idx in raw:
        if idx not in indices:
            indices.append(idx)
    if len(indices) < k:
        for idx in range(num_samples):
            if idx not in indices:
                indices.append(idx)
            if len(indices) == k:
                break
    return indices[:k]


def _should_run_validation(ep, n_ep):
    last_ep = max(int(n_ep) - 1, 0)
    return (ep < 5) or (ep % 10 == 0) or (ep > 250) or (ep == last_ep)


def _should_save_val_visualization(ep, n_ep):
    last_ep = max(int(n_ep) - 1, 0)
    if ep == last_ep:
        return True
    if ep < 5:
        return True
    if ep < 300:
        return (ep % 10 == 0)
    return (ep % 50 == 0)


class Train_and_test():
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.best_checkpoints = []
        self.start_ep = 0

        if getattr(self.config, 'model_variant', 'advanced') == 'advanced':
            self.model = AdvancedADRNetWrapper(self.config).cuda()
            print('[Model] Using Advanced_ADRNet (LoFTR + Evidential U-Net).')
        else:
            self.model = ADRNet(self.config).cuda()
            print('[Model] Using baseline ADRNet.')
        self._maybe_resume()
        self.model.set_scheduler(self.config, now_ep=self.start_ep - 1) 
        self.saver = Saver(self.config)

        traindataset = TrainData(config)
        self.train_loader = DataLoader(traindataset, batch_size=config.batch_size, shuffle=True, num_workers=config.nThreads)

        testdataset = TestData(config, is_validation=True)
        self.test_loader = DataLoader(testdataset, batch_size=1, shuffle=False, num_workers=config.nThreads)
        self.val_vis_indices = getattr(testdataset, 'fixed_vis_indices', _build_fixed_vis_indices(len(testdataset), k=5))
        self.val_vis_index_set = set(self.val_vis_indices)
        print(f'[Val] Fixed visualization indices: {self.val_vis_indices}')

    @staticmethod
    def _parse_start_epoch_from_path(path):
        basename = os.path.basename(path)
        match = re.search(r'ep(\d+)', basename)
        if match is None:
            return 0
        return int(match.group(1)) + 1

    def _maybe_resume(self):
        resume_path = str(getattr(self.config, 'resume_path', '')).strip()
        if not resume_path:
            return
        if not os.path.isfile(resume_path):
            raise FileNotFoundError(f'Resume checkpoint not found: {resume_path}')

        strict = bool(int(getattr(self.config, 'resume_strict', 1)))
        load_optimizer = bool(int(getattr(self.config, 'resume_load_optimizer', 1)))
        checkpoint = self.model.load(
            resume_path,
            map_location='cpu',
            strict=strict,
            load_optimizer=load_optimizer,
        )

        if isinstance(checkpoint, dict) and checkpoint.get('epoch', None) is not None:
            self.start_ep = int(checkpoint['epoch']) + 1
        else:
            self.start_ep = self._parse_start_epoch_from_path(resume_path)
        print(f'[Resume] Loaded: {resume_path}')
        print(f'[Resume] strict={strict} | load_optimizer={load_optimizer} | start_ep={self.start_ep}')

    def train_and_test(self):
        lowest_re = float('inf')
        result = {'re': 'N/A', 're_masked': 'N/A', 'valid_ratio': 'N/A', 'cor_rmse': 'N/A', 'avg_disp': 'N/A', 'cmr_1px': 'N/A', 'cmr_3px': 'N/A', 'cmr_5px': 'N/A'}  # Initialize with default values
        if self.start_ep >= self.config.n_ep:
            print(f'[Resume] start_ep={self.start_ep} >= n_ep={self.config.n_ep}, no training steps to run.')
            return
  
        for ep in range(self.start_ep, self.config.n_ep):
            self.model.train()
            total_tp_loss = total_disp_loss = 0
            advanced_loss_sums = None
            start = time()
            p_train_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
            for it, [rgb, sar, rgb_warp, sar_warp, gt_tp, gt_disp] in p_train_bar:

                image_sar = sar.cuda()      # [8,1,256,256] 
                image_rgb = rgb.cuda()
                image_sar_warp = sar_warp.cuda()
                image_rgb_warp = rgb_warp.cuda()
                gt_tp = gt_tp.squeeze(1).cuda()         #
                gt_disp = gt_disp.squeeze(1).cuda()

                self.model.update_RF(image_rgb, image_sar, image_rgb_warp, image_sar_warp, gt_tp, gt_disp)

                total_tp_loss = total_tp_loss + self.model.loss_tp
                total_disp_loss = total_disp_loss + self.model.loss_disp
                if hasattr(self.model, 'last_loss_components') and self.model.last_loss_components:
                    if advanced_loss_sums is None:
                        advanced_loss_sums = {k: 0.0 for k in self.model.last_loss_components.keys()}
                    for key, value in self.model.last_loss_components.items():
                        advanced_loss_sums[key] += _to_float(value)
                end = time()
                p_train_bar.set_description(f'training ep: %d | time : {str(round(end - start, 0))}' % ep)
                
                # 每 100 个批次清理一次内存
                if it % 100 == 0:
                    # 释放不需要的变量
                    del image_sar, image_rgb, image_sar_warp, image_rgb_warp, gt_tp, gt_disp
                    # 清理 GPU 缓存
                    torch.cuda.empty_cache()
                    # 强制垃圾回收
                    gc.collect()

            avg_tp = total_tp_loss / len(self.train_loader)
            avg_disp = total_disp_loss / len(self.train_loader)
            avg_advanced_losses = None
            if advanced_loss_sums is not None:
                avg_advanced_losses = {
                    key: value / max(len(self.train_loader), 1)
                    for key, value in advanced_loss_sums.items()
                }

            train_loss_record(
                self.config,
                ep=ep,
                tp_loss=avg_tp,
                disp_loss=avg_disp,
                lr=self.model.RES_opt.param_groups[0]['lr'],
                advanced_losses=avg_advanced_losses,
            )
            
            # epoch 结束后清理内存
            # 释放训练相关变量
            del total_tp_loss, total_disp_loss, avg_tp, avg_disp
            # 清理 GPU 缓存
            torch.cuda.empty_cache()
            # 强制垃圾回收
            gc.collect()
            
        
            self.saver.write_img(ep=ep, model=self.model)

            run_val = _should_run_validation(ep, self.config.n_ep)
            save_val_visualization = _should_save_val_visualization(ep, self.config.n_ep)
            if run_val:
                self.model.eval()
                if len(self.test_loader) > 0:
                    total_re_loss = 0
                    total_re_masked_loss = 0
                    total_valid_ratio = 0
                    total_cor_rmse_loss = 0
                    total_disp_rmse_loss = 0
                    total_cmr_1 = 0
                    total_cmr_3 = 0
                    total_cmr_5 = 0
                    total_present_ncc = 0
                    total_present_mi = 0
                    total_present_count = 0
 
                    start = time()
                    p_test_bar = tqdm(enumerate(self.test_loader), total=len(self.test_loader))
                    first_val_stats = None
                    for it, [rgb_, sar_, rgb_warp_, sar_warp_, gt_tp_, gt_disp_] in p_test_bar:
                        image_sar_ = sar_.cuda()    # [1,1,256,256]
                        image_rgb_ = rgb_.cuda()
                        image_sar_warp_ = sar_warp_.cuda()
                        image_rgb_warp_ = rgb_warp_.cuda() 
                        gt_tp_ = gt_tp_.squeeze(1).cuda()      # [1,2,3]
                        gt_disp_ = gt_disp_.squeeze(1).cuda()  # [1,256,256,2]
                        with torch.no_grad():
                            out = self.model.test_forward(image_rgb_, image_sar_, image_rgb_warp_, image_sar_warp_, gt_tp_, gt_disp_)
                            if len(out) >= 12:
                                image_sar_reg, cor_rmse, disp_rmse, sar_re, sar_re_masked, valid_ratio, cmr_1, cmr_3, cmr_5, valid_mask, present_ncc, present_mi = out
                            elif len(out) >= 11:
                                image_sar_reg, cor_rmse, disp_rmse, sar_re, sar_re_masked, valid_ratio, cmr_1, cmr_3, cmr_5, valid_mask, present_ncc = out
                                present_mi = None
                            else:
                                image_sar_reg, cor_rmse, disp_rmse, sar_re, sar_re_masked, valid_ratio, cmr_1, cmr_3, cmr_5, valid_mask = out
                                present_ncc = None
                                present_mi = None

                        total_re_loss = total_re_loss + sar_re
                        total_re_masked_loss = total_re_masked_loss + sar_re_masked
                        total_valid_ratio = total_valid_ratio + valid_ratio
                        total_cor_rmse_loss = total_cor_rmse_loss + cor_rmse
                        total_disp_rmse_loss = total_disp_rmse_loss + disp_rmse
                        total_cmr_1 = total_cmr_1 + cmr_1
                        total_cmr_3 = total_cmr_3 + cmr_3
                        total_cmr_5 = total_cmr_5 + cmr_5
                        if present_ncc is not None:
                            total_present_ncc = total_present_ncc + present_ncc
                            total_present_count += 1
                        if present_mi is not None:
                            total_present_mi = total_present_mi + present_mi

                        if it in self.val_vis_index_set:
                            if first_val_stats is None:
                                first_val_stats = tensor_stats(self.model.image_sar_reg)
                                first_val_stats['valid_ratio'] = float(valid_ratio.detach().cpu().item() if hasattr(valid_ratio, 'detach') else valid_ratio)
                                first_val_stats['valid_zero_ratio'] = float((valid_mask <= 0).float().mean().item())
                            if save_val_visualization:
                                opt_img_color = cv2.imread(self.test_loader.dataset.files_rgb[it], cv2.IMREAD_COLOR)
                                if opt_img_color is None:
                                    opt_img_color = self.model.image_rgb[0]
                                self.saver.write_val_img(ep=ep, idx=it, model=self.model)
                                self.saver.write_val_checkerboard(
                                    ep=ep,
                                    idx=it,
                                    opt_img=opt_img_color,
                                    sar_reg_img=self.model.image_sar_reg[0],
                                    num_checkers=6,
                                )
                                if hasattr(self.model, 'sar_reg_aff'):
                                    self.saver.write_val_checkerboard_coarse(
                                        ep=ep,
                                        idx=it,
                                        opt_img=opt_img_color,
                                        sar_reg_img=self.model.sar_reg_aff[0],
                                        num_checkers=6,
                                    )
                                self.saver.write_val_fine_sar_raw(
                                    ep=ep,
                                    idx=it,
                                    sar_reg_img=(
                                        self.model.image_sar_reg_nointerp[0]
                                        if getattr(self.model, 'image_sar_reg_nointerp', None) is not None
                                        else self.model.image_sar_reg[0]
                                    ),
                                )
                                self.saver.write_val_fine_sar_interp(
                                    ep=ep,
                                    idx=it,
                                    sar_reg_img=self.model.image_sar_reg[0],
                                )
                        
                        # 测试过程中清理内存
                        del image_sar_, image_rgb_, image_sar_warp_, image_rgb_warp_, gt_tp_, gt_disp_, image_sar_reg, cor_rmse, disp_rmse, sar_re, sar_re_masked, valid_ratio, cmr_1, cmr_3, cmr_5, valid_mask
                        torch.cuda.empty_cache()
                        gc.collect()

                    end = time()
                    p_test_bar.set_description(f'testing  ep: %d | time : {str(round(end - start, 2))}' % ep)
              

                    avg_disp = (total_disp_rmse_loss / len(self.test_loader))
                    avg_cor = (total_cor_rmse_loss / len(self.test_loader))
                    avg_re = total_re_loss / len(self.test_loader)
                    avg_re_masked = total_re_masked_loss / len(self.test_loader)
                    avg_valid_ratio = total_valid_ratio / len(self.test_loader)
                    avg_cmr_1 = total_cmr_1 / len(self.test_loader)
                    avg_cmr_3 = total_cmr_3 / len(self.test_loader)
                    avg_cmr_5 = total_cmr_5 / len(self.test_loader)
                    avg_disp_scalar = float(avg_disp.detach().cpu().item() if hasattr(avg_disp, 'detach') else avg_disp)
                    avg_cor_scalar = float(avg_cor.detach().cpu().item() if hasattr(avg_cor, 'detach') else avg_cor)
                    avg_re_scalar = float(avg_re.detach().cpu().item() if hasattr(avg_re, 'detach') else avg_re)
                    avg_re_masked_scalar = float(avg_re_masked.detach().cpu().item() if hasattr(avg_re_masked, 'detach') else avg_re_masked)
                    avg_valid_ratio_scalar = float(avg_valid_ratio.detach().cpu().item() if hasattr(avg_valid_ratio, 'detach') else avg_valid_ratio)
                    avg_cmr_1_scalar = float(avg_cmr_1.detach().cpu().item() if hasattr(avg_cmr_1, 'detach') else avg_cmr_1)
                    avg_cmr_3_scalar = float(avg_cmr_3.detach().cpu().item() if hasattr(avg_cmr_3, 'detach') else avg_cmr_3)
                    avg_cmr_5_scalar = float(avg_cmr_5.detach().cpu().item() if hasattr(avg_cmr_5, 'detach') else avg_cmr_5)
                    avg_ncc_scalar = None
                    avg_mi_scalar = None
                    if total_present_count > 0:
                        avg_ncc = total_present_ncc / total_present_count
                        avg_ncc_scalar = float(avg_ncc.detach().cpu().item() if hasattr(avg_ncc, 'detach') else avg_ncc)
                        avg_mi = total_present_mi / total_present_count
                        avg_mi_scalar = float(avg_mi.detach().cpu().item() if hasattr(avg_mi, 'detach') else avg_mi)
                  
                    test_loss_record(self.config, ep=ep, re=avg_re_scalar, re_masked=avg_re_masked_scalar, valid_ratio=avg_valid_ratio_scalar, cor_rmse=avg_cor_scalar, disp_rmse=avg_disp_scalar, cmr_1=avg_cmr_1_scalar, cmr_3=avg_cmr_3_scalar, cmr_5=avg_cmr_5_scalar)
                    if avg_ncc_scalar is not None:
                        presentation_loss_record(self.config, ep=ep, ncc=avg_ncc_scalar, mi=avg_mi_scalar)
                    print('Epoch %d validation | re: %.4f | re_masked: %.4f | valid_ratio: %.4f | cor: %.4f | disp: %.4f | cmr@1px: %.4f | cmr@3px: %.4f | cmr@5px: %.4f'
                          % (ep, avg_re_scalar, avg_re_masked_scalar, avg_valid_ratio_scalar, avg_cor_scalar, avg_disp_scalar, avg_cmr_1_scalar, avg_cmr_3_scalar, avg_cmr_5_scalar))
                    if avg_ncc_scalar is not None:
                        print('Epoch %d presentation | NCC: %.4f | MI: %.4f' % (ep, avg_ncc_scalar, avg_mi_scalar))
                    if first_val_stats is not None:
                        print('Epoch %d validation sample | image_sar_reg nan_ratio: %.6f | zero_ratio: %.6f | min: %.6f | max: %.6f | valid_ratio: %.6f | valid_zero_ratio: %.6f'
                              % (ep, first_val_stats['nan_ratio'], first_val_stats['zero_ratio'], first_val_stats['min'], first_val_stats['max'], first_val_stats['valid_ratio'], first_val_stats['valid_zero_ratio']))
                        append_validation_sample_stats(self.config, ep=ep, stats=first_val_stats)

                    if lowest_re > avg_re_scalar:
                        # 确保正确处理不同类型的变量
                        re_value = str(avg_re_scalar)
                        re_masked_value = str(avg_re_masked_scalar)
                        valid_ratio_value = str(avg_valid_ratio_scalar)
                        cor_value = str(avg_cor_scalar)
                        disp_value = str(avg_disp_scalar)
                        cmr_1_value = str(avg_cmr_1_scalar)
                        cmr_3_value = str(avg_cmr_3_scalar)
                        cmr_5_value = str(avg_cmr_5_scalar)
                        result = {'re': re_value, 're_masked': re_masked_value, 'valid_ratio': valid_ratio_value, 'cor_rmse': cor_value, 'avg_disp': disp_value, 'cmr_1px': cmr_1_value, 'cmr_3px': cmr_3_value, 'cmr_5px': cmr_5_value}
                        if avg_ncc_scalar is not None:
                            result['ncc'] = str(avg_ncc_scalar)
                            result['mi'] = str(avg_mi_scalar)
                        lowest_re = avg_re_scalar

                    if self.config.save_top_k > 0:
                        checkpoint_name = f'{self.config.data_name}_top_ep{ep:04d}_re{avg_re_scalar:.6f}.pth'
                        checkpoint_path = self.saver.save_named_model(checkpoint_name, self.model, epoch=ep)
                        self.best_checkpoints.append({
                            're': avg_re_scalar,
                            'epoch': ep,
                            'filename': checkpoint_name,
                            'path': checkpoint_path,
                        })
                        self.best_checkpoints.sort(key=lambda item: item['re'])

                        if len(self.best_checkpoints) > self.config.save_top_k:
                            removed_checkpoint = self.best_checkpoints.pop()
                            self.saver.remove_model(removed_checkpoint['filename'])
                    
                    # 测试结束后清理内存
                    del total_re_loss, total_re_masked_loss, total_valid_ratio, total_cor_rmse_loss, total_disp_rmse_loss, total_cmr_1, total_cmr_3, total_cmr_5, avg_re, avg_re_masked, avg_valid_ratio, avg_cor, avg_disp, avg_cmr_1, avg_cmr_3, avg_cmr_5
                    torch.cuda.empty_cache()
                    gc.collect()
                else:
                    print('No test data available, skipping test phase')

            if self.config.n_ep_decay > -1:     # 从那个epoch开始减少学习率
                self.model.update_lr()
        method_dict={'method':'opt_sar','id':0,'batch_size':self.config.batch_size,'epoch':self.config.n_ep}
        set_result(result=result,method_dict=method_dict)


    



def train_loss_record(config, ep, tp_loss, disp_loss, lr, advanced_losses=None):
    os.makedirs(config.train_logs_dir, exist_ok=True)
    log_path = os.path.join(config.train_logs_dir, 'train_log.txt')
    tp_loss_v = _to_float(tp_loss)
    disp_loss_v = _to_float(disp_loss)
    lr_v = _to_float(lr)

    base_msg = 'No.%d Epoch: tp_loss:%.6f | disp_loss:%.6f | lr:%.6f' % (ep, tp_loss_v, disp_loss_v, lr_v)
    if advanced_losses:
        advanced_msg = (
            ' | L_total:%.6f | L_sim:%.6f | L_evidential:%.6f | L_GCL:%.6f | L_tp:%.6f | L_disp:%.6f'
            % (
                _to_float(advanced_losses.get('L_total', 0.0)),
                _to_float(advanced_losses.get('L_sim', 0.0)),
                _to_float(advanced_losses.get('L_evidential', 0.0)),
                _to_float(advanced_losses.get('L_GCL', 0.0)),
                _to_float(advanced_losses.get('L_tp', 0.0)),
                _to_float(advanced_losses.get('L_disp', 0.0)),
            )
        )
        base_msg = base_msg + advanced_msg
        if 'L_sar_fid' in advanced_losses:
            base_msg += ' | L_sar_fid:%.6f' % _to_float(advanced_losses.get('L_sar_fid', 0.0))
        if 'L_sar_border' in advanced_losses:
            base_msg += ' | L_sar_border:%.6f' % _to_float(advanced_losses.get('L_sar_border', 0.0))
        if 'L_sar_mi' in advanced_losses:
            base_msg += ' | L_sar_mi:%.6f' % _to_float(advanced_losses.get('L_sar_mi', 0.0))

    with open(log_path, 'a') as f:
        f.write(base_msg + ' \n')


def test_loss_record(config, ep, re, re_masked, valid_ratio, cor_rmse, disp_rmse, cmr_1, cmr_3, cmr_5):
    os.makedirs(config.test_logs_dir, exist_ok=True)
    log_path = os.path.join(config.test_logs_dir, 'test_log.txt')
    with open(log_path, 'a') as f:
        f.write('No.%d Epoch: re:%.4f | re_masked:%.4f | valid_ratio:%.4f | cor:%.4f | disp:%.4f | cmr@1px:%.4f | cmr@3px:%.4f | cmr@5px:%.4f \n'
                % (ep, re, re_masked, valid_ratio, cor_rmse, disp_rmse, cmr_1, cmr_3, cmr_5))


def presentation_loss_record(config, ep, ncc, mi=None):
    os.makedirs(config.present_logs_dir, exist_ok=True)
    log_path = os.path.join(config.present_logs_dir, 'presentation_log.txt')
    with open(log_path, 'a') as f:
        if mi is None:
            f.write('No.%d Epoch: ncc:%.6f \n' % (ep, ncc))
        else:
            f.write('No.%d Epoch: ncc:%.6f | mi:%.6f \n' % (ep, ncc, mi))


def append_validation_sample_stats(config, ep, stats):
    os.makedirs(config.test_logs_dir, exist_ok=True)
    log_path = os.path.join(config.test_logs_dir, 'test_log.txt')
    with open(log_path, 'a') as f:
        f.write('No.%d ValidationSample: image_sar_reg nan_ratio:%.6f | zero_ratio:%.6f | min:%.6f | max:%.6f | valid_ratio:%.6f | valid_zero_ratio:%.6f \n'
                % (ep, stats['nan_ratio'], stats['zero_ratio'], stats['min'], stats['max'], stats['valid_ratio'], stats['valid_zero_ratio']))
