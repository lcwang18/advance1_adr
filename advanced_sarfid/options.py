import argparse


class TrainOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()


    # data related
        self.parser.add_argument('--train_data_path', type=str, default='/root/wlc/dataset/Gaza_512/train', help='path to train data')
        self.parser.add_argument('--test_data_path', type=str, default='/root/wlc/dataset/Gaza_512/val', help='path to test data')
        self.parser.add_argument('--batch_size', type=int, default=4, help='batch_size')
        self.parser.add_argument('--nThreads', type=int, default=4, help='cpu threads')
        self.parser.add_argument('--train', action='store_true',default=True, help='Train mode or test mode')
        self.parser.add_argument('--data_name', type=str, default='Gaza_512',
                                 help='name of the dataset (used for saving paths)')
    # generate data
        self.parser.add_argument('--rotation', type=int, default=20)
        self.parser.add_argument('--translation', type=float, default=15)
        self.parser.add_argument('--scaling', type=float, default=0.2)
        self.parser.add_argument('--dim', type=int, default=2)

    # output related
        self.parser.add_argument('--train_img_dir', type=str, default='./results3', help='save path for training process visualization')
        self.parser.add_argument('--model_dir', type=str, default='./model_save3', help='save path for model weights')

    # training related
        self.parser.add_argument('--lr_policy', type=str, default='lambda', help='')
        self.parser.add_argument('--n_ep', type=int, default=300, help='epoch')
        self.parser.add_argument('--n_ep_decay', type=int, default=150, help='which epoch starts to reduce the learning rate，-1：不变')
        self.parser.add_argument('--save_top_k', type=int, default=5, help='number of best checkpoints to keep')
        self.parser.add_argument('--lr', type=float, default=1e-4, help='base learning rate')
        self.parser.add_argument('--model_variant', type=str, default='advanced', choices=['advanced', 'baseline'],
                                 help='training model variant')
        self.parser.add_argument('--resume_path', type=str, default='', help='checkpoint path for resume training')
        self.parser.add_argument('--resume_strict', type=int, default=1, choices=[0, 1], help='strict load state dict')
        self.parser.add_argument('--resume_load_optimizer', type=int, default=1, choices=[0, 1], help='load optimizer state when resuming')
        self.parser.add_argument('--adv_w_tp', type=float, default=1.0, help='weight of coarse four-point RMSE loss')
        self.parser.add_argument('--adv_w_disp', type=float, default=1.0, help='weight of dense displacement RMSE loss')
        self.parser.add_argument('--adv_w_sim', type=float, default=0.0,
                                 help='weight of cross-modal local-NCC loss (0 for SAR-fidelity setting)')
        self.parser.add_argument('--adv_w_evidential', type=float, default=0.1, help='weight of evidential NIG loss')
        self.parser.add_argument('--adv_w_gcl', type=float, default=0.05, help='weight of DVF smooth geometry loss')
        self.parser.add_argument('--adv_w_sar_fid', type=float, default=10.0,
                                 help='weight of SAR self-fidelity loss (pixel+gradient)')
        self.parser.add_argument('--adv_w_sar_border', type=float, default=1.0,
                                 help='weight of SAR invalid-border suppression loss')
        self.parser.add_argument('--adv_w_sar_mi', type=float, default=0.0,
                                 help='weight of SAR-SAR MI loss (optional)')
        self.parser.add_argument('--adv_ncc_window', type=int, default=9, help='window size for local NCC')

    # test related
        self.parser.add_argument('--train_logs_dir', type=str, default='./logs/train3', help='Save path for training information')
        self.parser.add_argument('--test_logs_dir', type=str, default='./logs/test3', help='Save path for testing information')
        self.parser.add_argument('--present_logs_dir', type=str, default='./logs/presentation3',
                                 help='Save path for presentation-quality metrics')
        self.parser.add_argument('--present_metric', type=str, default='both', choices=['ncc', 'mi', 'both'],
                                 help='presentation metric type')
        self.parser.add_argument('--present_mi_bins', type=int, default=64, help='histogram bins for MI evaluation')
        self.parser.add_argument('--re_scale', type=float, default=255.0, help='scale factor for RE L1 metric')
        self.parser.add_argument('--val_warp_seed', type=int, default=2026,
                                 help='fixed random seed base for validation warp reproducibility')
        self.parser.add_argument('--val_warp_min_abs', type=float, default=0.35,
                                 help='minimum absolute random factor for visible val warp (0 disables)')
        self.parser.add_argument('--val_fixed_vis_num', type=int, default=5,
                                 help='number of fixed val visualization samples with deterministic warp')



    def parse(self):
        self.opt = self.parser.parse_args()
        args = vars(self.opt)
        print('\n--- train and test settings ---')
        for name, value in sorted(args.items()):
            print('%s: %s' % (str(name), str(value)))
        return self.opt
