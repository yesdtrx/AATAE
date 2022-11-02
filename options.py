""" Options

This script is largely based on junyanz/pytorch-CycleGAN-and-pix2pix.

Returns:
    [argparse]: Class containing argparse
"""

import argparse
import os
import torch

# pylint: disable=C0103,C0301,R0903,W0622

class Options():
    """Options class

    Returns:
        [argparse]: argparse containing train and test options
    """

    def __init__(self):
        ##
        #Threshold calculation method
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        # 扩展
        self.parser.add_argument('--training_mode', type=str, default='ae', help='T | AT | AAT')
        self.parser.add_argument('--nblock', type=int, default=6, help='resnet生成器残差模块个数')
        self.parser.add_argument('--ndown', type=int, default=2, help='resnet生成器下采样上采样次数')
        self.parser.add_argument('--ssim_std_mul', type=int, default=-1, help='ssim加上标准差的倍数')
        self.parser.add_argument('--d_std_mul', type=int, default=-1, help='d_real加上标准差的倍数')
        self.parser.add_argument('--train_from_checkpoints', action='store_true')
        self.parser.add_argument('--save_train_images', action='store_true', help='Save train images.')
        self.parser.add_argument('--weight_dir', default='', help='dir of trained model')
        self.parser.add_argument('--netG_weight', default='', help='name of trained negG')
        self.parser.add_argument('--netD_weight', default='', help='name of trained negD')
        self.parser.add_argument('--p1', type=int, default=20)
        self.parser.add_argument('--p2', type=int, default=20)
        self.parser.add_argument('--d_th', type=float, default=0)
        self.parser.add_argument('--netg', default='resnet', help='unet | resnet | efficientnet | resnet_light')
        self.parser.add_argument('--con', default='SSIM', help='L1 | L2 | SSIM ')
        self.parser.add_argument('--repetition', type=int, default=1, help='Experiment repetitions')
        self.parser.add_argument('--epoch1', type=int, default=0, help='The maximum epoch of the adversarial training phase')
        self.parser.add_argument('--epoch2', type=int, default=0, help='Maximum epoch of training g alone')
        self.parser.add_argument('--epoch3', type=int, default=0, help='Maximum epoch of training d alone')
        self.parser.add_argument('--window_train', type=int, default=5, help='windowsize of ssim')
        self.parser.add_argument('--window_infer', type=int, default=5, help='windowsize of ssim')


        # Base
        self.parser.add_argument('--dataset', default='')
        self.parser.add_argument('--dataroot', default='', help='path to dataset')
        self.parser.add_argument('--datasetroot', default='', help='path to datasets')
        self.parser.add_argument('--path', default='', help='path to the folder or image to be predicted.')
        self.parser.add_argument('--batchsize', type=int, default=64, help='input batch size')
        self.parser.add_argument('--workers', type=int, help='number of data loading workers', default=8)
        self.parser.add_argument('--droplast', action='store_true', default=False, help='Drop last batch size.')
        self.parser.add_argument('--isize', type=int, default=64, help='input image size.')
        self.parser.add_argument('--nc', type=int, default=3, help='input image channels')
        self.parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
        self.parser.add_argument('--ngf', type=int, default=64)
        self.parser.add_argument('--ndf', type=int, default=64)
        self.parser.add_argument('--extralayers', type=int, default=0, help='Number of extra layers on gen and disc')
        self.parser.add_argument('--device', type=str, default='gpu', help='Device: gpu | cpu')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
        self.parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment')
        self.parser.add_argument('--model', type=str, default='skipganomaly', help='chooses which model to use. ganomaly')
        self.parser.add_argument('--display_server', type=str, default="http://localhost", help='visdom server of the web display')
        self.parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
        self.parser.add_argument('--display_id', type=int, default=0, help='window id of the web display')
        self.parser.add_argument('--display', action='store_true', help='Use visdom.')
        self.parser.add_argument('--verbose', action='store_true', help='Print the training and model details.')
        self.parser.add_argument('--outf', default='./output_train', help='folder to output images and model checkpoints')
        self.parser.add_argument('--manualseed', default=-1, type=int, help='manual seed')
        self.parser.add_argument('--abnormal_class', default='automobile', help='Anomaly class idx for mnist and cifar datasets')
        self.parser.add_argument('--metric', type=str, default='roc', help='Evaluation metric.')

        ##
        # Train
        self.parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        self.parser.add_argument('--save_image_freq', type=int, default=100, help='frequency of saving real and fake images')
        self.parser.add_argument('--save_test_images', action='store_true', help='Save test images for demo.')
        self.parser.add_argument('--resume', default='', help="path to checkpoints (to continue training)")
        self.parser.add_argument('--start_iter', type=int, default=1, help='Start from iteration i')
        self.parser.add_argument('--niter', type=int, default=15, help='number of epochs to train for')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--lr_d', type=float, default=0.0008, help='initial learning rate for adam for netd')
        self.parser.add_argument('--lr_g', type=float, default=0.0002, help='initial learning rate for adam for netg')
        self.parser.add_argument('--w_adv', type=float, default=1, help='Weight for adversarial loss. default=1')
        self.parser.add_argument('--w_con', type=float, default=10, help='Weight for reconstruction loss. default=50')
        self.parser.add_argument('--w_lat', type=float, default=1, help='Weight for latent space loss. default=1')
        self.opt = None

    def parse(self):
        """ Parse Arguments.
        """

        self.opt = self.parser.parse_args()

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        if self.opt.verbose:
            print('------------ Options -------------')
            for k, v in sorted(args.items()):
                print('%s: %s' % (str(k), str(v)))
            print('-------------- End ----------------')

        # save to the disk
        if self.opt.name == 'experiment_name':
            self.opt.name = self.opt.dataset
        expr_dir = os.path.join(self.opt.outf, self.opt.name, 'train')
        test_dir = os.path.join(self.opt.outf, self.opt.name, 'test_images')

        if not os.path.isdir(expr_dir):
            os.makedirs(expr_dir)
        if not os.path.isdir(test_dir):
            os.makedirs(test_dir)

        ##根据niter计算各个阶段训练次数上限 epoch1、epoch2、epoch3
        self.opt.epoch2 = self.opt.niter // 10

        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        return self.opt