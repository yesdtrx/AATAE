import os
import csv


import numpy as np
from tqdm import tqdm
from numpy import mean
import torch.utils.data
import torchvision.utils as vutils
from lib.visualizer import Visualizer
import pytorch_ssim

def mkdir(path):
	folder = os.path.exists(path)
	if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
		os.makedirs(path)

class BaseModel():
    """ Base Model
    """

    def __init__(self, opt, data):
        self.seed(opt.manualseed)

        # Initalize variables.
        self.opt = opt
        self.visualizer = Visualizer(opt)
        self.data = data
        self.trn_dir = os.path.join(self.opt.outf, self.opt.name, 'train')
        self.tst_dir = os.path.join(self.opt.outf, self.opt.name, 'test')
        self.device = torch.device("cuda:0" if self.opt.device != 'cpu' else "cpu")

        self.train_which = ''
        self.g_ssim = 0
        self.g_test = 0
        self.auc = 0
        self.lr_g = self.opt.lr_g
        self.lr_d = self.opt.lr_d


        ##加入self.start和self.stop控制训练epoch，同时又不影响opt里面的值。
        self.start = 0
        self.stop = 0

        self.repe = 0
        self.loss_path = ''
        self.result_path = ''
        self.result_mean_path = ''

        self.best_auc = 0
        self.best_epoch = 0
        self.best_ssim = 0
        self.max_ssim = 0
        self.best_d_real = 0
        self.acc = 0
        self.recall = 0
        self.precision = 0

        ##表示整个训练集判别器分数的平均值
        self.d_real = 0
        self.d_fake = 0

        ## 记录训练 netD 或者 netG 多少次
        self.g_and_d_num = 0
        self.trainD_num = 0
        self.trainG_num = 0

        ##记录每个epoch有多少个iter
        self.iter_per_epoch = 0


        self.flag1 = True
        self.flag2 = True
        self.flag3 = True


        self.save_train_flag = True

    ##
    def seed(self, seed_value):
        if seed_value == -1:
            return

        import random
        random.seed(seed_value)
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        np.random.seed(seed_value)
        torch.backends.cudnn.deterministic = True

    def reinit_lr(self):

        ##如果学习率太小，重新设置学习率
        if self.lr_d < 0.01 * self.opt.lr_d:
            self.lr_d = self.lr_d * 2

        if self.lr_g < 0.01 * self.opt.lr_g:
            self.lr_g = self.lr_g * 2

        self.optimizer_d.param_groups[0]['lr'] = self.lr_d
        self.optimizer_g.param_groups[0]['lr'] = self.lr_g

    ##
    def set_input(self, input: torch.Tensor, noise: bool = False):
        """ Set input and ground truth

        Args:
            input (FloatTensor): Input data for batch i.
        """
        with torch.no_grad():
            self.input.resize_(input[0].size()).copy_(input[0])
            self.gt.resize_(input[1].size()).copy_(input[1])
            self.label.resize_(input[1].size())

            # Add noise to the input.
            if noise: self.noise.data.copy_(torch.randn(self.noise.size()))

    ##
    def get_current_images(self):

        reals = self.input.data
        fakes = self.fake.data

        return reals, fakes

    ##
    def save_weights(self, save_which):

        weight_dir = os.path.join('output_train', self.opt.dataset, 'weights')
        if not os.path.exists(weight_dir):
            os.makedirs(weight_dir)
        if save_which == 'best':
            path_d = f"{weight_dir}/{self.repe}_netD_best.pth"
            path_g = f"{weight_dir}/{self.repe}_netG_best.pth"
        elif save_which == 'stop':
            path_d = f"{weight_dir}/{self.repe}_netD_stop.pth"
            path_g = f"{weight_dir}/{self.repe}_netG_stop.pth"
        elif save_which == 'phase1':
            path_d = f"{weight_dir}/{self.repe}_netD_phase1.pth"
            path_g = f"{weight_dir}/{self.repe}_netG_phase1.pth"
        elif save_which == 'phase2':
            path_d = f"{weight_dir}/{self.repe}_netD_phase2.pth"
            path_g = f"{weight_dir}/{self.repe}_netG_phase2.pth"
        elif save_which == 'phase3':
            path_d = f"{weight_dir}/{self.repe}_netD_phase3.pth"
            path_g = f"{weight_dir}/{self.repe}_netG_phase3.pth"
        elif save_which == 'regular':
            path_d = f"{weight_dir}/{self.repe}_netD_{self.epoch}.pth"
            path_g = f"{weight_dir}/{self.repe}_netG_{self.epoch}.pth"

        torch.save({'epoch': self.epoch, 'state_dict': self.netd.state_dict()}, path_d)
        torch.save({'epoch': self.epoch, 'state_dict': self.netg.state_dict()}, path_g)


    def load_weights(self, netG_name, netD_name, weight_dir):

        if weight_dir == '':
            weight_dir = os.path.join('output_train', self.opt.dataset, 'weights')
        path_g = weight_dir + '/' + netG_name
        path_d = weight_dir + '/' + netD_name

        # Load the weights of netg and netd.
        print('>> Loading weights...')
        weights_g = torch.load(path_g)['state_dict']
        weights_d = torch.load(path_d)['state_dict']
        try:
            self.netg.load_state_dict(weights_g)
            self.netd.load_state_dict(weights_d)
        except IOError:
            raise IOError("netG weights not found")
        print('   Done.')

    def train_epochs_T(self):
        for self.epoch in range(self.start, self.stop):
            ## 设为训练模式
            self.netg.train()

            ##如果学习率太小，重新设置学习率
            self.reinit_lr()
            self.train_g()

            if self.auc > self.best_auc:
                self.best_auc = self.auc
                save_which = 'best'
                self.save_weights(save_which)
                self.best_epoch = self.epoch
                self.best_ssim = self.g_ssim
                self.best_d_real = self.d_real

            self.visualizer.print_current_performance(self.auc, self.best_auc)

            if self.g_ssim > self.max_ssim:
                self.max_ssim = self.g_ssim

            print(self.optimizer_d.param_groups[0]['lr'])
            print(self.optimizer_g.param_groups[0]['lr'])



    def train_epochs_AT(self):

        for self.epoch in range(self.start, self.stop):
            ## 设为训练模式
            self.netg.train()
            self.netd.train()

            ##如果学习率太小，重新设置学习率
            self.reinit_lr()

            self.train_one_epoch()

            if self.auc > self.best_auc:
                self.best_auc = self.auc
                save_which = 'best'
                self.save_weights(save_which)
                self.best_epoch = self.epoch
                self.best_ssim = self.g_ssim
                self.best_d_real = self.d_real

            self.visualizer.print_current_performance(self.auc, self.best_auc)

            if self.g_ssim > self.max_ssim:
                self.max_ssim = self.g_ssim

            print(self.optimizer_d.param_groups[0]['lr'])
            print(self.optimizer_g.param_groups[0]['lr'])
    ##
    def train_epochs_AAT(self):
        ##一阶段：g and d
        ##二阶段：g
        ##三阶段：d

        self.train_which = 'g and d'
        ##保持两个数组存放三个阶段的指标，当数量大于耐心 p，观察倒数 1到 p-1中是否有大于倒数 p 位置的指标，若没有则停止对抗训练。
        phase1_list = []
        phase2_list = []

        for self.epoch in range(self.start, self.stop):
            ## 设为训练模式
            self.netg.train()
            self.netd.train()

            ##如果学习率太小，重新设置学习率
            self.reinit_lr()

            ##第二阶段训练满epoch2次,进入第三阶段
            if self.trainG_num == self.opt.epoch2:
                self.train_which = 'd'
                ##计数器继续加1，以免再进入此判断条件
                self.trainG_num += 1

            ## 训练第1阶段的条件
            if self.train_which == 'g and d':
                phase1_list.append(self.g_ssim)
                if len(phase1_list) >= self.opt.p1:
                    flag = True
                    for ssim_tmp in phase1_list[-1 * self.opt.p1:]:
                        if ssim_tmp > phase1_list[-1 * self.opt.p1]:
                            flag = False
                            break
                    if flag:
                        self.train_which = 'g'
                        continue
                self.train_one_epoch()
                self.g_and_d_num += 1

            ###训练第2阶段的条件
            elif self.train_which == 'g':
                phase2_list.append(self.g_ssim)
                if len(phase2_list) >= self.opt.p2:
                    flag = True
                    for ssim_tmp in phase2_list[-1 * self.opt.p2:]:
                        if ssim_tmp > phase2_list[-1 * self.opt.p2]:
                            flag = False
                            break
                    if flag:
                        self.train_which = 'd'
                        continue
                self.train_g()
                ##这个阶段d_real和d_fake不变，是因为没有再计算这两者
                self.cal_g_loss()
                self.trainG_num += 1

            ##训练第3阶段的条件
            elif self.train_which == 'd':
                if self.d_real >= self.opt.d_th:
                    break
                self.train_d()
                self.cal_d_loss()
                self.trainD_num += 1

            if self.auc > self.best_auc:
                self.best_auc = self.auc
                save_which = 'best'
                self.save_weights(save_which)
                self.best_epoch = self.epoch
                ## #########保存生成图片
                ##self.save_images('images_best')
                self.best_ssim = self.g_ssim
                self.best_d_real = self.d_real

            self.visualizer.print_current_performance(self.auc, self.best_auc)

            if self.g_ssim > self.max_ssim:
                self.max_ssim = self.g_ssim


            print(self.optimizer_d.param_groups[0]['lr'])
            print(self.optimizer_g.param_groups[0]['lr'])

    def train_one_epoch(self):

        if self.opt.save_train_images and self.save_train_flag:
            self.save_train_images()
            self.save_train_flag = False

        self.train_which = 'g and d'
        print(">> Training g and d on %s. Epoch %d/%d." % (self.opt.dataset, self.epoch, self.stop - 1))
        if self.epoch % 10 == 0:
            self.lr_d = self.lr_d * 0.9
            self.lr_g = self.lr_g * 0.9

        for i, data in enumerate(self.data.train_ordered, 0):
            self.set_input(data)
            self.forward()
            self.update_netd()
        self.cal_d_loss()

        for i, data in enumerate(self.data.train_ordered, 0):
            self.set_input(data)
            self.forward()
            self.update_netg()
        self.cal_g_loss()

        self.save_loss()

    def train_d(self):

        self.train_which = 'd'
        print(">> Training d on %s. Epoch %d/%d." % (self.opt.dataset, self.epoch, self.stop))

        if self.epoch % 10 == 0:
            self.lr_d = self.lr_d * 0.9

        for i, data in enumerate(self.data.train_ordered, 0):
            self.set_input(data)
            self.forward()
            self.update_netd()

        self.save_loss()

    def train_g(self):

        print(">> Training g on %s. Epoch %d/%d." % (self.opt.dataset, self.epoch, self.stop))

        if self.epoch % 10 == 0:
            self.lr_g = self.lr_g * 0.9

        for i, data in enumerate(self.data.train_ordered, 0):
            self.set_input(data)
            self.forward()
            self.update_netg()

        self.save_loss()

    def save_csv_header(self):

        mkdir('./csv_train')
        mkdir('./tmp')
        ###保存 Loss文件的header
        with open(self.loss_path, "a", newline='') as file:
            f_csv = csv.writer(file)
            header = ['epoch', 'd_real', 'd_fake', 'ssim', 'auroc', 'train_which', 'lr_d', 'lr_g',
                      'n_ssim', 'n_d_real', 'ssim_dif', 'real_dif',
                      'trainD_num', 'trainG_num',
                      'Accuracy', 'Recall', 'Precision', 'p1', 'p2']
            f_csv.writerow(header)

        ###保存 result文件的header，前缀 best_ 代表auc最高的时候的值，如 best_ssim 并不是最高的ssim，而是best_epoch的ssim。
        if not os.path.exists(self.result_path):
            with open(self.result_path, "a", newline='') as file:
                f_csv = csv.writer(file)
                header = ['dataset', 'stop_auc', 'best_auc',  'g_test', 'acc', 'recall', 'precision', 'stop_epoch', 'best_epoch', 'stop_ssim', 'best_ssim',
                          'stop_d_real', 'best_d_real', 'max_ssim']
                f_csv.writerow(header)

        ###保存 result_mean文件的header
        if not os.path.exists(self.result_mean_path):
            with open(self.result_mean_path, "a", newline='') as file:
                f_csv = csv.writer(file)
                header = ['dataset', 'stop_auc', 'best_auc', 'g_test', 'stop_epoch', 'best_epoch', 'stop_ssim', 'best_ssim',
                          'stop_d_real', 'best_d_real']
                f_csv.writerow(header)

    def cal_d_loss(self):
        ###计算loss相关参数
        reals = []
        fakes = []
        for i, data in enumerate(self.data.train_ordered, 0):
            self.set_input(data)
            self.fake = self.netg(self.input)
            self.pred_real, _ = self.netd(self.input)
            self.pred_fake, _ = self.netd(self.fake)

            reals = reals + self.pred_real.tolist()
            fakes = fakes + self.pred_fake.tolist()

        self.d_real = mean(reals)
        self.d_fake = mean(fakes)

    def cal_g_loss(self):

        ssims = []
        for i, data in enumerate(self.data.train_ordered, 0):
            self.set_input(data)
            self.fake = self.netg(self.input)
            ssims = ssims + pytorch_ssim.ssim(self.fake, self.input, self.opt.window_train, size_average=False).cpu().tolist()
        self.g_ssim = mean(ssims)

    def save_loss(self):
        ###计算loss相关参数
        tmp = []
        tmp.append(self.epoch)
        tmp.append(str('%.6f' % self.d_real))
        tmp.append(str('%.6f' % self.d_fake))
        tmp.append(str('%.6f' % self.g_ssim))

        print('self.d_real = ' + str('%.6f' % self.d_real))
        print('self.d_fake = ' + str('%.6f' % self.d_fake))
        print('SSIM = ' + str('%.6f' % self.g_ssim))

        res = self.test()
        self.auc = res['auroc']
        self.acc = res['accuracy']
        self.recall = res['recall']
        self.precision = res['precision']
        n_ssim = res['normal_ssim']
        self.g_test = n_ssim
        n_d_real = res['n_d_real']
        ssim_dif = res['ssim_dif']
        real_dif = res['real_dif']

        tmp.append(str('%.6f' % self.auc))
        tmp.append(self.train_which)
        tmp.append(str('%.8f' % self.optimizer_d.param_groups[0]['lr']))
        tmp.append(str('%.8f' % self.optimizer_g.param_groups[0]['lr']))
        tmp.append(str('%.6f' % n_ssim))
        tmp.append(str('%.6f' % n_d_real))
        tmp.append(str('%.6f' % ssim_dif))
        tmp.append(str('%.6f' % real_dif))
        tmp.append(str('%d' % self.trainD_num))
        tmp.append(str('%d' % self.trainG_num))
        tmp.append(str('%.6f' % self.acc))
        tmp.append(str('%.6f' % self.recall))
        tmp.append(str('%.6f' % self.precision))
        tmp.append(str('%d' % self.opt.p1))
        tmp.append(str('%d' % self.opt.p2))

        ##保存 loss
        with open(self.loss_path, "a", newline='') as file:
            f_csv = csv.writer(file)
            f_csv.writerow(tmp)

    def save_train_images(self):
        dst = './train_images/' + self.opt.dataset
        if not os.path.exists(dst):
            img_names = []
            mkdir(dst)
            for item in self.data.train_ordered.sampler.data_source.imgs:
                img_path = item[0]
                name = os.path.basename(img_path)
                img_names.append(name)

            for i, data in enumerate(self.data.train_ordered, 0):
                self.set_input(data)
                reals = self.input.data
                j = i * self.opt.batchsize
                for img in reals:
                    vutils.save_image(img, '%s/%s' % (dst, img_names[j]), normalize=True)
                    j += 1
    def train(self, repe):

        ###########如果从checkpoints开始训练
        if self.opt.train_from_checkpoints:
            netG_name = self.opt.netG
            netD_name = self.opt.netD
            self.load_weights(netG_name, netD_name, self.opt.weight_dir)

        self.repe = repe
        self.start = self.opt.start_iter
        self.stop = self.opt.niter + 1

        self.loss_path = './csv_train/' + self.opt.dataset + '_' + str(self.opt.niter) + '_loss_' + str(self.repe) + '.csv'
        self.result_path = './csv_train/result.csv'
        self.result_mean_path = './csv_train/result_mean.csv'

        self.save_csv_header()

        ###根据训练集样本数量、批次大小、计算各个阶段的耐心epoch： p1、p2、p3
        nsample = len(self.data.train_ordered.dataset.samples)
        self.iter_per_epoch = nsample // self.opt.batchsize

        if self.opt.training_mode == 'T':
            self.train_epochs_T()
        elif self.opt.training_mode == 'AT':
            self.train_epochs_AT()
        elif self.opt.training_mode == 'AAT':
            self.train_epochs_AAT()

        stop_epoch = self.epoch
        save_which = 'stop'
        self.save_weights(save_which)

        ##记录每次实验 auc 在 result.csv
        info = []
        stop_auc = self.auc
        stop_ssim = self.g_ssim
        stop_d_real = self.d_real
        with open(self.result_path, "a", newline='') as file:
            info.append(self.opt.dataset)
            info.append('%.4f' % stop_auc)
            info.append('%.4f' % self.best_auc)
            info.append('%.4f' % self.g_test)

            info.append('%.4f' % self.acc)
            info.append('%.4f' % self.recall)
            info.append('%.4f' % self.precision)

            info.append(stop_epoch)
            info.append(self.best_epoch)

            info.append('%.4f' % stop_ssim)
            info.append('%.4f' % self.best_ssim)

            info.append('%.4f' % stop_d_real)
            info.append('%.4f' % self.best_d_real)
            info.append('%.4f' % self.max_ssim)


            f_csv = csv.writer(file)
            f_csv.writerow(info)

        ##记录 stop_auc、best_auc、stop_epoch、best_epoch、stop_ssim、best_ssim、
        # stop_d_real、best_d_real平均值在 result_mean.csv
        info = []
        stop_aucs = []
        best_aucs = []
        g_tests = []
        stop_epochs = []
        best_epochs = []
        stop_ssims = []
        best_ssims = []
        stop_d_reals = []
        best_d_reals = []
        with open(self.result_path, 'r') as f:
            result = list(csv.reader(f))
            l = len(result) - 1
            re = self.opt.repetition
            split = re * (-1)
            if l % re == 0:
                for row in result[split:]:
                    stop_aucs.append(row[1])
                    best_aucs.append(row[2])
                    g_tests.append(row[3])
                    stop_epochs.append(row[7])
                    best_epochs.append(row[8])
                    stop_ssims.append(row[9])
                    best_ssims.append(row[10])
                    stop_d_reals.append(row[11])
                    best_d_reals.append(row[12])
                info.append(self.opt.dataset)
                info.append('%.4f' % mean([float(x) for x in stop_aucs]))
                info.append('%.4f' % mean([float(x) for x in best_aucs]))
                info.append('%.4f' % mean([float(x) for x in g_tests]))
                info.append('%.4f' % mean([float(x) for x in stop_epochs]))
                info.append('%.4f' % mean([float(x) for x in best_epochs]))
                info.append('%.4f' % mean([float(x) for x in stop_ssims]))
                info.append('%.4f' % mean([float(x) for x in best_ssims]))
                info.append('%.4f' % mean([float(x) for x in stop_d_reals]))
                info.append('%.4f' % mean([float(x) for x in best_d_reals]))

        if info:
            with open(self.result_mean_path, "a", newline='') as file:
                f_csv = csv.writer(file)
                f_csv.writerow(info)

        self.test2()

        print(">> Training model %s.[Done]" % self.name)

