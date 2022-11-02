"""GANomaly
"""
# pylint: disable=C0301,E1101,W0622,C0103,R0902,R0915

##
import csv
import shutil
from collections import OrderedDict
import os
import numpy as np
import pytorch_ssim
from pytorch_ssim_2.ssim2 import ssim2
from torchsummary import summary
from numpy import mean
import torch.optim as optim
import torch.nn as nn
import torch.utils.data
import torchvision.utils as vutils
from matplotlib import pyplot as plt
from lib.models.networks import NetD, weights_init, define_G, define_D
from lib.loss import l2_loss
from lib.models.basemodel import BaseModel
from sklearn.metrics import recall_score, confusion_matrix, f1_score, accuracy_score, precision_score, auc, roc_curve


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)


class Skipganomaly(BaseModel):
    """GANomaly Class
    """

    @property
    def name(self):
        return 'skip-ganomaly'

    def __init__(self, opt, data=None):
        super(Skipganomaly, self).__init__(opt, data)
        ##

        self.epoch = 1

        ##设置一个数组保存loss
        self.loss = []

        ##
        # Create and initialize networks.
        self.netg = define_G(self.opt, norm='batch', use_dropout=False, init_type='normal')
        self.netd = define_D(self.opt, norm='batch', use_sigmoid=False, init_type='normal')

        ##
        if self.opt.resume != '':
            print("\nLoading pre-trained networks.")
            self.opt.iter = torch.load(os.path.join(self.opt.resume, 'netG.pth'))['epoch']
            self.netg.load_state_dict(torch.load(os.path.join(self.opt.resume, 'netG.pth'))['state_dict'])
            self.netd.load_state_dict(torch.load(os.path.join(self.opt.resume, 'netD.pth'))['state_dict'])
            print("\tDone.\n")

        if self.opt.verbose:
            print(self.netg)
            print(self.netd)

        ##
        # Loss Functions
        self.l_adv = nn.BCELoss()
        self.l_con = nn.L1Loss()
        self.l_lat = l2_loss

        ##
        # Initialize input tensors.
        self.input = torch.empty(size=(self.opt.batchsize, 3, self.opt.isize, self.opt.isize), dtype=torch.float32,
                                 device=self.device)
        self.noise = torch.empty(size=(self.opt.batchsize, 3, self.opt.isize, self.opt.isize), dtype=torch.float32,
                                 device=self.device)
        self.label = torch.empty(size=(self.opt.batchsize,), dtype=torch.float32, device=self.device)
        self.gt = torch.empty(size=(opt.batchsize,), dtype=torch.long, device=self.device)
        self.real_label = torch.ones(size=(self.opt.batchsize,), dtype=torch.float32, device=self.device)
        self.fake_label = torch.zeros(size=(self.opt.batchsize,), dtype=torch.float32, device=self.device)

        ##
        # Setup optimizer
        self.optimizer_d = optim.Adam(self.netd.parameters(), lr=self.opt.lr_d, betas=(self.opt.beta1, 0.999))
        self.optimizer_g = optim.Adam(self.netg.parameters(), lr=self.opt.lr_g, betas=(self.opt.beta1, 0.999))

    def forward(self):
        self.forward_g()
        self.forward_d()

    def forward_g(self):
        """ Forward propagate through netG
        """
        self.fake = self.netg(self.input)
        # summary(self.netg, input_size=(3, 64, 64))

    def forward_d(self):
        """ Forward propagate through netD
        """
        self.pred_real, self.feat_real = self.netd(self.input)
        self.pred_fake, self.feat_fake = self.netd(self.fake)

    def backward_g(self):
        """ Backpropagate netg
        """
        self.err_g_adv = self.opt.w_adv * self.l_adv(self.pred_fake, self.real_label[:len(self.pred_fake)])
        self.err_g_con_ssim = self.opt.w_con * (1 - pytorch_ssim.ssim(self.fake, self.input, self.opt.window_train))
        if self.train_which == 'g and d':
            self.err_g = self.err_g_adv + self.err_g_con_ssim
        elif self.train_which == 'g':
            self.err_g = self.err_g_con_ssim
        self.err_g.backward(retain_graph=True)


    def backward_d(self):
        # Fake
        self.err_d_fake = self.l_adv(self.pred_fake, self.fake_label[:len(self.pred_fake)])  # fake_label=0

        # Real
        self.err_d_real = self.l_adv(self.pred_real, self.real_label[:len(self.pred_real)])

        # Combine losses.
        self.err_d = self.err_d_real + self.err_d_fake
        self.err_d.backward(retain_graph=True)

    def update_netg(self):
        """ Update Generator Network.
        """
        self.optimizer_g.zero_grad()
        self.backward_g()
        self.optimizer_g.step()

    def update_netd(self):
        """ Update Discriminator Network.
        """
        self.optimizer_d.zero_grad()
        self.backward_d()
        self.optimizer_d.step()

    ##
    def optimize_params(self):
        """ Optimize netD and netG  networks.
        """
        self.forward()
        self.update_netg()
        self.update_netd()

    ## 保存测试集图片名称
    def get_test_img_names(self):
        img_names = []
        for item in self.data.valid.sampler.data_source.imgs:
            img_path = item[0]
            name = os.path.basename(img_path)
            img_names.append(name)
        return img_names

    ## 获取测试集输出
    def test_forward(self, save_images=False):
        ## 依次存放真实标签、测试集ssim、测试集判别器得分
        gt_labels = []
        ssim = []
        d_real = []
        d_fake = []
        img_names = self.get_test_img_names()

        for i, data in enumerate(self.data.valid, 0):

            self.set_input(data)
            self.fake = self.netg(self.input)
            self.pred_real, _ = self.netd(self.input)
            self.pred_fake, _ = self.netd(self.fake)

            gt_labels = gt_labels + self.gt.tolist()
            ssim = ssim + pytorch_ssim.ssim(self.fake, self.input, self.opt.window_infer, size_average=False).tolist()
            d_real = d_real + self.pred_real.tolist()
            d_fake = d_fake + self.pred_fake.tolist()
            dst_real = './test_images/' + self.opt.dataset + '/' + str(self.repe) + '/real'
            dst_fake = './test_images/' + self.opt.dataset + '/' + str(self.repe) + '/fake'
            if save_images:
                mkdir(dst_real)
                mkdir(dst_fake)
                real, fake = self.get_current_images()

                j = i * self.opt.batchsize
                for img in real:
                    vutils.save_image(img, '%s/%s' % (dst_real, img_names[j]),
                                      normalize=True)
                    j += 1

                j = i * self.opt.batchsize
                for img in fake:
                    vutils.save_image(img, '%s/%s' % (dst_fake, img_names[j]),
                                      normalize=True)
                    j += 1
        return gt_labels, ssim, d_real, d_fake

    ## 获取训练集输出
    def train_forward(self):
        reals_train = []
        ssims_train = []
        for i, data in enumerate(self.data.train_ordered, 0):
            self.set_input(data)
            self.fake = self.netg(self.input)
            self.pred_real, _ = self.netd(self.input)

            ssims_train = ssims_train + pytorch_ssim.ssim(self.fake, self.input, self.opt.window_infer,
                                                          size_average=False).tolist()
            reals_train = reals_train + self.pred_real.tolist()
        return reals_train, ssims_train

    ## 作图
    def draw_curves(self, result_list, split):
        ##训练集
        reals_train = result_list['reals_train']
        ssims_train = result_list['ssims_train']
        ssim_mean = np.mean(ssims_train)
        ssim_std = np.std(ssims_train)
        d_mean = np.mean(reals_train)
        d_std = np.std(reals_train)

        ##测试集
        an_scores = result_list['an_scores']
        ssim = result_list['ssim']
        d_real = result_list['d_real']
        img_names = self.get_test_img_names()
        total = len(img_names)

        curves_path = './curves/' + str(self.repe)
        mkdir(curves_path)
        #########################################################################################
        ###作ssim正常和异常图
        plt.figure(figsize=(20, 15))
        plt.title("ssim and d_real")
        plt.scatter(img_names[:split], ssim[:split], c='b', s=np.pi * 4)
        plt.scatter(img_names[split:], ssim[split:], c='r', s=np.pi * 4)
        d_real_2 = np.zeros(total)
        for i in range(total):
            d_real_2[i] += 1 + d_real[i]
        plt.scatter(img_names[:split], d_real_2[:split], c='g', s=np.pi * 4)
        plt.scatter(img_names[split:], d_real_2[split:], c='orange', s=np.pi * 4)
        plt.axhline(y=ssim_mean, ls='--', c='red')  # 添加水平线
        plt.axhline(y=ssim_mean + ssim_std, ls='--', c='red')  # 添加水平线
        plt.axhline(y=ssim_mean - ssim_std, ls='--', c='red')  # 添加水平线
        plt.axhline(y=ssim_mean + 2 * ssim_std, ls='--', c='red')  # 添加水平线
        plt.axhline(y=ssim_mean - 2 * ssim_std, ls='--', c='red')  # 添加水平线
        plt.axhline(y=d_mean + 1, ls='--', c='blue')  # 添加水平线
        plt.axhline(y=d_mean + d_std + 1, ls='--', c='blue')  # 添加水平线
        plt.axhline(y=d_mean - d_std + 1, ls='--', c='blue')  # 添加水平线
        plt.axhline(y=d_mean + 2 * d_std + 1, ls='--', c='blue')  # 添加水平线
        plt.axhline(y=d_mean - 2 * d_std + 1, ls='--', c='blue')  # 添加水平线
        # plt.grid()
        path1 = curves_path + '/ssim_d_real.png'
        plt.savefig(path1, dpi=120)
        plt.close()

        ##################################################################################
        ###作an_score正常和异常图
        plt.figure(figsize=(20, 10))
        plt.title("an_score")
        plt.scatter(img_names[:split], an_scores[:split], c='b', s=np.pi * 4)
        plt.scatter(img_names[split:], an_scores[split:], c='r', s=np.pi * 4)
        # plt.grid()
        path2 = curves_path + '/score.png'
        plt.savefig(path2, dpi=120)
        plt.close()
        #################################################################################
        ###作ssim和d_real二维图
        plt.figure(figsize=(15, 15))
        plt.title("ssim and d_real")
        plt.scatter(ssim[:split], d_real[:split], c='b', s=np.pi * 4)
        plt.scatter(ssim[split:], d_real[split:], c='r', s=np.pi * 4)

        plt.axvline(x=ssim_mean, ls='--', c='green')  # 添加垂直线
        plt.axvline(x=ssim_mean + ssim_std, ls='--', c='green')  # 添加垂直线
        plt.axvline(x=ssim_mean - ssim_std, ls='--', c='green')  # 添加垂直线
        plt.axvline(x=ssim_mean + 2 * ssim_std, ls='--', c='green')  # 添加垂直线
        plt.axvline(x=ssim_mean - 2 * ssim_std, ls='--', c='green')  # 添加垂直线

        plt.axhline(y=d_mean, ls='--', c='green')  # 添加水平线
        plt.axhline(y=d_mean + d_std, ls='--', c='green')  # 添加水平线
        plt.axhline(y=d_mean - d_std, ls='--', c='green')  # 添加水平线
        plt.axhline(y=d_mean + 2 * d_std, ls='--', c='green')  # 添加水平线
        plt.axhline(y=d_mean - 2 * d_std, ls='--', c='green')  # 添加水平线
        path3 = curves_path + '/Two-dimensional.png'
        plt.savefig(path3, dpi=120)
        plt.close()

    ### 保存测试集每张图片的ssim和d_real
    def save_samples(self, result_list, img_names, total):
        ssim = result_list['ssim']
        d_real = result_list['d_real']
        d_fake = result_list['d_fake']
        an_scores = result_list['an_scores']
        path = './csv_test/' + self.opt.dataset + '/' + str(self.repe) + '/'
        ###保存测试集每张图片的ssim和d_real
        csv_samples = path + 'samples.csv'
        mkdir(path)
        with open(csv_samples, "a", newline='') as file:
            f_csv = csv.writer(file)
            header = ['image', 'ssim', 'd_real', 'd_fake', 'an_score']
            f_csv.writerow(header)
            for i in range(total):
                tmp = []
                tmp.append(img_names[i])
                tmp.append(ssim[i])
                tmp.append(d_real[i])
                tmp.append(d_fake[i])
                tmp.append(an_scores[i])
                f_csv.writerow(tmp)

    ### 保存测试集和训练集的总体指标
    def save_datasets(self, performance):
        csv_path = './csv_test/' + self.opt.dataset
        mkdir(csv_path)
        csv_datasets = csv_path + '/datasets.csv'
        if not os.path.exists(csv_datasets):
            with open(csv_datasets, "a", newline='') as file:
                f_csv = csv.writer(file)
                header = ['dataset', 'AUROC', 'recall', 'precision', 'accuracy',
                          'ssim_test', 'ssim_n', 'ssim_ab', 'd_real_test',
                          'd_real_n', 'd_real_ab', 'ssim_train', 'd_real_train']
                f_csv.writerow(header)

        with open(csv_datasets, "a", newline='') as file:
            f_csv = csv.writer(file)
            tmp = []
            tmp.append(self.opt.dataset)
            tmp.append(str('%.6f' % performance['auroc']))
            tmp.append(str('%.6f' % performance['recall']))
            tmp.append(str('%.6f' % performance['precision']))
            tmp.append(str('%.6f' % performance['accuracy']))
            tmp.append(str('%.6f' % performance['ssim_test_mean']))
            tmp.append(str('%.6f' % performance['normal_ssim']))
            tmp.append(str('%.6f' % performance['abnormal_ssim']))
            tmp.append(str('%.6f' % performance['d_real_test_mean']))
            tmp.append(str('%.6f' % performance['n_d_real']))
            tmp.append(str('%.6f' % performance['ab_d_real']))
            tmp.append(str('%.6f' % performance['ssim_mean']))
            tmp.append(str('%.6f' % performance['d_mean']))
            f_csv.writerow(tmp)

    ##模型判断错误的样本，复制到error文件夹
    def copy_error_images(self, gt_labels, model_pred, img_names):
        total = len(gt_labels)
        dst_real = './test_images/' + self.opt.dataset + '/' + str(self.repe) + '/real'
        dst_fake = './test_images/' + self.opt.dataset + '/' + str(self.repe) + '/fake'
        if self.opt.save_test_images:
            error_real = './test_images/' + self.opt.dataset + '/' + str(self.repe) + '/error_real/'
            error_fake = './test_images/' + self.opt.dataset + '/' + str(self.repe) + '/error_fake/'
            mkdir(error_real)
            mkdir(error_fake)
            for j in range(total):
                img_name = img_names[j]
                if gt_labels[j] != model_pred[j]:
                    shutil.copyfile(dst_real + '/' + img_name, error_real + img_name)
                    shutil.copyfile(dst_fake + '/' + img_name, error_fake + img_name)

    ## 计算测试集指标
    def cal_test(self, save_images=False):
        gt_labels, ssim, d_real, d_fake = self.test_forward(save_images)
        ssim_test_mean = mean(ssim)
        d_real_test_mean = mean(d_real)

        ## 计算测试集中正常样本数量
        split = 0
        for label in gt_labels:
            if label == 0:
                split += 1

        normal_ssim = mean(ssim[:split])
        abnormal_ssim = mean(ssim[split:])
        n_d_real = mean(d_real[:split])
        ab_d_real = mean(d_real[split:])
        ssim_dif = normal_ssim - abnormal_ssim
        real_dif = n_d_real - ab_d_real



        ###设定阈值求其他指标
        reals_train, ssims_train = self.train_forward()
        ssim_mean = np.mean(ssims_train)
        ssim_std = np.std(ssims_train)
        d_mean = np.mean(reals_train)
        d_std = np.std(reals_train)

        ##如果判别器训练不佳，判别器不参与判别
        if d_mean < self.opt.d_th * 0.98 or self.opt.training_mode == 'ae':
            an_scores = (np.array(ssim))
            th = ssim_mean + self.opt.ssim_std_mul * ssim_std
        else:
            an_scores = (np.array(ssim) + np.array(d_real))/2
            th = 0.5 * (ssim_mean + self.opt.ssim_std_mul * ssim_std) + 0.5 * (d_mean + self.opt.d_std_mul * d_std)

        fpr, tpr, _ = roc_curve(gt_labels, 1 - an_scores)
        auroc = auc(fpr, tpr)

        model_pred = []
        for score in an_scores:
            if score > th:
                model_pred.append(0)
            else:
                model_pred.append(1)
        accuracy = accuracy_score(gt_labels, model_pred)
        precision = precision_score(gt_labels, model_pred)
        recall = recall_score(gt_labels, model_pred)

        print('Accuracy = ' + str('%.6f' % accuracy))
        print('Precision = ' + str('%.6f' % precision))
        print('Recall = ' + str('%.6f' % recall))
        performance = OrderedDict([('auroc', auroc), ('ssim_test_mean', ssim_test_mean),
                                   ('normal_ssim', normal_ssim), ('abnormal_ssim', abnormal_ssim),
                                   ('d_real_test_mean', d_real_test_mean), ('n_d_real', n_d_real),
                                   ('ab_d_real', ab_d_real),
                                   ('ssim_dif', ssim_dif),
                                   ('real_dif', real_dif),
                                   ('accuracy', accuracy),
                                   ('precision', precision), ('recall', recall), ('ssim_mean', ssim_mean),
                                   ('d_mean', d_mean)])
        ###返回训练集和测试集逐张图片计算结果的list
        result_list = OrderedDict([('an_scores', an_scores), ('gt_labels', gt_labels), ('ssim', ssim),
                                   ('d_real', d_real), ('d_fake', d_fake), ('reals_train', reals_train),
                                   ('ssims_train', ssims_train), ('model_pred', model_pred)])

        ##
        # RETURN
        return performance, result_list, split

    ### test()用于每个epoch结束后的验证，只需返回各项测试集指标
    def test(self):
        with torch.no_grad():
            print("   Testing %s" % self.name)
            performance, _, _ = self.cal_test()
            return performance

    ###test2()用于训练结束后的验证，需要：
    # 1、保存测试集每张图片的ssim和d_real进samples.csv；
    # 2、保存训练集和测试集的总体指标进dataset.csv；
    # 3、save_test_images；
    def test2(self):
        with torch.no_grad():
            print("   Testing %s" % self.opt.dataset)
            ## 获取测试集图片名称
            img_names = self.get_test_img_names()
            total = len(img_names)

            performance, result_list, split = self.cal_test(save_images=True)

            ###保存测试集每张图片的ssim和d_real
            self.save_samples(result_list, img_names, total)

            ##保存测试集和训练集的总体指标
            self.save_datasets(performance)

            ##模型判断错误的样本，复制到error文件夹
            self.copy_error_images(result_list['gt_labels'], result_list['model_pred'], img_names)

            ## 作图
            self.draw_curves(result_list, split)

    def test20211111(self,item,i):
        with torch.no_grad():
            # Load the weights of netg and netd.
            netG_name = self.opt.netG
            netD_name = self.opt.netD
            self.load_weights(netG_name, netD_name, self.opt.weight_dir)
            print("   Testing %s" % self.opt.dataset)
            print(item)
            print(i)
            out_path = './test_1111/' + item + '/' + str(i)
            mkdir(out_path)

            ## 获取测试集图片名称
            img_names = self.get_test_img_names()

            ## 依次存放真实标签、测试集ssim、测试集判别器得分
            gt_labels = []
            ssim = []
            d_real = []

            for i, data in enumerate(self.data.valid, 0):
                # aaa = data
                # Forward - Pass
                self.set_input(data)
                self.fake = self.netg(self.input)
                self.pred_real, _ = self.netd(self.input)


                gt_labels = gt_labels + self.gt.tolist()
                ssim = ssim + pytorch_ssim.ssim(self.fake, self.input, self.opt.windowsize, size_average=False).tolist()
                d_real = d_real + self.pred_real.tolist()
                d_fake = d_fake + self.pred_fake.tolist()
                dst_real = out_path + '/test_images/real'
                dst_fake = out_path + '/test_images/fake'
                if self.opt.save_test_images:
                    mkdir(dst_real)
                    mkdir(dst_fake)
                    real, fake = self.get_current_images()

                    j = i * self.opt.batchsize
                    for img in real:
                        vutils.save_image(img, '%s/%s' % (dst_real, img_names[j]),
                                          normalize=True)
                        j += 1

                    j = i * self.opt.batchsize
                    for img in fake:
                        vutils.save_image(img, '%s/%s' % (dst_fake, img_names[j]),
                                          normalize=True)
                        j += 1



            ## 计算测试集中正常样本数量
            split = 0
            for label in gt_labels:
                if label == 0:
                    split += 1
            total = len(gt_labels)

            normal_ssim = mean(ssim[:split])
            abnormal_ssim = mean(ssim[split:])
            n_d_real = mean(d_real[:split])
            ab_d_real = mean(d_real[split:])
            ssim_dif = normal_ssim - abnormal_ssim
            real_dif = n_d_real - ab_d_real
            an_scores = np.zeros(total)
            count = 0
            if ssim_dif > 0:
                an_scores += np.ones(total) - ssim
                count += 1
            if real_dif > 0:
                an_scores += np.ones(total) - d_real
                count += 1
            if count > 0:
                an_scores = an_scores / count
                fpr, tpr, _ = roc_curve(gt_labels, an_scores)
                roc_auc = auc(fpr, tpr)
            else:
                roc_auc = 0

            ######################################################################################################################
            ###设定阈值求其他指标,异常样本得分高，因此阈值越高，召回率越低，精确度越高
            reals_train = []
            ssims_train = []
            for i, data in enumerate(self.data.train_ordered, 0):
                self.set_input(data)
                self.fake = self.netg(self.input)
                self.pred_real, _ = self.netd(self.input)

                ssims_train = ssims_train + pytorch_ssim.ssim(self.fake, self.input, self.opt.windowsize, size_average=False).tolist()
                reals_train = reals_train + self.pred_real.tolist()

            ssim_mean = np.mean(ssims_train)
            ssim_std = np.std(ssims_train)
            d_mean = np.mean(reals_train)
            d_std = np.std(reals_train)

            # th = (0.5 * (1 - ssim_mean) + 0.5 * (1 - d_mean)) * 2
            th = 0.5 * (ssim_mean - 2 * ssim_std) + 0.5 * (d_mean - d_std)

            n_score = mean(an_scores[:split])
            ab_score = mean(an_scores[split:])

            model_pred = []
            for score in an_scores:
                if score > th:
                    model_pred.append(0)
                else:
                    model_pred.append(1)
            accuracy = accuracy_score(gt_labels, model_pred)
            precision = precision_score(gt_labels, model_pred)
            recall = recall_score(gt_labels, model_pred)
            performance = OrderedDict([('AUC', roc_auc), ('Normal_ssim', normal_ssim), ('Abnormal_ssim', abnormal_ssim),
                                       ('Normal_d_real', n_d_real), ('Abnormal_d_real', ab_d_real),
                                       ('ssim_dif', ssim_dif),
                                       ('real_dif', real_dif),
                                       ('Accuracy', accuracy),
                                       ('Precision', precision), ('Recall', recall)])
            print(performance)

            csv_path = out_path + '/csv_test/'
            mkdir(csv_path)
            ###保存测试集每张图片的ssim和d_real
            csv_samples = csv_path + '/' + 'samples.csv'
            with open(csv_samples, "a", newline='') as file:
                f_csv = csv.writer(file)
                header = ['image', 'ssim', 'd_real', 'd_fake', 'an_score']
                f_csv.writerow(header)
                for i in range(total):
                    tmp = []
                    tmp.append(img_names[i])
                    tmp.append(ssim[i])
                    tmp.append(d_real[i])
                    tmp.append(d_fake[i])
                    tmp.append(an_scores[i])
                    f_csv.writerow(tmp)

            ##保存测试集和训练集的总体指标
            csv_datasets = './csv_test/datasets.csv'
            if not os.path.exists(csv_datasets):
                with open(csv_datasets, "a", newline='') as file:
                    f_csv = csv.writer(file)
                    header = ['dataset', 'AUROC', 'recall', 'precision', 'accuracy',
                              'd_real_n', 'd_real_ab', 'ssim_train', 'd_real_train']
                    f_csv.writerow(header)

            with open(csv_datasets, "a", newline='') as file:
                f_csv = csv.writer(file)
                tmp = []
                tmp.append(self.opt.dataset)
                tmp.append(str('%.6f' % roc_auc))
                tmp.append(str('%.6f' % recall))
                tmp.append(str('%.6f' % precision))
                tmp.append(str('%.6f' % accuracy))
                tmp.append(str('%.6f' % n_d_real))
                tmp.append(str('%.6f' % ab_d_real))
                tmp.append(str('%.6f' % ssim_mean))
                tmp.append(str('%.6f' % d_mean))
                f_csv.writerow(tmp)

            if self.opt.save_test_images:
                ##异常得分大于ab_score的测试集正常样本，以及异常得分小于n_score的测试集异常样本，复制到error文件夹
                error_path = './test_images/' + self.opt.dataset + '/' + str(self.repe) + '/error/'
                mkdir(error_path)
                for j in range(total):
                    img_name = img_names[j]
                    if (j < split and an_scores[j] > ab_score) or (j >= split and an_scores[j] < n_score):
                        shutil.copyfile(dst_real + '/' + img_name, error_path + img_name)

                ##模型判断错误的样本，复制到error2文件夹
                error2_path = './test_images/' + self.opt.dataset + '/' + str(self.repe) + '/error2/'
                mkdir(error2_path)
                for j in range(total):
                    img_name = img_names[j]
                    if gt_labels[j] != model_pred[j]:
                        shutil.copyfile(dst_real + '/' + img_name, error2_path + img_name)

            curves_path = './curves/' + str(self.repe)
            mkdir(curves_path)
            #########################################################################################
            ###作ssim正常和异常图
            plt.figure(figsize=(20, 15))
            plt.title("ssim and d_real")
            plt.scatter(img_names[:split], ssim[:split], c='b', s=np.pi)
            plt.scatter(img_names[split:], ssim[split:], c='r', s=np.pi)
            d_real_2 = np.zeros(total)
            for i in range(total):
                d_real_2[i] += 1 + d_real[i]
            plt.scatter(img_names[:split], d_real_2[:split], c='b', s=np.pi * 4)
            plt.scatter(img_names[split:], d_real_2[split:], c='r', s=np.pi * 4)
            plt.axhline(y=ssim_mean, ls='--', c='red')  # 添加水平线
            plt.axhline(y=ssim_mean + ssim_std, ls='--', c='red')  # 添加水平线
            plt.axhline(y=ssim_mean - ssim_std, ls='--', c='red')  # 添加水平线
            plt.axhline(y=ssim_mean + 2 * ssim_std, ls='--', c='red')  # 添加水平线
            plt.axhline(y=ssim_mean - 2 * ssim_std, ls='--', c='red')  # 添加水平线
            plt.axhline(y=d_mean + 1, ls='--', c='blue')  # 添加水平线
            plt.axhline(y=d_mean + d_std + 1, ls='--', c='blue')  # 添加水平线
            plt.axhline(y=d_mean - d_std + 1, ls='--', c='blue')  # 添加水平线
            plt.axhline(y=d_mean + 2 * d_std + 1, ls='--', c='blue')  # 添加水平线
            plt.axhline(y=d_mean - 2 * d_std + 1, ls='--', c='blue')  # 添加水平线
            # plt.grid()
            path1 = curves_path + '/ssim_d_real.png'
            plt.savefig(path1, dpi=120)
            plt.close()

            ##################################################################################
            ###作an_score正常和异常图
            plt.figure(figsize=(20, 10))
            plt.title("an_score")
            plt.scatter(img_names[:split], an_scores[:split], c='b', s=np.pi * 4)
            plt.scatter(img_names[split:], an_scores[split:], c='r', s=np.pi * 4)
            # plt.grid()
            path2 = curves_path + '/score.png'
            plt.savefig(path2, dpi=120)
            # plt.show()
            plt.close()
            #################################################################################
            ###作ssim和d_real二维图
            plt.figure(figsize=(15, 15))
            plt.title("ssim and d_real")
            plt.scatter(ssim[:split], d_real[:split], c='b', s=np.pi * 4)
            plt.scatter(ssim[split:], d_real[split:], c='r', s=np.pi * 4)

            plt.axvline(x=ssim_mean, ls='--', c='green')  # 添加垂直线
            plt.axvline(x=ssim_mean + ssim_std, ls='--', c='green')  # 添加垂直线
            plt.axvline(x=ssim_mean - ssim_std, ls='--', c='green')  # 添加垂直线
            plt.axvline(x=ssim_mean + 2 * ssim_std, ls='--', c='green')  # 添加垂直线
            plt.axvline(x=ssim_mean - 2 * ssim_std, ls='--', c='green')  # 添加垂直线

            plt.axhline(y=d_mean, ls='--', c='green')  # 添加水平线
            plt.axhline(y=d_mean + d_std, ls='--', c='green')  # 添加水平线
            plt.axhline(y=d_mean - d_std, ls='--', c='green')  # 添加水平线
            plt.axhline(y=d_mean + 2 * d_std, ls='--', c='green')  # 添加水平线
            plt.axhline(y=d_mean - 2 * d_std, ls='--', c='green')  # 添加水平线
            path3 = curves_path + '/Two-dimensional.png'
            plt.savefig(path3, dpi=120)
            plt.close()
