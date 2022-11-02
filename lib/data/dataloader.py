"""
LOAD DATA from file.
"""

# pylint: disable=C0301,E1101,W0622,C0103,R0902,R0915

##
import os,csv
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


class Data:
    """ Dataloader containing train and valid sets.
    """
    def __init__(self, train, valid, train_ordered):
        self.train = train
        self.valid = valid
        self.train_ordered = train_ordered

##
def load_data(opt):
    """ Load Data

    Args:
        opt ([type]): Argument Parser

    Raises:
        IOError: Cannot Load Dataset

    Returns:
        [type]: dataloader
    """

    transform = transforms.Compose([transforms.Resize(opt.isize),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])

    ##
    # LOAD DATA SET
    if opt.datasetroot == 'mvtec':
        opt.dataroot = '/aidata2/niuli/database/mvtec/{}'.format(opt.dataset)
    elif opt.datasetroot == 'cifar10':
        opt.dataroot = '/aidata2/niuli/database/cifar10_ae/{}'.format(opt.dataset)
    elif opt.datasetroot == 'smoke':
        opt.dataroot = '/aidata2/niuli/database/smoke_ae/{}'.format(opt.dataset)
    elif opt.datasetroot == 'mnist':
        opt.dataroot = '/aidata2/niuli/database/mnist_ae/{}'.format(opt.dataset)
        transform = transforms.Compose([transforms.Resize(opt.isize),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.1307,), (0.3081,))])

    train_ds = ImageFolder(os.path.join(opt.dataroot, 'train'), transform)
    train_ordered_ds = ImageFolder(os.path.join(opt.dataroot, 'train'), transform)
    valid_ds = ImageFolder(os.path.join(opt.dataroot, 'test'), transform)


    train_dl = DataLoader(dataset=train_ds, batch_size=opt.batchsize, shuffle=True, drop_last=True, num_workers=opt.workers, pin_memory = True)
    valid_dl = DataLoader(dataset=valid_ds, batch_size=opt.batchsize, shuffle=False, drop_last=False, num_workers=opt.workers, pin_memory = True)
    train_ordered_dl = DataLoader(dataset=train_ordered_ds, batch_size=opt.batchsize, shuffle=False, drop_last=False, num_workers=opt.workers, pin_memory = True)

    return Data(train_dl, valid_dl, train_ordered_dl)
