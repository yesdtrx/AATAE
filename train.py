

from options import Options
from lib.data.dataloader import load_data
from lib.models import load_model

##
def main():
    """ Training
    """
    opt = Options().parse()
    data = load_data(opt)
    for repe in range(0,opt.repetition):
        model = load_model(opt, data)
        model.train(repe + 1)

if __name__ == '__main__':
    main()