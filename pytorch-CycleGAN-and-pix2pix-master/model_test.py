import torch
from options.train_options import TrainOptions
from models import pix2pix_custom_model as pix

'''Currently Unexecutable'''
if __name__ == '__main__':
    opt = TrainOptions().parse()
    model = pix.Pix2PixCustomModel()
    model.initialize(opt)
    x = torch.randn(3, 256, 256)
    from torchviz import make_dot, make_dot_from_trace
    A,B = model.get_netG()
    g = make_dot(A(x))
    g.view()
