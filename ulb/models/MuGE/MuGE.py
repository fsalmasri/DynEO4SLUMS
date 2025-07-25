import numpy as np
import torch
import torchvision

from torch.distributions import Normal, Independent

from ..default_model import default_model
from .UAED_MuGE import Mymodel as net

class MuGE(default_model):

    def __init__(self, opt):
        super().__init__(opt)

        self.net = net(args=opt)

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.opt.lrdecay_iter, gamma=self.opt.lrdecay_gamma)

        self.content_loss = torch.nn.MSELoss(reduction='mean')

        # self.ffl = FocalFrequencyLoss()

        self.initialize_model()

        # dummpy_inpt = torch.FloatTensor(np.zeros((1, 1, 128, 128))).to("cuda:0")
        # self.tb_writer.add_graph(self.de_parallelize_model(), dummpy_inpt)

        self.test_imgs_list = []

    def set_input(self, data):
        self.img = data['lr'].to(self.device)
        self.gt = data['hr'].to(self.device)


    def forward(self):
        mean, std = self.net(self.img)
        # print(mean.shape, std.shape)
        self.outputs_dist = Independent(Normal(loc=mean, scale=std + 0.001), 1)


    def backward(self):
        outputs = self.sample_from_output_dist()

        # bce_loss, mask = cross_entropy_loss_RCF(outputs, self.gt)
        # ffl_loss = self.ffl(outputs, self.gt)
        # print(outputs.shape, self.gt.shape)
        # exit()
        content_loss = self.content_loss(outputs, self.gt)

        self.loss = content_loss #+ ffl_loss
        self.loss.backward()

        self.loss_epoch_train.append(self.loss.item())

    def sample_from_output_dist(self):
        return torch.sigmoid(self.outputs_dist.rsample())

    def test_forward(self):
        with torch.no_grad():
            self.forward()
        outputs = self.sample_from_output_dist()
        self.test_imgs_list.append(self.img[0])
        self.test_imgs_list.append(outputs[0])


    def write_image(self):
        img_grid = torchvision.utils.make_grid(self.test_imgs_list[:30], nrow=2)
        self.tb_writer.add_image('Edges construction test', img_grid, global_step=self.epoch)


    def reset_params(self):
        self.loss_epoch_train = []
        self.test_imgs_list = []


