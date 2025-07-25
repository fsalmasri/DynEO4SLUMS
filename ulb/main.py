import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from tqdm import tqdm

from utils.utils import get_opt
from data import create_dataloaders

from models.MuGE.MuGE import MuGE


if __name__ == '__main__':

    opt = get_opt('config.yaml')
    trainLoader, valLoader = create_dataloaders(opt)
    model = MuGE(opt)

    for model.epoch in range(model.epoch + 1, opt.epochs + 1, 1):
        model.reset_params()

        train_bar = tqdm(trainLoader)
        for data in train_bar:
            model.set_input(data)
            model.optimize_parameters()

            train_bar.set_description(f'[TRAIN] {model.epoch}/{opt.epochs} | Loss : {np.mean(model.loss_epoch_train)}')

        # for data in valLoader:
        #     model.set_input(data)
        #     model.test_forward()

        model.write_image()





# hr = Image.fromarray((hr.permute(1, 2, 0).numpy()[:,:,0] * 255.0).astype(np.uint8))
# lr = Image.fromarray((lr.permute(1, 2, 0).numpy()[:,:,0] * 255.0).astype(np.uint8))
# sr = lr.resize((512, 512))
# diff = (np.array(hr)/255.0) - (np.array(sr)/255.0)
#
# plt.imshow(diff, cmap='gray')
# plt.show()
#
# hr.save('hr_rch.png')
# lr.save('lr-rch.png')