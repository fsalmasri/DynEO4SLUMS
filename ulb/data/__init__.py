from torch.utils.data import DataLoader

from .dataset import PlanetScope


def create_dataloaders(opt):
    """
    Create a data loaders given the option.
    """

    training_set = PlanetScope(opt, train=True, aug=False)
    validation_set = PlanetScope(opt, train=False, aug=False)

    trainLoader = DataLoader(training_set,
                             batch_size=opt.batchSize,
                             drop_last=False,
                             shuffle=True,
                             num_workers=opt.nThreads,
                             pin_memory=True
                             )

    valLoader = DataLoader(validation_set, batch_size=1, drop_last=False, shuffle=False)

    return trainLoader, valLoader