import numpy as np
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
import os
from torch.utils import tensorboard
from pathlib import Path

class default_model(ABC):
    def __init__(self, opt):
        torch.manual_seed(2)
        np.random.seed(2)

        self.epoch = 0
        self.iterval = 0
        self.loss_epoch_train =[]

        self.opt = opt
        self.device = torch.device("cuda:%i" % opt.gpuID if opt.use_gpu else "cpu")

        self.net = None

        if not opt.load:
            opt.log_dir = get_new_logdir(opt.log_dir)

        self.tb_writer = tensorboard.writer.SummaryWriter(log_dir=opt.log_dir)


    def initialize_model(self):
        """Print model parameters after the model is initialized."""
        if self.net is None:
            raise ValueError("Model (`self.net`) must be defined in the child class before calling `initialize_model`.")

        print(f"# Model Parameters: {sum(param.numel() for param in self.net.parameters()):,}")

        self.net = self.net.to(self.device)

        if self.opt.dparallel:
            self.net = nn.DataParallel(self.net)

        if self.opt.load:
            self.opt.log_dir = f'{self.opt.log_dir}{self.opt.load_path}'
            self.load_model()


    @abstractmethod
    def set_input(self, data):
        """This method must be implemented in the child class."""
        pass

    @abstractmethod
    def forward(self):
        """This method must be implemented in the child class."""
        pass

    @abstractmethod
    def backward(self):
        """This method must be implemented in the child class."""
        pass


    def optimize_parameters(self):
        """Performs a training step: forward, backward, and optimizer update."""

        if self.optimizer is None:
            raise ValueError("Optimizer is not defined! Ensure the child class initializes `self.optimizer`.")

        self.net.train()  # Set model to training mode
        self.forward()  # Forward pass
        self.optimizer.zero_grad()  # Clear previous gradients
        self.backward()  # Compute gradients
        self.optimizer.step()  # Update weights

        self.tb_writer.add_scalar('Train', np.mean(self.loss_epoch_train), self.epoch)

    def load_model(self):
        """Load the model, optimizer, and training state from a checkpoint."""
        model_path = f'{self.opt.log_dir}/model_final.pt'
        print(f'Loading model from {model_path}...')

        try:
            saved_dict = torch.load(model_path, map_location=self.device)
            self.net.load_state_dict(saved_dict['model_state_dict'])

            if self.optimizer is not None:  # Ensure optimizer is set before loading state
                self.optimizer.load_state_dict(saved_dict['optimizer'])
            else:
                print("Warning: Optimizer is not defined. Skipping optimizer state loading.")

            self.epoch = saved_dict.get('epoch', 0)  # Default to epoch 0 if not found

            print('Model loaded successfully.')

        except FileNotFoundError:
            print(f"Error: Model file not found at {model_path}. Check the path and try again.")


    def save_model(self, acc):
        """Saves the model, optimizer, and training state to a file."""

        # Define filename based on accuracy and epoch
        fname = f"{self.opt.ver}_{self.epoch}.pt" if acc is None else f"{self.opt.ver}_best.pt"

        save_path = f"{self.opt.log_dir}/{fname}"

        # Ensure model is on CPU and handle DataParallel case
        model_state = self.de_parallelize_model().state_dict()

        checkpoint = {
            'model_state_dict': model_state,
            'optimizer': self.optimizer.state_dict(),
            'epoch': self.epoch,
            'acc': acc
        }

        try:
            torch.save(checkpoint, save_path)
            print(f"Model saved successfully at: {save_path}")
        except Exception as e:
            print(f"Error saving model: {e}")

    def de_parallelize_model(self) -> nn.Module:
        """
        Converts a model wrapped in DataParallel or DistributedDataParallel
        back to a regular nn.Module.

        Args:
            model (nn.Module): The input model (possibly wrapped in parallel wrappers).

        Returns:
            nn.Module: The de-parallelized model.
        """
        if isinstance(self.net, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            return self.net.module

        return self.net

def get_new_logdir(base_dir):
    Path(base_dir).mkdir(parents=True, exist_ok=True)

    folders = [name for name in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, name))]
    numbered_folders = [int(folder) for folder in folders if folder.isdigit()]
    next_number = max(numbered_folders, default=0) + 1

    new_folder = f'{next_number:03}'
    new_folder_path = os.path.join(base_dir, new_folder)

    return new_folder_path


