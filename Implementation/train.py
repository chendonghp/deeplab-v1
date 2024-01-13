import itertools, model, utils

from datetime import datetime
import os

import numpy as np
from PIL import Image
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as F
from torch.utils.tensorboard import SummaryWriter


class VGG16_LargeFOV:
    def __init__(
        self,
        num_classes,
        init_weights=False,
        ignore_index=-100,
        gpu_id=0,
        print_freq=10,
        epoch_print=10,
    ):
        self.num_classes = num_classes

        self.ignore_index = ignore_index
        self.gpu = gpu_id
        self.print_freq = print_freq
        self.epoch_print = epoch_print

        torch.cuda.set_device(self.gpu)

        self.loss_function = nn.CrossEntropyLoss(ignore_index=self.ignore_index).cuda(
            self.gpu
        )
        self.model = model.VGG16_LargeFOV(self.num_classes, init_weights).cuda(self.gpu)
        self.optimizer = None
        self.eps = 1e-10
        self.best_mIoU = 0.0
        self.epoch = 0

    # Assuming you have already initialized the model and optimizer
    def load_checkpoint(self, load_path):
        checkpoint = torch.load(load_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.best_mIoU = checkpoint["best_mIoU"]
        self.epoch = checkpoint["epoch"]
        print("Loaded model and optimizer from {}".format(load_path))

    def save_checkpoint(
        self, save_path, loss, epoch, it, model_name="vgg16_large_fov_best"
    ):
        # Save both the model state and the optimizer state
        save_dict = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_mIoU": self.best_mIoU * 100,
            "val_loss": loss,
            "epoch": epoch,
            "iter": it,
            "time": datetime.now(),
        }
        save_path = os.path.join(save_path, f"{model_name}.pt")
        torch.save(save_dict, save_path)
        print(f"Saved Best Model at {save_path}.")

    def save_train_log(
        self, epoch, i, num_batch, loss, test_loss, test_mIoU, save_path=None
    ):
        # Initialize a DataFrame to store the logs
        log_df = pd.DataFrame(
            columns=["epoch", "iteration", "train_loss", "test_loss", "test_mIoU"]
        )
        log_df = log_df.append(
            {
                "epoch": epoch + 1,
                "iteration": i + 1,
                "total_iter": epoch * num_batch + i + 1,
                "train_loss": loss.item(),
                "test_loss": test_loss,
                "test_mIoU": test_mIoU,
            },
            ignore_index=True,
        )
        # Save the DataFrame to a CSV file
        log_csv_path = (
            os.path.join(save_path, "vgg_largefov_training_log.csv")
            if save_path
            else "vgg_largefov_training_log.csv"
        )
        log_df.to_csv(log_csv_path, index=False)
        print(f"Training log saved to {log_csv_path}.")

    def train(
        self,
        train_data,
        test_data,
        load_path=None,
        save_path=None,
        log_path="/content/drive/MyDrive",
        epochs=1,
        lr=0.01,
        momentum=0.9,
        weight_decay=0.0005,
    ):
        # train_data: torch dataset object
        self.optimizer = optim.SGD(
            self.model.parameters(), lr, momentum=momentum, weight_decay=weight_decay
        )
        num_batch = len(train_data)
        if load_path:
            self.load_checkpoint(load_path)
            epochs = epochs - self.epoch
        # set model in training mode, enables the training-specific operations such as dropout and batch normalization
        self.model.train()
        for epoch in range(epochs):
            epoch += self.epoch
            if epoch % self.epoch_print == 0:
                print("Epoch {} Started...".format(epoch + 1))
            for i, (X, y) in enumerate(train_data):
                n, c, h, w = y.shape
                y = y.view(n, h, w).type(torch.LongTensor)

                X, y = X.cuda(self.gpu, non_blocking=True), y.cuda(
                    self.gpu, non_blocking=True
                )
                output = self.model(X)
                output = F.resize(output, (h, w), Image.BILINEAR)

                loss = self.loss_function(output, y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if (i + 1) % self.print_freq == 0:
                    test_mIoU, test_loss = self.test(test_data)
                    self.save_train_log(
                        epoch, i, num_batch, loss, test_loss, test_mIoU, log_path
                    )
                    # Create a SummaryWriter for logging
                    writer = SummaryWriter(log_dir=f'{log_path}/vgg16_largefov')
                    # log scalar values
                    step = epoch * num_batch + i
                    writer.add_scalar("Loss/train", loss.item(), step)
                    writer.add_scalar("Loss/test", test_loss, step)
                    writer.add_scalar("mIoU/test", test_mIoU, step)
                    # writer.add_scalar("Accuracy/test", test_accuracy, step)

                    state = f"Iteration : {i+1} - Train Loss : {loss.item():.6f}, Test Loss : {test_loss:.6f}, Test mIoU : {100 * test_mIoU:.4f}"
                    if test_mIoU > self.best_mIoU:
                        print("\n", "*" * 35, "Best mIoU Updated", "*" * 35)
                        print(state)
                        self.best_mIoU = test_mIoU
                        if save_path:
                            self.save_checkpoint(
                                save_path=save_path,
                                loss=test_loss,
                                epoch=epoch,
                                it=i,
                            )
                        print()
                    else:
                        print(state)
        # Close the SummaryWriter
        writer.close()

    def test(self, test_data):
        tps = torch.zeros(self.num_classes).cuda(self.gpu, non_blocking=True)
        fps = torch.zeros(self.num_classes).cuda(self.gpu, non_blocking=True)
        fns = torch.zeros(self.num_classes).cuda(self.gpu, non_blocking=True)
        losses = list()

        self.model.eval()
        with torch.no_grad():
            for i, (X, y) in enumerate(test_data):
                n, c, h, w = y.shape
                y = y.view(n, h, w).type(torch.LongTensor)

                X, y = X.cuda(self.gpu, non_blocking=True), y.cuda(
                    self.gpu, non_blocking=True
                )
                output = self.model(X)
                output = F.resize(output, (h, w), Image.BILINEAR)

                loss = self.loss_function(output, y)
                losses.append(loss.item())

                tp, fp, fn = utils.mIoU(output, y, self.num_classes, self.gpu)
                tps += tp
                fps += fp
                fns += fn
        self.model.train()
        mIoU = torch.sum(tps / (self.eps + tps + fps + fns)) / self.num_classes
        return (mIoU.item(), sum(losses) / len(losses))
