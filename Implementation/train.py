import itertools, model, utils
from pickle import NONE

import tensorboard

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
import asyncio


class VGG16_LargeFOV:
    def __init__(
        self,
        num_classes,
        init_weights=False,
        ignore_index=-100,
        use_gpu=False,
        device=None,
    ):
        self.num_classes = num_classes
        self.use_gpu = use_gpu
        self.device = device
        self.ignore_index = ignore_index
        self.loss_function = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        self.model = model.VGG16_LargeFOV(self.num_classes, init_weights)
        if self.use_gpu:
            self.model.to(self.device)
            self.loss_function = self.loss_function.to(self.device)
        self.optimizer = None
        self.eps = 1e-10
        self.best_mIoU = 0.0
        self.epoch = 0

    # Assuming you have already initialized the model and optimizer
    def load_checkpoint(self, load_path, model_name="vgg16_large_fov_latest"):
        checkpoint = torch.load(os.path.join(load_path, f"{model_name}.pt"))
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.best_mIoU = checkpoint["best_mIoU"]
        self.epoch = checkpoint["epoch"]
        print("Loaded model and optimizer from {}".format(load_path))

    async def save_checkpoint(
        self, save_path, loss, mIoU, mpa, epoch, model_name="vgg16_large_fov_best"
    ):
        # Save both the model state and the optimizer state
        save_dict = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_mIoU": mIoU * 100,
            "mpa": mpa * 100,
            "val_loss": loss,
            "epoch": epoch + 1,
            "time": datetime.now(),
        }
        save_path = os.path.join(save_path, f"{model_name}.pt")
        await asyncio.to_thread(torch.save, save_dict, save_path)
        print(f"Saved Model at {save_path}.")

    async def save_train_log(
        self, epoch, loss, test_loss, test_mIoU, test_mpa, save_path=None
    ):
        log_csv_path = (
            os.path.join(save_path, "vgg_largefov_training_log.csv")
            if save_path
            else "vgg_largefov_training_log.csv"
        )
        if os.path.exists(log_csv_path):
            log_df = await asyncio.to_thread(pd.read_csv, log_csv_path)
        else:
            log_df = pd.DataFrame(
                columns=[
                    "epoch",
                    "train_loss",
                    "test_loss",
                    "test_mIoU",
                    "test_mpa",
                ]
            )
        new_row = pd.DataFrame(
            {
                "epoch": [epoch],
                "train_loss": [loss.item()],
                "test_loss": [test_loss],
                "test_mIoU": [test_mIoU],
                "test_mpa": [test_mpa],
            }
        )

        log_df = pd.concat([log_df, new_row], ignore_index=True)

        # Save the DataFrame to a CSV file
        await asyncio.to_thread(log_df.to_csv, log_csv_path, index=False)
        print(f"Training log saved to {log_csv_path}.")

    def tensorboard_log(self, writer, step, loss, test_loss, test_mIoU, test_mpa):
        # log scalar values
        writer.add_scalar("Loss/train", loss.item(), step)
        writer.add_scalar("Loss/test", test_loss, step)
        writer.add_scalar("mIoU/test", test_mIoU, step)
        writer.add_scalar("mpa/test", test_mpa, step)
        # writer.add_scalar("Accuracy/test", test_accuracy, step)
        return writer

    async def train(
        self,
        train_loader,
        test_loader,
        load_path=None,
        save_path=None,
        log_path="/content/drive/MyDrive",
        epochs=1,
        test_freq=10,
        lr=0.01,
        momentum=0.9,
        weight_decay=0.0005,
        nesterov=False,
        create_scheduler=None,
    ):
        # train_data: torch dataset object
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
        scheduler = None
        if create_scheduler:
            scheduler = create_scheduler(self.optimizer)
        if load_path:
            self.load_checkpoint(load_path)
            epochs = epochs - self.epoch
        # set model in training mode, enables the training-specific operations such as dropout and batch normalization
        self.model.train()
        print("Train Started: ", "\n")
        for epoch in range(epochs):
            epoch += self.epoch
            test_mIoU, test_loss = 0, 0
            for i, (X, y) in enumerate(train_loader):
                n, c, h, w = y.shape
                y = y.view(n, h, w).type(torch.LongTensor)
                if self.use_gpu:
                    X, y = X.to(self.device, non_blocking=True), y.to(
                        self.device, non_blocking=True
                    )
                output = self.model(X)
                output = F.resize(output, (h, w), Image.BILINEAR)
                loss = self.loss_function(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if i % test_freq == 0:
                    test_mIoU, test_loss, test_mpa = self.test(test_loader)
                    scheduler.step(test_loss)
                    print(
                        f"Learning rate now at {self.optimizer.param_groups[0]['lr']}."
                    )
                    writer = SummaryWriter(log_dir=f"{log_path}/runs")
                    writer = self.tensorboard_log(
                        writer,
                        epoch * len(train_loader) + i,
                        loss,
                        test_loss,
                        test_mIoU,
                        test_mpa,
                    )
                    state = f"Epoch : {epoch} Iter : {i} - Train Loss : {loss.item():.6f}, Test Loss : {test_loss:.6f}, Test mIoU : {100 * test_mIoU:.4f}, Test mpa : {100 * test_mpa:.4f}"
                    print(state)

                    await self.save_train_log(
                        epoch * len(train_loader) + i,
                        loss,
                        test_loss,
                        test_mIoU,
                        test_mpa,
                        log_path,
                    )

            if test_mIoU > self.best_mIoU:
                print("\n", "*" * 35, "Best mIoU Updated", "*" * 35)
                print(state)
                self.best_mIoU = test_mIoU
                await self.save_checkpoint(
                    save_path=save_path,
                    loss=test_loss,
                    mIoU=test_mIoU,
                    mpa=test_mpa,
                    epoch=epoch,
                )
                print()
            await self.save_checkpoint(
                save_path=save_path,
                loss=test_loss,
                mIoU=test_mIoU,
                mpa=test_mpa,
                epoch=epoch,
                model_name="vgg16_large_fov_latest",
            )
        # Close the SummaryWriter
        writer.close()

    def test(self, test_data):
        tps = torch.zeros(self.num_classes)
        fps = torch.zeros(self.num_classes)
        fns = torch.zeros(self.num_classes)
        if self.use_gpu:
            tps = tps.to(self.device)
            fps = fps.to(self.device)
            fns = fns.to(self.device)
        losses = []
        avg_miou = []
        avg_mpa = []
        self.model.eval()
        with torch.no_grad():
            for i, (X, y) in enumerate(test_data):
                n, c, h, w = y.shape
                y = y.view(n, h, w).type(torch.LongTensor)
                if self.use_gpu:
                    X, y = X.to(self.device, non_blocking=True), y.to(
                        self.device, non_blocking=True
                    )
                output = self.model(X)
                output = F.resize(output, (h, w), Image.BILINEAR)

                loss = self.loss_function(output, y)
                losses.append(loss.item())

                tp, fp, fn = utils.metrics(
                    output, y, self.num_classes, self.use_gpu, self.device
                )
                miou = torch.mean(tp / (self.eps + tp + fp + fn))
                avg_miou.append(miou.item())
                mpa = torch.mean(tp / (self.eps + tp + fn))
                avg_mpa.append(mpa.item())
                # tps += tp
                # fps += fp
                # fns += fn
        # self.model.train()
        # mIoU = torch.sum(tps / (self.eps + tps + fps + fns)) / self.num_classes
        iou, loss, pa = [sum(v) / len(v) for v in [avg_miou, losses, avg_mpa]]
        return iou, loss, pa
