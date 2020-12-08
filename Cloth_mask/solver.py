
import os
import time
import numpy as np

import torch
from torch import nn, optim
from torch.backends import cudnn
from tensorboardX import SummaryWriter
from torchvision import transforms
import torchvision.utils as vision
from PIL import Image
import matplotlib.pyplot as plt

import data_loader
import network
import misc, metric


class Solver(object):
    def __init__(self, config):
        # Get configuration parameters
        self.config = config

        # Get data loader
        self.image_loader, self.num_images, self.num_steps = dict(), dict(), dict()

        # Model, optimizer, criterion
        self.models, self.optimizers, self.criteria = dict(), dict(), dict()

        # Loss, metric
        self.loss, self.metric = dict(), dict()

        # Training status
        self.phase_types = ["train", "valid"]
        self.lr = self.config.lr_opt["init"]
        self.complete_epochs = 0
        self.best_metric, self.best_epoch = 0, 0

        # Model and loss types
        self.model_types, self.optimizer_types = list(), list()
        self.loss_types, self.metric_types = list(), list()

        # Member variables for data
        self.images, self.labels, self.weights, self.outputs = None, None, None, None
        self.model_types = [self.config.model_type]
        self.optimizer_types = [self.config.model_type]
        self.loss_types = ["bce", "l1"]
        self.metric_types = ["dice"]

        # Tensorboard
        self.tensorboard = None

        # CPU or CUDA
        self.device = torch.device("cuda:%d" % self.config.device_ids[0] if torch.cuda.is_available() else "cpu")
        cudnn.benchmark = True if torch.cuda.is_available() else False

        # Get data loader & build new model or load existing one
        self.load_model()
        self.get_data_loader(data_loader.get_loader)

    def get_data_loader(self, image_loader):
        # Get data loader
        self.image_loader, self.num_images, self.num_steps = dict(), dict(), dict()
        for phase in ["train", "valid", "test"]:
            self.image_loader[phase] = image_loader(dataset_path=self.config.dataset_path,
                                                    num_classes=self.config.num_classes,
                                                    phase=phase,
                                                    shuffle=True,
                                                    patch_size=self.config.patch_size,
                                                    sample_weight=self.config.sample_weight,
                                                    batch_size=self.config.batch_size if phase in ["train", "valid"] else 4,
                                                    num_workers=self.config.num_workers)
            self.num_images[phase] = int(self.image_loader[phase].dataset.__len__())
            self.num_steps[phase] = int(np.ceil(self.num_images[phase] /
                                                (self.config.batch_size if phase in ["train", "valid"] else 4)))

    def build_model(self):
        # Build model
        if self.config.model_type == "UNet":
            self.models[self.model_types[0]] = network.UNet(in_channels=self.config.num_img_ch,
                                                            out_channels=self.config.num_classes,
                                                            num_features=self.config.num_features)
        else:
            raise NotImplementedError("Model type [%s] is not implemented" % self.config.model_type)

        # Build optimizer
        self.optimizers[self.model_types[0]] = optim.Adam(self.models[self.model_types[0]].parameters(),
                                                          lr=self.config.lr_opt["init"],
                                                          betas=(0.9, 0.999),
                                                          weight_decay=self.config.l2_penalty)

        # Build criterion
        self.criteria["bce"] = nn.BCELoss  # Binary cross entropy
        self.criteria["l1"] = nn.L1Loss()  # absolute-value norm (L1 norm)

        # Model initialization
        for model_type in self.model_types:
            self.models[model_type] = network.init_net(self.models[model_type],
                                                       init_type="kaiming", init_gain=0.02,
                                                       device_ids=self.config.device_ids)

    def save_model(self, epoch):
        checkpoint = {"config": self.config,
                      "lr": self.lr,
                      "model_types": self.model_types,
                      "optimizer_types": self.optimizer_types,
                      "loss_types": self.loss_types,
                      "complete_epochs": epoch + 1,
                      "best_metric": self.best_metric,
                      "best_epoch": self.best_epoch}
        model_state_dicts = {"model_%s_state_dict" % model_type:
                             self.models[model_type].state_dict() for model_type in self.model_types}
        optimizer_state_dicts = {"optimizer_%s_state_dict" % optimizer_type:
                                 self.optimizers[optimizer_type].state_dict() for optimizer_type in self.optimizer_types}
        checkpoint = dict(checkpoint, **model_state_dicts)
        checkpoint = dict(checkpoint, **optimizer_state_dicts)
        torch.save(checkpoint, os.path.join(self.config.model_path, "model.pth"))

        print("Best model (%.3f) is saved to %s" % (self.best_metric, self.config.model_path))

    def save_epoch(self, epoch):
        temp = torch.load(os.path.join(self.config.model_path, "model.pth"))
        temp["lr"] = self.lr
        temp["complete_epochs"] = epoch + 1
        torch.save(temp, os.path.join(self.config.model_path, "model.pth"))

        print("")

    def load_model(self):
        if os.path.isfile(os.path.join(self.config.model_path, "model.pth")):
            checkpoint = torch.load(os.path.join(self.config.model_path, self.config.model_name))

            self.config = checkpoint["config"]
            self.lr = checkpoint["lr"]
            self.model_types = checkpoint["model_types"]
            self.optimizer_types = checkpoint["optimizer_types"]
            self.loss_types = checkpoint["loss_types"]
            self.complete_epochs = checkpoint["complete_epochs"]
            self.best_metric = checkpoint["best_metric"]
            self.best_epoch = checkpoint["best_epoch"]

            self.build_model()
            self.load_model_state_dict(checkpoint)
        else:
            self.build_model()

    def load_model_state_dict(self, checkpoint):
        for model_type in self.model_types:
            self.models[model_type].load_state_dict(checkpoint["model_%s_state_dict" % model_type])
        for optimizer_type in self.optimizer_types:
            self.optimizers[optimizer_type].load_state_dict(checkpoint["optimizer_%s_state_dict" % optimizer_type])

    def set_train(self, is_train=True):
        for model_type in self.model_types:
            if is_train:
                self.models[model_type].train(True)
            else:
                self.models[model_type].eval()

    def update_lr(self, epoch, improved=False):
        if self.config.lr_opt["policy"] == "linear":
            self.lr = self.config.lr_opt["init"] / (1.0 + self.config.lr_opt["gamma"] * epoch)
        elif self.config.lr_opt["policy"] == "flat_linear":
            self.lr = self.config.lr_opt["init"]
            if epoch > self.config.lr_opt["step_size"]:
                self.lr /= (1.0 + self.config.lr_opt["gamma"] * (epoch - self.config.lr_opt["step_size"]))
        elif self.config.lr_opt["policy"] == "step":
            self.lr = self.config.lr_opt["init"] * self.config.lr_opt["gamma"] ** \
                 int(epoch / self.config.lr_opt["step_size"])
        elif self.config.lr_opt["policy"] == "plateau":
            if not improved:
                self.config.lr_opt["step"] += 1
                if self.config.lr_opt["step"] >= self.config.lr_opt["step_size"]:
                    self.lr *= self.config.lr_opt["gamma"]
                    self.config.lr_opt["step"] = 0
            else:
                self.config.lr_opt["step"] = 0
        else:
            return NotImplementedError("Learning rate policy [%s] is not implemented", self.config.lr_opt["policy"])

        for optimizer_type in self.optimizer_types:
            for param_group in self.optimizers[optimizer_type].param_groups:
                param_group["lr"] = self.lr

        ending = False if self.lr >= self.config.lr_opt["term"] else True
        return ending

    def print_info(self, phase="train", print_func=None, epoch=0, step=0):
        # Assert
        assert(phase in self.phase_types)

        # Print process information
        total_epoch = self.complete_epochs + self.config.num_epochs
        total_step = self.num_steps[phase]

        prefix = "[Epoch %4d / %4d] lr %.1e" % (epoch, total_epoch, self.lr)
        suffix = "[%s] " % phase
        for loss_type in self.loss_types:
            suffix += "%s: %.5f / " % (loss_type,
                                       sum(self.loss[loss_type][phase]) / max([len(self.loss[loss_type][phase]), 1]))
        for metric_type in self.metric_types:
            suffix += "%s: %.5f / " % (metric_type,
                                       sum(self.metric[metric_type][phase]) / max([len(self.metric[metric_type][phase]), 1]))
        if print_func is not None:
            print_func(step + 1, total_step, prefix=prefix, suffix=suffix, dec=2, bar_len=30)
        else:
            print(suffix, end="")

    def log_to_tensorboard(self, epoch, elapsed_time=None, intermediate_output=None, accuracy=None):
        if elapsed_time is not None:
            self.tensorboard.add_scalar("elapsed_time", elapsed_time, epoch)
        self.tensorboard.add_scalar("learning_rate", self.lr, epoch)
        for loss_type in self.loss_types:
            self.tensorboard.add_scalars("%s" % loss_type, {phase: sum(self.loss[loss_type][phase]) /
                                                                   max([len(self.loss[loss_type][phase]), 1])
                                                            for phase in self.phase_types}, epoch)
        for metric_type in self.metric_types:
            self.tensorboard.add_scalars("%s" % metric_type, {phase: sum(self.metric[metric_type][phase]) /
                                                                     max([len(self.metric[metric_type][phase]), 1])
                                                              for phase in self.phase_types}, epoch)
        if (epoch % 10) == 0:
            if intermediate_output is not None:
                self.tensorboard.add_image("intermediate_output", intermediate_output, epoch)
            if accuracy is not None:
                self.tensorboard.add_scalars("accuracy", {"f-score": accuracy[0],
                                                          "precision": accuracy[1],
                                                          "recall": accuracy[2]}, epoch)

    def forward(self, images, labels, weights):
        # Image to device
        self.images = images.to(self.device)  # n1hw (grayscale)
        self.labels = labels.to(self.device)  # n2hw (binary classification)
        self.weights = weights.to(self.device)  # n1hw?

        # Prediction (forward)
        self.outputs = self.models[self.model_types[0]](self.images)

    def backward(self, phase="train"):
        # Backward to calculate the gradient
        # Loss defition
        bce_loss = self.criteria["bce"](self.weights)(self.outputs, self.labels)
        l1_loss = self.config.l1_weight * self.criteria["l1"](self.outputs, self.labels)

        # Loss integration and gradient calculation (backward)
        loss = bce_loss + l1_loss
        if phase == "train":
            loss.backward()

        self.loss["bce"][phase].append(bce_loss.item())
        self.loss["l1"][phase].append(l1_loss.item())

    def optimize(self, backward):
        """ Optimize and update weights according to the calculated gradients. """
        self.optimizers[self.optimizer_types[0]].zero_grad()
        backward()
        self.optimizers[self.optimizer_types[0]].step()

    def calculate_metric(self, phase="train"):
        self.metric["dice"][phase].append(metric.get_similiarity(self.outputs, self.labels, ch=1))

    def train(self):
        self.tensorboard = SummaryWriter(os.path.join(self.config.model_path, "logs"))
        for epoch in range(self.complete_epochs, self.complete_epochs + self.config.num_epochs):
            # ============================= Training ============================= #
            # ==================================================================== #

            # Training status parameters
            t0 = time.time()
            self.loss = {loss_type: {"train": list(), "valid": list()} for loss_type in self.loss_types}
            self.metric = {metric_type: {"train": list(), "valid": list()} for metric_type in self.metric_types}

            # Image generating for training process
            self.set_train(is_train=True)
            for i, (images, labels, weights) in enumerate(self.image_loader["train"]):
                # Forward
                self.forward(images, labels, weights)

                # Backward & Optimize
                self.optimize(self.backward)

                # Calculate evaluation metrics
                self.calculate_metric()

                # Print training info
                self.print_info(phase="train", print_func=misc.print_progress_bar,
                                epoch=epoch + 1, step=i)

            # ============================ Validation ============================ #
            # ==================================================================== #
            # Image generating for validation process
            with torch.no_grad():
                self.set_train(is_train=False)
                for i, (images, labels, weights) in enumerate(self.image_loader["valid"]):
                    # Forward
                    self.forward(images, labels, weights)

                    # Backward
                    self.backward(phase="valid")

                    # Calculate evaluation metrics
                    self.calculate_metric(phase="valid")

            # Print validation info
            self.print_info(phase="valid")

            # Tensorboard logs
            self.log_to_tensorboard(epoch + 1, elapsed_time=time.time() - t0)

            # ============================ Model Save ============================ #
            # ==================================================================== #
            # Best valiation metric logging
            valid_metric = (sum(self.metric["dice"]["valid"]) / len(self.metric["dice"]["valid"])).item()
            if valid_metric > self.best_metric:
                self.best_metric = valid_metric
                self.best_epoch = epoch + 1

                # Model save
                self.save_model(epoch)
            else:
                # Save current epoch
                self.save_epoch(epoch)

            # Learning rate adjustment
            if self.update_lr(epoch, epoch == (self.best_epoch - 1)):
                print("Model is likely to be fully optimized. Terminating the training...")
                break

        self.tensorboard.close()

    def test(self):
        # Image generating for test process
        self.set_train(is_train=False)
        with torch.no_grad():
            script_dir = os.path.dirname(__file__)
            os.makedirs(script_dir+'/test_results', exist_ok = True)
            os.makedirs(script_dir+'/clothes', exist_ok = True)
            os.makedirs(script_dir+'/mask', exist_ok = True)
            results_dir = os.path.join(script_dir, 'test_results/')
            clothes_dir = os.path.join(script_dir, 'clothes/')
            mask_dir = os.path.join(script_dir, 'mask/')
            for i, (images, labels, weights) in enumerate(self.image_loader["test"]):
                if i == 100:
                    break
                # Image to device
                images = images.to(self.device)  # n1hw (grayscale)
                # Make prediction
                for j in range(4):
                    if j == 0 : 
                        tmp = transforms.ToPILImage()(images[j]).convert('L')
                        tmp = np.array(tmp).astype(np.float32) / 255.0
                        tmp = tmp/np.std(tmp)
                        gray_img = transforms.ToTensor()(tmp).unsqueeze(0)
                        continue
                    tmp = transforms.ToPILImage()(images[j]).convert('L')
                    tmp = np.array(tmp).astype(np.float32) / 255.0
                    tmp = tmp/np.std(tmp)
                    gray = transforms.ToTensor()(tmp).unsqueeze(0)
                    gray_img = torch.cat((gray_img, gray), dim = 0)
                gray_img = gray_img.to(self.device)
                outputs = self.models[self.model_types[0]](gray_img)
                outputs[outputs >=0.7] = 1
                outputs[outputs < 0.7] = 0
                for k in range(4):
                    result = images[k]*outputs[k][1]
                    mask = (outputs[k][1] == 0)
                    result[0][mask] = 1
                    result[1][mask] = 1
                    result[2][mask] = 1
                    vision.save_image(result.detach().cpu(), clothes_dir + 'cloth' + str(i) + '_' + str(k) + '.jpg')
                    vision.save_image(outputs[k][1].detach().cpu(), mask_dir + 'mask' + str(i) + '_' + str(k) + '.jpg')
                    # Image View
                    fig, axes = plt.subplots(1, 4)
                    axes = axes.flatten()
                    axes[0].imshow(images[k].permute(1,2,0).detach().cpu().numpy())
                    axes[1].imshow(labels[k][1].detach().cpu().numpy())
                    axes[2].imshow(outputs[k][1].detach().cpu().numpy())
                    axes[3].imshow(result.permute(1,2,0).detach().cpu().numpy())

                    fig.set_dpi(300)
                    fig.tight_layout()
                    fig.subplots_adjust(hspace=0.01, wspace=0.01)
                    fname = 'result' + str(i) + '_' + str(k) + '.jpg'
                    plt.savefig(results_dir + fname, dpi=200)
                    #plt.show()