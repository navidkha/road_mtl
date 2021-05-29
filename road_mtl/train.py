import comet_ml
import gc
from pathlib import Path
from datetime import datetime
import sys
from comet_ml.connection import pending_rpcs_url
import torch.nn as nn

from tasks.visionTask import VisionTask
from utils.imageUtils import *

try:
    sys.path.append(str(Path(".").resolve()))
except:
    raise RuntimeError("Can't append root directory of the project the path")

import numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.swa_utils import AveragedModel, SWALR

from model.net import CustomModel
#from model.data_loader import CustomDataset, CustomDatasetVal
from utils.nn import check_grad_norm, init_weights_normal, EarlyStopping, op_counter
from utils.io import save_checkpoint, load_checkpoint
from utils.utility import get_conf, timeit
import random

class Learner:
    def __init__(self, cfg_dir: str, data_loader:DataLoader, model, labels_definition):
        self.cfg = get_conf(cfg_dir)
        self._labels_definition = labels_definition
        #TODO
        self.logger = self.init_logger(self.cfg.logger)
        #self.dataset = CustomDataset(**self.cfg.dataset)
        self.data = data_loader
        #self.val_dataset = CustomDatasetVal(**self.cfg.val_dataset)
        #self.val_data = DataLoader(self.val_dataset, **self.cfg.dataloader)
        # self.logger.log_parameters({"tr_len": len(self.dataset),
        #                             "val_len": len(self.val_dataset)})
        self.model = model
        #self.model._resnet.conv1.apply(init_weights_normal)
        self.device = self.cfg.train_params.device
        self.model = self.model.to(device=self.device)
        if self.cfg.train_params.optimizer.lower() == "adam":
            self.optimizer = optim.Adam(self.model.parameters(), **self.cfg.adam)
        elif self.cfg.train_params.optimizer.lower() == "rmsprop":
            self.optimizer = optim.RMSprop(self.model.parameters(), **self.cfg.rmsprop)
        else:
            raise ValueError(f"Unknown optimizer {self.cfg.train_params.optimizer}")

        self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
        self.criterion = nn.BCELoss()

        if self.cfg.logger.resume:
            # load checkpoint
            print("Loading checkpoint")
            save_dir = self.cfg.directory.load
            checkpoint = load_checkpoint(save_dir, self.device)
            self.model.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            self.epoch = checkpoint["epoch"]
            self.e_loss = checkpoint["e_loss"]
            self.best = checkpoint["best"]
            print(
                f"{datetime.now():%Y-%m-%d %H:%M:%S} "
                f"Loading checkpoint was successful, start from epoch {self.epoch}"
                f" and loss {self.best}"
            )
        else:
            self.epoch = 1
            self.best = np.inf
            self.e_loss = []

        # initialize the early_stopping object
        self.early_stopping = EarlyStopping(
            patience=self.cfg.train_params.patience,
            verbose=True,
            delta=self.cfg.train_params.early_stopping_delta,
        )

        # stochastic weight averaging
        self.swa_model = AveragedModel(self.model)
        self.swa_scheduler = SWALR(self.optimizer, **self.cfg.SWA)

    def train(self, task:VisionTask):
        task.go_to_gpu(self.device)

        visualize_idx = np.random.randint(0, len(self.data), 50)

        while self.epoch <= self.cfg.train_params.epochs:
            running_loss = []
            self.model.train()

            for internel_iter, (images, gt_boxes, gt_labels, ego_labels, counts, img_indexs, wh) in enumerate(self.data):
                self.optimizer.zero_grad()

                # fl = task.get_flat_label(gt_labels)

                m = nn.Sigmoid()
                y = task.get_flat_label(gt_labels)
                x = images


                # move data to device
                x = x.to(device=self.device)
                y = y.to(device=self.device)

                # forward, backward
                encoded_vector = self.model(x)
                out = task.decode(encoded_vector)
                loss = self.criterion(m(out), y)
                loss.backward()
                # check grad norm for debugging
                grad_norm = check_grad_norm(self.model)
                # update
                self.optimizer.step()

                running_loss.append(loss.item())

                #print("Loss:", loss.item())
                #print("grad_norm", grad_norm)

                self.logger.log_metrics(
                    {
                    #"epoch": self.epoch,
                    "batch": internel_iter,
                    "loss": loss.item(),
                    "GradNorm": grad_norm,
                    },
                    epoch=self.epoch
                )


                #validation
                if internel_iter % 1000 == 0 and self.epoch % 5 == 0:
                    print("Internel iter: ", internel_iter)
                    out = m(out[-1])
                    definitions = []
                    l = task.boundary[1]-task.boundary[0]
                    n_boxes = len(gt_boxes[-1][-1])
                    print("Number of Boxes:", n_boxes)
                    name = "img_" + str(self.epoch) + "_" + str(internel_iter/1000)
                    for i in range (n_boxes):
                        prediction = out[i*l + 1 + i : i*l + l + 1 + i]
                        prediction = prediction.argmax()
                        definitions.append(name + ": " + self._labels_definition[task.get_name()][prediction])
                        
                    print("list", definitions)
                    sz = wh[0][0].item()    
                    img = torch.zeros([3, sz, sz])
                    img[0] = images[-1][self.cfg.dataloader.seq_len -1]
                    img[1] = images[-1][2*self.cfg.dataloader.seq_len - 1]
                    img[2] = images[-1][3*self.cfg.dataloader.seq_len - 1]
                    self.logger.log_image(img, name=name, image_channels='first')


                #if internel_iter < 10:
                #    sz = wh[0][0].item()
                #    img = torch.zeros([3, sz, sz])
                #    print(img.shape)
                #    print(images.shape)
                #    img[0] = images[-1][self.cfg.dataloader.seq_len -1]
                #    img[1] = images[-1][2*self.cfg.dataloader.seq_len - 1]
                #    img[2] = images[-1][3*self.cfg.dataloader.seq_len - 1]
                #    self.log_image_with_text_on_it(img, gt_labels[-1][-1], task)
                    #self.logger.log_image(img, name="v", image_channels='first')

            #bar.close()
            
            # Visualize
            # self.predict_visualize(index_list=visualize_idx, task=task)

            if self.epoch > self.cfg.train_params.swa_start:
                self.swa_model.update_parameters(self.model)
                self.swa_scheduler.step()
            else:
                self.lr_scheduler.step()

            # validate on val set
            # val_loss, t = self.validate()
            # t /= len(self.val_dataset)

            # average loss for an epoch
            self.e_loss.append(np.mean(running_loss))  # epoch loss
            # print(
            #     f"{datetime.now():%Y-%m-%d %H:%M:%S} Epoch {self.epoch} summary: train Loss: {self.e_loss[-1]:.2f} \t| Val loss: {val_loss:.2f}"
            #     f"\t| time: {t:.3f} seconds"
            # )

            self.logger.log_metrics(
                {
                    "epoch": self.epoch,
                    "epoch_loss": self.e_loss[-1],
                }
            )

            # early_stopping needs the validation loss to check if it has decreased,
            # and if it has, it will make a checkpoint of the current model
            #self.early_stopping(val_loss, self.model)

            if self.early_stopping.early_stop:
                print("Early stopping")
                self.save()
                break

            if self.epoch % self.cfg.train_params.save_every == 0:
                self.save()

            gc.collect()
            print("Task: " + task.get_name() + " epoch[" + str(self.epoch) + "] finished.")
            self.epoch += 1

        # Update bn statistics for the swa_model at the end
        #if self.epoch >= self.cfg.train_params.swa_start:
#            torch.optim.swa_utils.update_bn(self.data.to(self.device), self.swa_model)
            #self.save(name=self.cfg.directory.model_name + "-final" + str(self.epoch) + "-swa")

        #macs, params = op_counter(self.model, sample=x)
        #print(macs, params)
        #self.logger.log_metrics({"GFLOPS": macs[:-1], "#Params": params[:-1], "task name": task.get_name(), "total_loss": self.e_loss[-1]})
        print("Training Finished!")
        return loss

    def train_multi(self, primary_task, auxiliary_tasks):

        # 1- got to gpu fo all tasks
        for auxilary_task in auxiliary_tasks:
            auxilary_task.go_to_gpu(self.device)
        primary_task.go_to_gpu(self.device)

        activation_function = nn.Sigmoid()

        while self.epoch <= self.cfg.train_params.epochs:
            running_loss = []
            self.model.train()

            for internel_iter, (images, gt_boxes, gt_labels, ego_labels, counts, img_indexs, wh) in enumerate(
                    self.data):
                self.optimizer.zero_grad()

                x = images
                x = x.to(device=self.device)
                encoded_vector = self.model(x)

                total_loss = None
                # for auxiliary tasks
                for auxiliary_task in auxiliary_tasks:
                    y = auxiliary_task.get_flat_label(gt_labels)
                    # move data to device
                    y = y.to(device=self.device)
                    # forward
                    out = auxiliary_task.decode(encoded_vector)
                    auxiliary_loss = self.criterion(activation_function(out), y)
                    total_loss += auxiliary_loss

                # for primary task
                y = primary_task.get_flat_label(gt_labels)
                # move data to device
                y = y.to(device=self.device)
                # forward
                out = primary_task.decode(encoded_vector)
                primary_loss = self.criterion(activation_function(out), y)
                total_loss += primary_loss


                total_loss.backward()
                # check grad norm for debugging
                grad_norm = check_grad_norm(self.model)
                # update
                self.optimizer.step()

                running_loss.append(primary_loss.item())

                self.logger.log_metrics(
                    {
                        # "epoch": self.epoch,
                        "batch": internel_iter,
                        "primary_loss": primary_loss.item(),
                        "GradNorm": grad_norm,
                    },
                    epoch=self.epoch
                )

                # validation
                if internel_iter % 1000 == 0 and self.epoch % 5 == 0:
                    print("Internel iter: ", internel_iter)
                    out = activation_function(out[-1])
                    definitions = []
                    l = primary_task.boundary[1] - primary_task.boundary[0]
                    n_boxes = len(gt_boxes[-1][-1])
                    print("Number of Boxes:", n_boxes)
                    name = "img_" + str(self.epoch) + "_" + str(internel_iter / 1000)
                    for i in range(n_boxes):
                        prediction = out[i * l + 1 + i: i * l + l + 1 + i]
                        prediction = prediction.argmax()
                        definitions.append(name + ": " + self._labels_definition[primary_task.get_name()][prediction])

                    print("list", definitions)
                    sz = wh[0][0].item()
                    img = torch.zeros([3, sz, sz])
                    img[0] = images[-1][self.cfg.dataloader.seq_len - 1]
                    img[1] = images[-1][2 * self.cfg.dataloader.seq_len - 1]
                    img[2] = images[-1][3 * self.cfg.dataloader.seq_len - 1]

                    draw_text(img, definitions)
                    #self.logger.log_image(img, name=name, image_channels='first')

            # Visualize
            # self.predict_visualize(index_list=visualize_idx, task=task)

            if self.epoch > self.cfg.train_params.swa_start:
                self.swa_model.update_parameters(self.model)
                self.swa_scheduler.step()
            else:
                self.lr_scheduler.step()

            # validate on val set
            # val_loss, t = self.validate()
            # t /= len(self.val_dataset)

            # average loss for an epoch
            self.e_loss.append(np.mean(running_loss))  # epoch loss
            # print(
            #     f"{datetime.now():%Y-%m-%d %H:%M:%S} Epoch {self.epoch} summary: train Loss: {self.e_loss[-1]:.2f} \t| Val loss: {val_loss:.2f}"
            #     f"\t| time: {t:.3f} seconds"
            # )

            self.logger.log_metrics(
                {
                    "epoch": self.epoch,
                    "epoch_loss": self.e_loss[-1],
                }
            )

            # early_stopping needs the validation loss to check if it has decreased,
            # and if it has, it will make a checkpoint of the current model
            # self.early_stopping(val_loss, self.model)

            if self.early_stopping.early_stop:
                print("Early stopping")
                self.save()
                break

            if self.epoch % self.cfg.train_params.save_every == 0:
                self.save()

            gc.collect()
            print("Task: " + primary_task.get_name() + " epoch[" + str(self.epoch) + "] finished.")
            self.epoch += 1

        # Update bn statistics for the swa_model at the end
        # if self.epoch >= self.cfg.train_params.swa_start:
        #            torch.optim.swa_utils.update_bn(self.data.to(self.device), self.swa_model)
        # self.save(name=self.cfg.directory.model_name + "-final" + str(self.epoch) + "-swa")

        # macs, params = op_counter(self.model, sample=x)
        # print(macs, params)
        # self.logger.log_metrics({"GFLOPS": macs[:-1], "#Params": params[:-1], "task name": task.get_name(), "total_loss": self.e_loss[-1]})
        print("Training Finished!")
        return primary_loss

    def predict_visualize(self, index_list, task):
        print("===================================================")
        for i in index_list:
            images, gt_boxes, gt_labels, ego_labels, counts, img_indexs, wh = self.data.dataset.__getitem__(i)
            sz = img_indexs[0]

            y = task.get_flat_label(gt_labels)
            x = images

            # move data to device
            x = x.to(device=self.device)
            y = y.to(device=self.device)

            encoded_vector = self.model(x)
            out = task.decode(encoded_vector)
            self.log_image_with_text(img_tensor=images, out_vector=out, index=i, task=task)
        print("===================================================")

    def log_image_with_text(self, img_tensor, out_vector, index, task):
        definitions = []
        label_len = task.boundary[1]-task.boundary[0]
        name = "img_" + str(index)
        i = 0
        while True:
            finished = out_vector[i]
            if finished == True:
                break
            i += 1

            l = out_vector[i, label_len]
            i += label_len
            if len(np.nonzero(l)) > 0:
                definition_idx = np.nonzero(l)[0][0]
                definitions.append(name + ": " + self._labels_definition[task.get_name()][definition_idx])

        print(definitions)
        self.logger.log_image(img_tensor, name=name, image_channels='first')


    def log_image_with_text_on_it(self, img_tensor, labels, task):
        definitions = []
        box_count = len(labels)
        for j in range(min(box_count, VisionTask._max_box_count)):
            l = labels[j]  # len(l) = 149
            l = l[task.boundary[0]:task.boundary[1]]
            if len(np.nonzero(l)) > 0:
                definition_idx = np.nonzero(l)[0][0]
                definitions.append(self._labels_definition[task.get_name()][definition_idx])

        img = draw_text(img_tensor, definitions)
        print(definitions)
        # print(images.shape)
        self.logger.log_image(img_tensor, name="v", image_channels='first')


    # @timeit
    # @torch.no_grad()
    # def validate(self):
    #
    #     self.model.eval()
    #
    #     running_loss = []
    #
    #     for idx, (x, y) in tqdm(enumerate(self.val_data), desc="Validation"):
    #         # move data to device
    #         x = x.to(device=self.device)
    #         y = y.to(device=self.device)
    #
    #         # forward, backward
    #         if self.epoch > self.cfg.train_params.swa_start:
    #             # Update bn statistics for the swa_model
    #             torch.optim.swa_utils.update_bn(self.data, self.swa_model)
    #             out = self.swa_model(x)
    #         else:
    #             out = self.model(x)
    #
    #         loss = self.criterion(out, y)
    #         running_loss.append(loss.item())
    #
    #     # average loss
    #     loss = np.mean(running_loss)
    #
    #     return loss

    def init_logger(self, cfg):
        logger = None
        # Check to see if there is a key in environment:
        EXPERIMENT_KEY = cfg.experiment_key

        # First, let's see if we continue or start fresh:
        CONTINUE_RUN = cfg.resume
        if (EXPERIMENT_KEY is not None):
            # There is one, but the experiment might not exist yet:
            api = comet_ml.API()  # Assumes API key is set in config/env
            try:
                api_experiment = api.get_experiment_by_id(EXPERIMENT_KEY)
            except Exception:
                api_experiment = None
            if api_experiment is not None:
                CONTINUE_RUN = True
                # We can get the last details logged here, if logged:
                # step = int(api_experiment.get_parameters_summary("batch")["valueCurrent"])
                # epoch = int(api_experiment.get_parameters_summary("epochs")["valueCurrent"])

        if CONTINUE_RUN:
            # 1. Recreate the state of ML system before creating experiment
            # otherwise it could try to log params, graph, etc. again
            # ...
            # 2. Setup the existing experiment to carry on:
            logger = comet_ml.ExistingExperiment(
                previous_experiment=EXPERIMENT_KEY,
                log_env_details=True,  # to continue env logging
                log_env_gpu=True,  # to continue GPU logging
                log_env_cpu=True,  # to continue CPU logging
                auto_histogram_weight_logging=True,
                auto_histogram_gradient_logging=True,
                auto_histogram_activation_logging=True
            )
            # Retrieved from above APIExperiment
            # self.logger.set_epoch(epoch)

        else:
            # 1. Create the experiment first
            #    This will use the COMET_EXPERIMENT_KEY if defined in env.
            #    Otherwise, you could manually set it here. If you don't
            #    set COMET_EXPERIMENT_KEY, the experiment will get a
            #    random key!
            logger = comet_ml.Experiment(
                disabled=cfg.disabled,
                project_name=cfg.project,
                auto_histogram_weight_logging=True,
                auto_histogram_gradient_logging=True,
                auto_histogram_activation_logging=True
            )
            logger.add_tags(cfg.tags.split())
            logger.log_parameters(self.cfg)

        return logger

    def save(self, name=None):
        checkpoint = {"epoch": self.epoch,
                      "model": self.model.state_dict(),
                      "optimizer": self.optimizer.state_dict(),
                      "lr_scheduler": self.lr_scheduler.state_dict(),
                      "best": self.best,
                      "e_loss": self.e_loss
                      }

        if name is None and self.epoch >= self.cfg.train_params.swa_start:
            save_name = self.cfg.directory.model_name + str(self.epoch) + "-swa"
            checkpoint['model-swa'] = self.swa_model.state_dict()

        elif name is None:
            save_name = self.cfg.directory.model_name + str(self.epoch)

        else:
            save_name = name

        if self.e_loss[-1] < self.best:
            self.best = self.e_loss[-1]
            checkpoint["best"] = self.best
            save_checkpoint(
                checkpoint, True, self.cfg.directory.save, save_name
            )
        else:
            save_checkpoint(
                checkpoint, False, self.cfg.directory.save, save_name
            )



