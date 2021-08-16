import gc
import sys
from datetime import datetime
from pathlib import Path

import comet_ml
import torch.nn as nn

from tasks.visionTask import VisionTask
from utils.imageUtils import *

from utils.utility import timeit
from torchvision import models
import copy
import os
import time
from model.multi_output_classification import MultiOutputClassification

try:
    sys.path.append(str(Path(".").resolve()))
except:
    raise RuntimeError("Can't append root directory of the project the path")

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.swa_utils import AveragedModel

# from model.data_loader import CustomDataset, CustomDatasetVal
from utils.nn import check_grad_norm, EarlyStopping
from utils.io import save_checkpoint, load_checkpoint
from utils.utility import get_conf
from model.basics import EfficientConvBlock


class Learner:
    def __init__(self, cfg_dir: str, data_loader_train: DataLoader, data_loader_val: DataLoader, task: VisionTask,
                 labels_definition):
        self.cfg = get_conf(cfg_dir)
        self._labels_definition = labels_definition
        self.logger = self.init_logger(self.cfg.logger)
        self.task = task
        self.data_loaders = {'train': data_loader_train, 'val': data_loader_val}
        self.device = self.cfg.train_params.device
        self.model = MultiOutputClassification(task)
        self.model = self.model.to(device=self.device)
        # self.optimizer = optim.Adam(self.model.fc.parameters(), **self.cfg.adam)
        self.optimizer = optim.Adam(self.model.fc.parameters(), lr=0.001)
        self.criterion_class = nn.CrossEntropyLoss()
        self.criterion_box = nn.SmoothL1Loss()
        # self.criterion_lbl = nn.BCELoss()
        # self.criterion_box = nn.MSELoss()
        # self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
        self.sig = nn.Sigmoid()

        if self.cfg.logger.resume:
            # load checkpoint
            print("Loading checkpoint")
            save_dir = self.cfg.directory.load
            checkpoint = load_checkpoint(save_dir, self.device)
            self.model.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            # self.scheduler.load_state_dict(checkpoint["lr_scheduler"])
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

    def train(self):
        since = time.time()
        best_acc = 0.0
        best_model_wts = copy.deepcopy(self.model.state_dict())
        while self.epoch <= self.cfg.train_params.epochs:
            running_loss = []
            self.logger.set_epoch(self.epoch)
            best_acc = 0.0

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for internel_iter, (x, gt_boxes, gt_labels, ego_labels, counts, img_indexs, wh) \
                        in enumerate(self.data_loaders[phase]):

                    input = x.to(self.device)
                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        out_list_class, out_list_box = self.model(input)
                        loss = None
                        for i in range(len(out_list_class)):
                            loss += self.criterion_class(out_list_class[0], out_list_class[1])
                            loss += self.criterion_box(out_list_box[0], out_list_box[1])

                        # preds = self.refine_out(outputs)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            # print("----------------------------------------------")
                            # print(outputs)
                            # print(labels)

                            self.optimizer.zero_grad()
                            loss.backward()
                            self.optimizer.step()

                        # visualize
                        if phase == 'val' and internel_iter % 20 == 0:
                            img_name = "img_" + str(self.epoch).zfill(2) + "_" + str(internel_iter).zfill(3)
                            self.visualize(images=x,
                                           out_list_class=out_list_class,
                                           out_list_box=out_list_box,
                                           task=self.task,
                                           img_name=img_name,
                                           img_size=wh[0][0].item())

                    self.logger.log_metrics(
                        {
                            # "epoch": self.epoch,
                            # "batch": internel_iter,
                            "loss": loss.item() * input.size(0)
                        },
                        epoch=self.epoch
                    )

                    # statistics
                    running_loss += loss.item() * input.size(0)
                    # print("----------------------------------")
                    # print(labels.shape)
                    # print(preds.shape)
                    # print(labels.data)
                    # running_corrects += torch.equal(preds, labels.data)

                # if phase == 'train':
                #    self.scheduler.step()

                epoch_loss = running_loss / len(self.data_loaders[phase])
                # epoch_acc = running_corrects.double() / len(self.data_loaders[phase])
                # epoch_acc = float(running_corrects) / len(self.data_loaders[phase])

                # print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                #     phase, epoch_loss, epoch_acc))
                #
                # # deep copy the model
                # if phase == 'val' and epoch_acc > best_acc:
                #     best_acc = epoch_acc
                #     best_model_wts = copy.deepcopy(self.model.state_dict())
            print()
            self.epoch += 1

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        self.model.load_state_dict(best_model_wts)

    def refine_out(self, out_orig):
        # print("---------------------------------")
        # print(out_orig.size(0))
        out = torch.clone(out_orig)
        for i in range(out.size(0)):
            idx = 0
            class_len = self.task.get_num_classes()
            while idx < out.shape[1]:
                if out[i][idx] > 0.5:
                    out[i][idx] = 1
                else:
                    out[i][idx] = 0
                idx += 1

                temp = out[i][idx:idx + class_len]
                mx = torch.max(temp)
                max_idx = torch.argmax(temp)
                temp[temp > 0] = 0
                temp[max_idx] = 1

                idx += class_len
        return out

    def train_multi(self, auxiliary_tasks):

        # 1- go to gpu fo all tasks
        for auxilary_task in auxiliary_tasks:
            auxilary_task.go_to_gpu(self.device)
        self.task.go_to_gpu(self.device)

        while self.epoch <= self.cfg.train_params.epochs:
            running_loss = []
            self.model.train()

            for internel_iter, (x, gt_boxes, gt_labels, ego_labels, counts, img_indexs, wh) in enumerate(self.data_val):

                x = x.to(device=self.device)
                encoded_vector = self.model(x)

                total_loss = None
                # for auxiliary tasks
                for auxiliary_task in auxiliary_tasks:
                    y_lbl = auxiliary_task.get_flat_label(gt_labels)
                    y_box = auxiliary_task.get_flat_boxes(gt_boxes)
                    # move data to device
                    y_lbl = y_lbl.to(device=self.device)
                    y_box = y_box.to(device=self.device)
                    # forward
                    out = auxiliary_task.decode(encoded_vector)
                    auxiliary_loss_lbl = self.criterion_class(self.sig(out[:, :-VisionTask._output_box_max_size]),
                                                              y_lbl)
                    auxiliary_loss_box = self.criterion_box(out[:, -VisionTask._output_box_max_size:], y_box)
                    auxiliary_loss = auxiliary_loss_lbl + auxiliary_loss_box
                    if total_loss is None:
                        total_loss = auxiliary_loss
                    else:
                        total_loss += auxiliary_loss

                # for primary task
                y_lbl = self.task.get_flat_label(gt_labels)
                y_box = self.task.get_flat_boxes(gt_boxes)
                # move data to device
                y_lbl = y_lbl.to(device=self.device)
                y_box = y_box.to(device=self.device)
                # forward
                out = self.task.decode(encoded_vector)
                primary_loss_lbl = self.criterion_class(self.sig(out[:, :-VisionTask._output_box_max_size]), y_lbl)
                primary_loss_box = self.criterion_box(out[:, -VisionTask._output_box_max_size:], y_box)
                primary_loss = primary_loss_lbl + primary_loss_box
                total_loss += primary_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                # check grad norm for debugging
                grad_norm = check_grad_norm(self.model)
                grad_norm_decoder = check_grad_norm(self.decoder)
                # update
                self.optimizer.step()

                running_loss.append(primary_loss.item())

                self.logger.log_metrics(
                    {
                        # "epoch": self.epoch,
                        # "batch": internel_iter,
                        "primary_loss": primary_loss.item(),
                        "GradNormEncoder": grad_norm,
                        "GradNormDecoder": grad_norm_decoder,
                    },
                    epoch=self.epoch
                )

            self.lr_scheduler.step()

            # validate on val set
            val_loss = self.validate_multi(auxiliary_tasks)
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

            # if self.early_stopping.early_stop:
            #     print("Early stopping")
            #     self.save()
            #     break

            if self.epoch % self.cfg.train_params.save_every == 0:
                self.save()

            gc.collect()
            # print("Task: " + self.task.get_name() + ", Auxiliary: " + auxiliary_tasks[0].get_name() + " epoch[" + str(self.epoch) + "] finished.")
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

    def visualize_lbl(self, images, labels, task, output, img_name, img_size):

        box_size = 4
        # 1- prepare image
        img = torch.zeros([3, img_size, img_size])
        img[0] = images[-1][self.cfg.dataloader.seq_len - 1]
        img[1] = images[-1][2 * self.cfg.dataloader.seq_len - 1]
        img[2] = images[-1][3 * self.cfg.dataloader.seq_len - 1]

        # 2- find predictions definitions
        out = output[-1]
        definitions_pred = []

        l = task.boundary[1] - task.boundary[0]
        index = 0
        rep_count = 0
        skip_indexes = []
        while rep_count < VisionTask._max_box_count:
            # read flag
            go_on = out[index]
            index += 1
            if go_on == 0:
                index += l
                skip_indexes.append(rep_count)
                rep_count += 1
                continue
                # read label prediction
            prediction = out[index:index + l]
            pred_index = prediction.argmax()
            index += l
            definitions_pred.append(self._labels_definition[task.get_name()][pred_index])
            rep_count += 1

        # print("prediction size: ", len(definitions_pred))

        # 3- draw labels definitions
        definitions_lbl = []
        box_count = len(labels[-1][-1])
        for j in range(min(box_count, VisionTask._max_box_count)):
            l = labels[-1][-1][j]  # len(l) = 149
            if l[0] == 0:
                break
            l = l[task.boundary[0]:task.boundary[1]]
            label_index = l.argmax()
            definitions_lbl.append(self._labels_definition[task.get_name()][label_index])

        # print("label size: ", len(definitions_lbl))

        # 4- draw both definitions
        img_with_text = draw_text_box2(img_tensor=img, text_list_pred=definitions_pred, text_list_lbl=definitions_lbl)
        self.logger.log_image(image_data=img_with_text, name=img_name, image_channels='first')

    def visualize(self, images, out_list_class, out_list_box, task, img_name, img_size):

        box_size = 4
        # 1- prepare image
        img = torch.zeros([3, img_size, img_size])
        img[0] = images[-1][self.cfg.dataloader.seq_len - 1]
        img[1] = images[-1][2 * self.cfg.dataloader.seq_len - 1]
        img[2] = images[-1][3 * self.cfg.dataloader.seq_len - 1]

        class_lbl = []
        class_pred = []
        boxes_real = []
        boxes_pred = []

        for out_class_lbl in out_list_class:
            class_pred_index = out_class_lbl[0].argmax()
            class_lbl_index = out_class_lbl[1].argmax()
            class_pred.append(self._labels_definition[task.get_name()][class_pred_index])
            class_lbl.append(self._labels_definition[task.get_name()][class_lbl_index])

        for out_box_predbox in out_list_box:
            boxes_pred.append(out_box_predbox[0] * 224.)
            boxes_real.append(out_box_predbox[1] * 224.)

        # 4- draw both definitions
        img_with_text = draw_text_box(img_tensor=img, text_list_pred=class_pred, text_list_lbl=class_lbl
                                      , box_list_lbl=boxes_real, box_list_pred=boxes_pred)
        self.logger.log_image(image_data=img_with_text, name=img_name, image_channels='first')

    @timeit
    @torch.no_grad()
    def validate(self):

        self.model.eval()

        running_loss = []

        for internel_iter, (x, gt_boxes, gt_labels, ego_labels, counts, img_indexs, wh) in enumerate(self.data_val):
            y_lbl = self.task.get_flat_label(labels=gt_labels)
            y_box = self.task.get_flat_boxes(boxes=gt_boxes)

            # move data to device
            x = x.to(device=self.device)
            y_lbl = y_lbl.to(device=self.device)
            y_box = y_box.to(device=self.device)

            # forward, backward
            encoded_vector = self.model(x)
            out = self.decoder(encoded_vector)

            loss_lbl = self.criterion_class(self.sig(out[:, :-VisionTask._output_box_max_size]), y_lbl)
            loss_box = self.criterion_box(out[:, -VisionTask._output_box_max_size:], y_box)
            loss = loss_lbl + loss_box
            running_loss.append(loss.item())

            img_name = "img_" + str(self.epoch)
            self.visualize(images=x, labels=gt_labels, boxes=gt_boxes, task=self.task,
                           output=out, img_name=img_name, img_size=wh[0][0].item())

        # average loss
        loss = np.mean(running_loss)

        return loss

    @timeit
    @torch.no_grad()
    def validate_multi(self, auxiliary_tasks):

        self.model.eval()

        running_loss = []

        for internel_iter, (x, gt_boxes, gt_labels, ego_labels, counts, img_indexs, wh) in enumerate(self.data_val):

            x = x.to(device=self.device)
            encoded_vector = self.model(x)

            total_loss = None
            # for auxiliary tasks
            for auxiliary_task in auxiliary_tasks:
                y_lbl = auxiliary_task.get_flat_label(gt_labels)
                y_box = auxiliary_task.get_flat_boxes(gt_boxes)
                # move data to device
                y_lbl = y_lbl.to(device=self.device)
                y_box = y_box.to(device=self.device)
                # forward
                out = auxiliary_task.decode(encoded_vector)
                auxiliary_loss_lbl = self.criterion_class(self.sig(out[:, :-VisionTask._output_box_max_size]), y_lbl)
                auxiliary_loss_box = self.criterion_box(out[:, -VisionTask._output_box_max_size:], y_box)
                auxiliary_loss = auxiliary_loss_lbl + auxiliary_loss_box
                if total_loss is None:
                    total_loss = auxiliary_loss
                else:
                    total_loss += auxiliary_loss

            # for primary task
            y_lbl = self.task.get_flat_label(gt_labels)
            y_box = self.task.get_flat_boxes(gt_boxes)
            # move data to device
            y_lbl = y_lbl.to(device=self.device)
            y_box = y_box.to(device=self.device)
            # forward
            out = self.task.decode(encoded_vector)
            primary_loss_lbl = self.criterion_class(self.sig(out[:, :-VisionTask._output_box_max_size]), y_lbl)
            primary_loss_box = self.criterion_box(out[:, -VisionTask._output_box_max_size:], y_box)
            primary_loss = primary_loss_lbl + primary_loss_box
            total_loss += primary_loss

            running_loss.append(primary_loss.item())

            img_name = "img_" + str(self.epoch)
            self.visualize(images=x, labels=gt_labels, boxes=gt_boxes, task=self.task,
                           output=out, img_name=img_name, img_size=wh[0][0].item())

        return primary_loss

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
