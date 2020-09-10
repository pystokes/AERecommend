#!/usr/bin/python3
# -*- coding: utf-8 -*-

from logging import getLogger
import math
import torch
import torch.nn as nn
from torch import optim
from utils.common import CommonUtils
from utils.optimizers import Optimizers

logger = getLogger('ML_ENGINE')

class Trainer(object):

    def __init__(self, model, device, config, save_dir):
        
        self.model = model
        self.device = device
        self.config = config
        self.save_dir = save_dir


    def run(self, train_loader, validate_loader):

        loss_fn = nn.MSELoss()

        optimizer = Optimizers.get_optimizer(self.config.train.optimizer, self.model.parameters())
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, self.config.train.optimizer.T_max)
        
        logger.info('Begin training')
        for epoch in range(1, self.config.train.epoch+1):

            enable_scheduler = (epoch > self.config.train.optimizer.wait_decay_epoch)
            if epoch == self.config.train.optimizer.wait_decay_epoch + 1:
                logger.info(f'Enable learning rate scheduler at Epoch: {epoch:05}')

            # Warm restart
            if enable_scheduler and (epoch % self.config.train.optimizer.T_max == 1):
                for param_group in optimizer.param_groups:
                    param_group['lr'] = self.config.train.optimizer.lr

            train_loss = self._train(loss_fn, optimizer, train_loader)
            valid_loss = self._validate(loss_fn, validate_loader)

            if enable_scheduler:
                scheduler.step()

            logger.info(f'Epoch [{epoch:05}/{self.config.train.epoch:05}], Loss: {train_loss:.5f}, Val Loss: {valid_loss:.5f}')

            if epoch % self.config.train.weight_save_period == 0:
                save_path = self.save_dir.joinpath('weights', f'weight-{str(epoch).zfill(5)}_{train_loss:.5f}_{valid_loss:.5f}.pth')
                CommonUtils.save_weight(self.model, save_path)
                logger.info(f'Saved weight at Epoch : {epoch:05}')


    def _train(self, loss_fn, optimizer, train_data_loader):

        # Keep track of training loss
        train_loss = 0.

        # Train the model in each mini-batch
        self.model.train()
        for mini_batch in train_data_loader:

            # Send data to GPU dvice
            if self.device.type == 'cuda':
                images = mini_batch[1].to(self.device)
            else:
                images = mini_batch[1]

            # Forward
            optimizer.zero_grad()
            outputs = self.model(images)
            loss = loss_fn(outputs, images)

            # Backward and update weights
            loss.backward()
            optimizer.step()

            # Update training loss
            train_loss += loss.item()

        train_loss /= len(train_data_loader.dataset)

        return train_loss


    def _validate(self, loss_fn, validate_data_loader):

        # Keep track of validation loss
        valid_loss = 0.0

        # Not use gradient for inference
        self.model.eval()
        with torch.no_grad():

            # Validate in each mini-batch
            for mini_batch in validate_data_loader:

                # Send data to GPU dvice
                if self.device.type == 'cuda':
                    images = mini_batch[1].to(self.device)
                else:
                    images = mini_batch[1]

                # Forward
                outputs = self.model(images)
                loss = loss_fn(outputs, images)

                # Update validation loss
                valid_loss += loss.item()

        valid_loss /= len(validate_data_loader.dataset)

        return valid_loss
