import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import MetricTracker

class Trainer(BaseTrainer):
    def __init__(self, model, loss_fn, metric_ftns, optimizer, config,
                device, data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None, swa_model=None, swa_start=None, swa_scheduler=None):
        super().__init__(model, loss_fn, metric_ftns, optimizer, config, data_loader)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        #  self.swa_model = swa_model
        #  self.swa_start = swa_start
        #  self.swa_scheduler = swa_scheduler

        self.len_epoch = len(self.data_loader)
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns])
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns])


    def _train_epoch(self, epoch):
        self.model.train() # set to training mode
        self.train_metrics.reset()

        for batch_idx, (data, target, real_age) in enumerate(self.data_loader):
            #############Training Step##############
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            scores = self.model(data)
            loss = self.loss_fn(scores, target, real_age)
            loss.backward()
            self.optimizer.step()

            #  if epoch > self.swa_start:
            #      self.swa_model.update_parameters(self.model)
            #      self.swa_scheduler.step()
            #  else:
            #  self.lr_scheduler.step()

            #  self.swa_model.cpu()
            #  torch.optim.swa_utils.update_bn(self.data_loader, self.swa_model)
            #  self.swa_model.cuda()
            ########################################

            ############Update Metrics##############
            self.train_metrics.update('loss', loss.item())
            #  val_preds = self.swa_model(data)
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(scores, target))
                #  self.train_metrics.update(met.__name__, met(val_preds, target))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} batch: {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()
                ))
            ########################################


        log = self.train_metrics.result()
        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{
                'val_'+k : v for k, v in val_log.items()
            })


        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log

    def _valid_epoch(self, epoch):
        self.model.eval()
        self.valid_metrics.reset()

        with torch.no_grad():
            for batch_idx, (data, target, real_age) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                scores = self.model(data)
                #  scores = self.swa_model(data)
                loss = self.loss_fn(scores, target, real_age)

                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(scores, target))

        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{}]'
        current_batch = batch_idx
        total_batch = len(self.data_loader)
        return base.format(current_batch, total_batch)

