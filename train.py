import argparse
import collections
import torch
import numpy as np
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils import prepare_device, load_checkpoint, freeze_model
from parse_config import ConfigParser

import data_loader.data_loaders as module_data_loaders

import model.loss as module_loss
import model.metric as module_metric
import model.model as module_model
from model.densenet import DenseNet

from trainer import Trainer

import adabound

# For reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.benchmark = False
import random
random.seed(SEED)
#  torch.cuda.manualseed(SEED)

def main(config):
    # prepare device
    device, device_ids = prepare_device(config['n_gpu'])

    if torch.cuda.is_available():
        print("Running on {}...".format('cuda'))

    logger = config.get_logger('train')

    # setup dataloader instances
    data_loader = config.init_obj('data_loader', module_data_loaders)
    valid_data_loader = data_loader.split_validation()

    # build model architecture, then print to console
    if config['mode'] == 'ensemble_gender':
        pretrained_maker = getattr(module_model, 'PretrainedModel')

        model_gender_resnext50_32x4d = pretrained_maker('gender', 'resnext50_32x4d')()
        model_gender_efficientnet_b1 = pretrained_maker('gender', 'efficientnet_b1')()
        model_gender_regnet_x_32gf = pretrained_maker('gender', 'regnet_x_32gf')()

        model_gender_resnext50_32x4d = model_gender_resnext50_32x4d.to(device)
        model_gender_efficientnet_b1 = model_gender_efficientnet_b1.to(device)
        model_gender_regnet_x_32gf = model_gender_regnet_x_32gf.to(device)

        load_checkpoint(
            torch.load(config['ensemble_gender']['resnext50_32x4d']), model_gender_resnext50_32x4d
        )
        load_checkpoint(
            torch.load(config['ensemble_gender']['efficientnet_b1']), model_gender_efficientnet_b1
        )
        load_checkpoint(
            torch.load(config['ensemble_gender']['regnet_x_32gf']), model_gender_regnet_x_32gf
        )

        freeze_model(model_gender_resnext50_32x4d)
        freeze_model(model_gender_efficientnet_b1)
        freeze_model(model_gender_regnet_x_32gf)

        model = getattr(module_model, 'FinalBinaryEnsembleModel')(
            model_gender_resnext50_32x4d,
            model_gender_efficientnet_b1,
            model_gender_regnet_x_32gf,
            config['ensemble_mask']['num_classes']
        )

        model = model.to(device)
    else:
        model = config.init_obj('arch', module_model)()
        #  model = DenseNet(num_blocks=[6,12,24,16], growth_rate=32)
        model = model.to(device)

    #  logger.info(model)

    # loss_fn and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # optimizer, lr_cheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    #  swa_model = AveragedModel(model)
    #  swa_model = swa_model.to(device)

    #  lr_scheduler = CosineAnnealingLR(optimizer, T_max=100)
    #  swa_start = 5
    #  swa_scheduler = SWALR(optimizer, swa_lr=1e-4)
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(model, loss_fn, metrics, optimizer,
                      config=config,
                      device=device,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      #  lr_scheduler=lr_scheduler,
                      lr_scheduler=lr_scheduler,
                      #  swa_model=swa_model, swa_start=swa_start, swa_scheduler=swa_scheduler
                      )

    trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="All-In-One Template")
    parser.add_argument('-c','--config', default=None, type=str)
    parser.add_argument('-r', '--resume', default=None, type=str)
    parser.add_argument('-d', '--device', default=None, type=str)

    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(flags=['--lr','--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(flags=['--bs','--batch_size'], type=int, target='data_loader;args;batch_size'),
        CustomArgs(flags=['--m','--model_name'], type=str, target='arch;args;model_name'),
        CustomArgs(flags=['--rid','--run_id'], type=str, target='run_id'),
        CustomArgs(flags=['--es','--early_stop'], type=int, target='trainer;early_stop'),
        CustomArgs(flags=['--ld','--lr_decay'], type=float, target='lr_scheduler;args;factor'),
        CustomArgs(flags=['--e','--epoch_num'], type=int, target='trainer;epochs'),
        CustomArgs(flags=['--fn','--fold_num'], type=int, target='data_loader;args;Fold_num'),
        CustomArgs(flags=['--imgS','--img_resize'], type=int, target='data_loader;args;img_resize'),
    ]

    config = ConfigParser.from_args(parser, options)
    main(config)

