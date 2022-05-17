import torch
from pathlib import Path
import argparse

from parse_config import ConfigParser
from data_loader import data_loaders as module_data_loaders
from model.model import get_pretrained_model_by_name
import model.loss as module_loss
import model.metric as module_metric

def main(config):
    # SET DEVICE
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # SET LOGGER
    logger = config.get_logger('test')

    # Must have resume for testing
    no_resume_msg = 'You must provide --resume argument for testing'
    assert config.resume is not None, no_resume_msg
    print("ATTENTION: {}".format(Path(config['data_loader']['args']['data_dir'])))
    # Test DataLoader
    dataloader = getattr(module_data_loaders, config['data_loader']['type'])(
        Path(config['data_loader']['args']['data_dir']).parent / 'test',
        batch_size=64,
        shuffle=False,
        validation_split=0.0,
        num_workers=1,
        training_mode=False
    )

    # Model
    model = get_pretrained_model_by_name(config['arch']['type'])

    # Load checkpoint for testing...
    logger.info('Loading checkpoint: {}'.format(config.resume))
    checkpoint = torch.load(config.resume)
    model.load_state_dict(checkpoint['state_dict'])

    # Loss Function
    loss_fn = getattr(module_loss, config['loss'])

    # Metric Functions
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    # Start Evaluation
    model = model.to(DEVICE)
    model.eval()  # DO NOT FORGET

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))



    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            batch_size = images.shape[0]

            scores = model(images)
            loss = loss_fn(scores, labels)

            total_loss += loss.item() * batch_size
            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(scores, labels) * batch_size
    n_samples = len(dataloader.sampler)

    avg_loss = total_loss / n_samples
    log = {'loss': avg_loss}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    })
    logger.info(log)

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)



