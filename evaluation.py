import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import read_json, load_checkpoint
import data_loader.data_loaders as module_data_loaders
import pandas as pd
import model.model as module_model
from torch.optim.swa_utils import AveragedModel, SWALR
from multiprocessing import Process, Pool, set_start_method
import torch.multiprocessing as mp
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mapping = {
    0: torch.tensor(0),
    1: torch.tensor(0),
    2: torch.tensor(1),
    3: torch.tensor(1),
    4: torch.tensor(1),
    5: torch.tensor(2)
}
def accuracy(output, target):
    with torch.no_grad():
        output, target = output.to(DEVICE), target.to(DEVICE)
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        for idx in range(len(target)):
            p = int(pred[idx].cpu().clone().detach().numpy())
            t = int(target[idx].cpu().clone().detach().numpy())
            pred[idx] = mapping[p]
            target[idx] = mapping[t]
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


import torch
torch.cuda.empty_cache()


def main():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = read_json('config.json')
    test_config = read_json('test_config.json')
    submission = pd.read_csv(test_config['data_loader']['args']['csv_path'])

    # DataLoader
    dataloader = init_obj(test_config, 'data_loader', module_data_loaders)
    #  valid_data_loader = dataloader.split_validation()

    # Models
    model_maker = getattr(module_model, config['arch']['type'])
    model = model_maker('binary_60', 'densenet161')().to(DEVICE)
    load_checkpoint(torch.load('/opt/ml/code/BaseTemplate/saved/models/Version3/binary_60_1/checkpoint-epoch8.pth'), model, "state_dict")

    model.eval()

    all_predictions_age = []
    cnt = 0
    # Inference
    for batch_idx, (images, labels) in enumerate(dataloader):
        running_acc = 0.0
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        # Age
        scores = F.softmax(model(images))

        for p in range(scores.shape[0]):
            print(cnt,":",scores[p][1].item()*100,"%")
            if scores[p][1] >= 0.05:
                scores[p][1] = 999.
            cnt += 1

        preds = torch.argmax(scores, dim=1).squeeze()

        all_predictions_age.extend(preds.cpu().clone().detach().numpy())

    submission['ans'] = all_predictions_age
    submission.to_csv('./submission_result/submission_binary_60_PROBAB_PRED5.csv')

def init_obj(config, name, module, *args, **kwargs):
    module_name = config[name]['type']
    module_args = dict(config[name]['args'])
    module_args.update(kwargs)
    return getattr(module, module_name)(*args, **module_args)


if __name__ == '__main__':
    main()
