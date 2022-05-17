import torch
from torchmetrics import F1Score
import torch.nn.functional as F
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def accuracy(output, target):
    with torch.no_grad():
        #  print(output)
        #  print(target)
        output, target = output.to(DEVICE), target.to(DEVICE)
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)

def bce_accuracy(output, target):
    with torch.no_grad():
        output = F.sigmoid(output)
        pred = output > 0.5

        pred = pred.squeeze(1)
        #  print("pred shape: {}".format(pred.shape))
        #  print("target shape:{}".format(pred.shape))
        assert pred.shape == target.shape
        correct = torch.sum(pred == target).item()
    return correct / len(target)

def f1_score(output, target):
    output, target = output.to(DEVICE), target.to(DEVICE)
    f1 = F1Score(num_classes=2, average='macro').to(DEVICE)
    score = f1(output, target)
    return score.item()

#  def f1_score(y_pred:torch.Tensor, y_true:torch.Tensor, is_training=True) -> torch.Tensor:
#      '''Calculate F1 score. Can work with gpu tensors
#
#      The original implmentation is written by Michal Haltuf on Kaggle.
#
#      Returns
#      -------
#      torch.Tensor
#          `ndim` == 1. 0 <= val <= 1
#
#      Reference
#      ---------
#      - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
#      - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
#      - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6
#
#      '''
#      assert y_true.ndim == 1
#      assert y_pred.ndim == 1 or y_pred.ndim == 2
#
#      if y_pred.ndim == 2:
#          y_pred = y_pred.argmax(dim=1)
#
#
#      tp = (y_true * y_pred).sum().to(torch.float32)
#      tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
#      fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
#      fn = (y_true * (1 - y_pred)).sum().to(torch.float32)
#
#      epsilon = 1e-7
#
#      precision = tp / (tp + fp + epsilon)
#      recall = tp / (tp + fn + epsilon)
#
#      f1 = 2* (precision*recall) / (precision + recall + epsilon)
#      f1.requires_grad = is_training
#      return f1
