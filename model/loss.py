import torch.nn as nn
import torch
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 18

#  class FocalLoss(nn.Module):
#      def __init__(self, weight=None,
#                   gamma=2.0, reduction='mean'):
#          nn.Module.__init__(self)
#          self.weight = torch.tensor([1.0, 1.5, 2.0, 1.0, 1.0, 2.0,
#                                      2.0, 2.0, 3.0, 2.0, 2.0, 3.0,
#                                      2.0, 2.0, 3.0, 2.0, 2.0, 3.0
#                                      ]).to(device)
#          self.gamma = gamma
#          self.reduction = reduction
#          self.weight = weight
#
#      def forward(self, input_tensor, target_tensor, real_age):
#          target_one_hot = F.one_hot(target_tensor, num_classes=6)
#          target_one_hot = target_one_hot.type(torch.DoubleTensor)
#
#          #  age_30_60_classes = [1, 4, 7, 10, 13, 16]
#          #  age_60_classes = [2, 5, 8, 11, 14, 17]
#          age_classes = {
#              4: {
#                  "upper": 60,
#                  "range": 5
#              },
#              3: 55,
#              2: 47,
#              1: 30
#          }
#          age_20_class = 0
#          age_20_30_class = 1
#          age_30_47_class = 2
#          age_47_55_class = 3
#          age_55_60_class = 4
#          age_60_class = 5
#
#          for batch_idx, target_cls in enumerate(target_tensor):
#              if target_cls == 5:
#                  target_one_hot[batch_idx, target_cls] = torch.tensor(0.95)
#                  target_one_hot[batch_idx, target_cls - 1] = torch.tensor(0.05)
#              elif target_cls == 0:
#                  pass
#              else:
#                  offset = age_classes[target_class] - real_age[batch_idx]
#                  upper_smoothing = (0.2) / offset
#                  lower_smoothing = (0.2) - upper_smoothing
#                  target_smoothing = 1.0 - (upper_smoothing + lower_smoothing)
#                  target_one_hot[batch_idx, target_cls] = target_smoothing.clone().detach().requires_grad_(False)
#                  target_one_hot[batch_idx, target_cls + 1] = upper_smoothing.clone().detach().requires_grad_(False)
#                  target_one_hot[batch_idx, target_cls - 1] = lower_smoothing.clone().detach().requires_grad_(False)
#              if True:
#                  pass
#              elif target_cls == age_55_60_class:
#                  offset = 60 - real_age[batch_idx]
#                  smoothing_offset = (0.3) / offset
#                  target_offset = 1.0 - smoothing_offset
#                  target_one_hot[batch_idx, target_cls] = target_offset.clone().detach().requires_grad_(False)
#                  target_one_hot[batch_idx, target_cls + 1] = smoothing_offset.clone().detach().requires_grad_(False)
#
#              elif target_cls == age_47_55_class:
#                  offset = 55 - real_age[batch_idx]
#                  smoothing_offset = (0.3) / offset
#                  lower = 0.2 - smoothing_offset
#                  target_offset = 1.0 - smoothing_offset
#                  target_one_hot[batch_idx, target_cls] = target_offset.clone().detach().requires_grad_(False)
#                  target_one_hot[batch_idx, target_cls + 1] = smoothing_offset.clone().detach().requires_grad_(False)
#                  target_one_hot[batch_idx, target_cls - 1] = lower.clone().detach().requires_grad_(False)
#
#              elif target_cls == age_30_47_class:
#                  offset = 47 - real_age[batch_idx]
#                  smoothing_offset = (0.3) / offset
#                  lower = 0.2 - smoothing_offset
#                  target_offset = 1.0 - smoothing_offset
#                  target_one_hot[batch_idx, target_cls] = target_offset.clone().detach().requires_grad_(False)
#                  target_one_hot[batch_idx, target_cls + 1] = smoothing_offset.clone().detach().requires_grad_(False)
#                  target_one_hot[batch_idx, target_cls - 1] = lower.clone().detach().requires_grad_(False)
#              elif target_cls == age_20_30_class:
#                  offset = 30 - real_age[batch_idx]
#                  smoothing_offset = (0.3) / offset
#                  lower = 0.2 - smoothing_offset
#                  target_offset = 1.0 - smoothing_offset
#                  target_one_hot[batch_idx, target_cls] = target_offset.clone().detach().requires_grad_(False)
#                  target_one_hot[batch_idx, target_cls + 1] = smoothing_offset.clone().detach().requires_grad_(False)
#                  target_one_hot[batch_idx, target_cls - 1] = lower.clone().detach().requires_grad_(False)
#
#
#
#          target_one_hot = target_one_hot.to(device)
#          log_prob = F.log_softmax(input_tensor, dim=-1)
#          prob = torch.exp(log_prob)
#          pred_converted = ((1-prob) ** self.gamma) * log_prob
#
#          nll_loss = -pred_converted * target_one_hot
#          nll_loss = torch.sum(nll_loss, dim=-1)
#          nll_loss = torch.mean(nll_loss)
#          return nll_loss
#
#
#          #  print("After smoothing target_tensor: {}".format(target_tensor))
#          #
#          #  log_prob = F.log_softmax(input_tensor, dim=-1)
#          #  prob = torch.exp(log_prob)
#          #  return F.nll_loss(
#          #      0.25 * ((1 - prob) ** self.gamma) * log_prob,
#          #      target_tensor,
#          #      weight=self.weight,
#          #      reduction=self.reduction
#          #  )
#

class FocalLoss(nn.Module):
    def __init__(self, weight=None,
                 gamma=2., reduction='mean'):
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input_tensor, target_tensor, real_age):
        #  pred = torch.argmax(input_tensor, dim=-1).squeeze()
        #  total_penalty = 0.0
        #  total_class_4 = 0
        #  for i in range(len(target_tensor)):
        #      if pred[i] == 4 and pred[i]==target_tensor[i]:
        #          total_class_4 += 1
        #          offset = 60 - real_age[i]
        #          total_penalty += (1.2 ** offset)
        #
        #  if total_class_4 != 0:
        #      total_penalty /= (2 * total_class_4)
        #  else:
        #      total_penalty = 1

        self.weights = torch.tensor([1.0, 2.0]).to(device)

        log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob,
            target_tensor,
            weight=self.weight,
            reduction=self.reduction
        )


class F1Loss(nn.Module):
    def __init__(self, classes=6, epsilon=1e-7):
        super().__init__()
        self.classes = classes
        self.epsilon = epsilon
    def forward(self, y_pred, y_true):
        assert y_pred.ndim == 2
        assert y_true.ndim == 1
        y_true = F.one_hot(y_true, self.classes).to(torch.float32)
        y_pred = F.softmax(y_pred, dim=1)

        tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        f1 = 2 * (precision * recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1 - self.epsilon)
        return 1 - f1.mean()


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes=18, smoothing=0.2, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


def combined_loss(output, target):
    focal = FocalLoss()(output, target)
    labelsmooth = LabelSmoothingLoss()(output, target)
    f1 = F1Loss()(output,target)

    return 0.8*focal + 0.2*f1

def f1_loss(output, target):
    return F1Loss()(output, target)


def focal_loss(output, target, real_age):
    return FocalLoss()(output, target, real_age)

def cross_entropy_loss(output, target, real_age):
    return nn.CrossEntropyLoss()(output, target)

def weighted_cross_entropy_loss(output, target, real_age):
    #  pred = torch.argmax(output, dim=-1).squeeze()
    #  total_penalty = 0.0
    #  total_class_4 = 0
    #  for i in range(len(target)):
    #      if pred[i] == 4 and pred[i]==target[i]:
    #          total_class_4 += 1
    #          offset = 60 - real_age[i]
    #          total_penalty += (1.2 ** offset)
    #
    #  if total_class_4 != 0:
    #      total_penalty /= (3 * total_class_4)
    #  else:
    #      total_penalty = 1

    #  total_penalty = total_penalty.to(device)

    weights = torch.tensor([1.0, 8.0]).to(device)
    return nn.CrossEntropyLoss(weight=weights)(output, target)

def bce_with_logit_loss(output, target, real_age):
    #  pred = F.sigmoid(output).squeeze()
    #  total_penalty = 0.0
    #  total_class_0 = 0
    #  for i in range(len(target)):
    #      if pred[i] == 0 and pred[i]==target[i]:
    #          total_class_4 += 1
    #          offset = 60 - real_age[i]
    #          total_penalty += (1.2 ** offset)
    #
    #  if total_class_4 != 0:
    #      total_penalty /= (3 * total_class_4)
    #  else:
    #      total_penalty = 1

    target = target.unsqueeze(-1)
    target = target.float()
    return nn.BCEWithLogitsLoss()(output, target)



class FocalLoss_LabelSmoothing(nn.Module):
    def __init__(self, weight=None,
                 gamma=3.5, reduction='mean', smoothing=0.2):
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction
        self.smoothing = smoothing
        self.weight = torch.tensor([1.0, 1.5, 2.0, 1.0, 1.0, 2.0,
                                    2.0, 2.0, 3.0, 2.0, 2.0, 3.0,
                                    2.0, 2.0, 3.0, 2.0, 2.0, 3.0
                                    ]).to(device)

    def forward(self, input_tensor, target_tensor):
        log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)

        # FOCAL LOSS
        focal_loss = F.nll_loss(
            0.25 * ((1 - prob) ** self.gamma) * log_prob,
            target_tensor,
            weight=self.weight,
            reduction=self.reduction
        )

        # Label Smoothing
        C = input_tensor.size(-1)
        #  loss = (-log_prob.sum(dim=-1)).mean()
        final_loss = self.smoothing * (1 / C) + (1 - self.smoothing) * focal_loss
        return final_loss

def focal_loss_label_smoothing(output, target):
    return FocalLoss_LabelSmoothing()(output, target)
