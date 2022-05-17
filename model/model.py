import torch
import torch.nn as nn
from torchvision import models
from functools import reduce

class PretrainedModel:
    def __init__(self, classifier_target, model_name):
        '''
        Returns a pretrained model provided by model_name with the appropriate number of output classes.

        Parameters:
            classifier_target
                'all': 18 output classes - Gender, Age, Mask
                'gender': 2 output classes (male/female)
                'age': 3 output classes (<30/<30<=60/>60)
                'mask': 3 output classes (wear/incorrect/not wear),
                'age_6': 6 output classes
        '''
        self.classifier_target = classifier_target
        self.model_name = model_name

        model = getattr(models, model_name)(pretrained=True)

        num_classes_match = {
            'all': 18,
            'gender': 1,
            'age': 3,
            'mask': 3,
            'age_6': 6,
            'mask_gender': 6,
            'binary_60': 2
        }

        num_classes = num_classes_match[classifier_target]
        for name, mod in reversed(list(model.named_modules())):
            early_break = False
            if isinstance(mod, nn.Linear):
                mod_path = name.split('.')
                classifier_parent = reduce(nn.Module.get_submodule, mod_path[:-1], model)
                setattr(classifier_parent, mod_path[-1], nn.Sequential(
                    nn.Dropout(0.7),
                    nn.Linear(mod.in_features, 1024),
                    nn.Dropout(0.3),
                    nn.Linear(1024, 512),
                    nn.Dropout(0.2),
                    nn.Linear(512, num_classes)
                ))
                break

        self.model = model

    def __call__(self):
        return self.model


class Ensemble(nn.Module):
    def __init__(self, modelA, modelB, modelC, num_classes, multiclass=True):
        super(FinalEnsembleModel, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.modelC = modelC
        self.multiclass = multiclass
        # Make sure that all models are frozen!!!!!

        self.multi_fc = nn.Linear(num_classes, num_classes)
        self.binary_fc = nn.Linear(num_classes, 1)

    def forward(self, x):
        out1 = self.modelA(x)
        out2 = self.modelB(x)
        out3 = self.modelC(x)
        if self.multiclass:
            out = out1 + out2 + out3
            out = self.multi_fc(out)
            out = torch.softmax(out, dim=1)
        else:
            out = torch.cat((out1, out2, out3), dim=1)
            out = self.binary_fc(out)

        return out


def get_pretrained_model_by_name(model_name):
    model = getattr(models, model_name)(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    model.classifier[1] = nn.Linear(1280, 18, bias=True)
    return model
