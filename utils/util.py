import torch
import json
from pathlib import Path
from collections import OrderedDict
import pandas as pd
import pickle

import requests
# 카카오톡 메시지 API
def kakao_init():
    url = "https://kauth.kakao.com/oauth/token"

    data = {
        "grant_type" : "authorization_code",
        "client_id" : "f87c9fbc992598da06ddbad25ac8a1ce",
        "redirect_url" : "https://localhost:3000",
        "code" : "Mii16IW-1WY35CDa3K8oOlEDpOq02t8JC9DthjjCkedYwwwOv3wYDT44gy-SWoD2K898wAorDKgAAAF_O_hiLw"
    }
    response = requests.post(url, data=data)
    tokens = response.json()
    print(tokens)

# 카카오톡 메시지 API
    url = "https://kauth.kakao.com/oauth/token"

    data = {
        "grant_type": "refresh_token",
        "client_id": "f87c9fbc992598da06ddbad25ac8a1ce",
        "refresh_token": tokens['refresh_token']
    }
    response = requests.post(url, data=data)
    tokens = response.json()

# kakao_code.json 파일 저장
    with open("kakao_code.json", "w") as fp:
        json.dump(tokens, fp)

def kakao_send(msg):
    with open("kakao_code.json", "r") as fp:
        tokens = json.load(fp)

    url = "https://kapi.kakao.com/v2/api/talk/memo/default/send"

    headers = {
        "Authorization": "Bearer " + tokens['access_token']
    }


    data = data = {
    "template_object" : json.dumps({ "object_type" : "text",
                                     "text" : str(msg),
                                     "link" : {
                                              }
        })
    }

    response = requests.post(url, headers=headers, data=data)
    if response.json().get('result_code') == 0:
        print('메시지를 성공적으로 보냈습니다.')
    else:
        print('메시지를 성공적으로 보내지 못했습니다. 오류메시지 : ' + str(response.json()))



def read_json(fname):
    fname = Path(fname)
    with fname.open('r') as f:
        return json.load(f, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    # Notice: 'dump' saves to file, 'dumps' creates json data
    with fname.open('w') as f:
        json.dump(content, f, indent=4, sort_keys=False)

def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids

class MetricTracker:
    def __init__(self, *keys):
        self._data = pd.DataFrame(index=keys, columns=['total','counts','average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)


def load_checkpoint(checkpoint, model, name):
    print("Loading Checkpoint....")
    #  model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.load_state_dict(checkpoint[name])
    print("-"*50)

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False

def load_pickle(fpath):
    with open(fpath, 'rb') as f:
        d = pickle.load(f)

    return d
