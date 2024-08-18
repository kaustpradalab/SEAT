import torch
import numpy as np

import json
import os
import shutil
from copy import deepcopy

import torch
import torch.nn as nn
from allennlp.common import Params
from sklearn.utils import shuffle
# from tqdm import tqdm
import time

def get_sorting_index_with_noise_from_lengths(lengths, noise_frac):
    if noise_frac > 0:
        noisy_lengths = [x + np.random.randint(np.floor(-x * noise_frac), np.ceil(x * noise_frac)) for x in lengths]
    else:
        noisy_lengths = lengths
    return np.argsort(noisy_lengths)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def kld(a1, a2):
    # (B, *, A), #(B, *, A)
    a1 = torch.clamp(a1, 0, 1)
    a2 = torch.clamp(a2, 0, 1)
    log_a1 = torch.log(a1 + 1e-10)
    log_a2 = torch.log(a2 + 1e-10)

    kld = a1 * (log_a1 - log_a2)
    kld = kld.sum(-1)

    return kld


def jsd(p, q):
    m = 0.5 * (p + q)
    jsd = 0.5 * (kld(p, m) + kld(q, m))  # for each instance in the batch

    return jsd.unsqueeze(-1)  # jsd.squeeze(1).sum()


from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, average_precision_score
import numpy as np
from pandas.io.json._normalize import nested_to_record
from collections import defaultdict
import pandas as pd
import torch
from IPython.display import display

def tvd(predictions, targets): #accepts two numpy arrays of dimension: (num. instances, )
    return (0.5 * np.abs(predictions - targets)).sum()

def batch_tvd(predictions, targets): #accepts two Torch tensors... " "
    return (0.5 * torch.abs(predictions - targets)).sum()

def calc_metrics_classification(target, predictions, target_scores=None, jsd_score=None) :

    if target_scores is not None :
        if predictions.shape[0] != target_scores.shape[0] :
            print("prediction shape: ", predictions.shape)
            print("target_scores shape: ", target_scores.shape)
            raise ValueError("predictions and target_scores must have the same number of instances")
        # assert predictions.squeeze(1).shape == target_scores.shape
        tvdist = tvd(predictions.squeeze(1), target_scores)

    if predictions.shape[-1] == 1 :
        predictions = predictions[:, 0]
        predictions = np.array([1 - predictions, predictions]).T

    predict_classes = np.argmax(predictions, axis=-1)

    if len(np.unique(target)) < 4 :
        rep = nested_to_record(classification_report(target, predict_classes, output_dict=True), sep='/')
    else :
        rep = {}
    rep.update({'accuracy' : accuracy_score(target, predict_classes)})

    if jsd_score :
        rep.update({'js_divergence' : jsd_score})
    if target_scores is not None :
        rep.update({'TVD' : tvdist})

    if predictions.shape[-1] == 2 :
        rep.update({'roc_auc' : roc_auc_score(target, predictions[:, 1])})
        rep.update({"pr_auc" : average_precision_score(target, predictions[:, 1])})
    return rep

def print_metrics(metrics, adv=False) :
    tabular = {k:v for k, v in metrics.items() if '/' in k}
    non_tabular = {k:v for k, v in metrics.items() if '/' not in k}
    print(non_tabular)

    d = defaultdict(dict)
    for k, v in tabular.items() :
        if not k.startswith('label_') :
            d[k.split('/', 1)[0]][k.split('/', 1)[1]] = v
        if '/1/' in k or 'auc' in k:
            d[k.split('/', 1)[0]][k.split('/', 1)[1]] = v

    df = pd.DataFrame(d)
    with pd.option_context('display.max_columns', 30):
        display(df.round(3))

    # if adv :
    #     print("TVD:", metrics['TVD'])
        # print("JS:", metrics['js_divergence'])



def intersection_of_two_tensor(t1, t2):
    combined = torch.cat((t1, t2))
    uniques, counts = combined.unique(return_counts=True)
    intersection = uniques[counts > 1]
    return intersection

def topK_overlap_true_loss(a,b,K=2):
    t1 = torch.argsort(a, descending=True)
    t2 = torch.argsort(b, descending=True)
    t1 = t1.detach().cpu().numpy()
    t2 = t2.detach().cpu().numpy()
    N = t1.shape[0]
    loss = []
    for i in range(N):
        inset = np.intersect1d(t1[i,:K],t2[i,:K])
        overlap = len(inset)/K
        # print(overlap)
        loss.append(overlap)
    return np.mean(loss)


class AverageMeter():
    def __init__(self):
        self.cnt = 0
        self.sum = 0
        self.mean = 0

    def update(self, val, cnt):
        self.cnt += cnt
        self.sum += val * cnt
        self.mean = self.sum / self.cnt

    def average(self):
        return self.mean

    def total(self):
        return self.sum



def topk_overlap_loss(gt,pred,K=2,metric='l1'):
    idx = torch.argsort(gt,dim=1,descending=True)
    # print(idx)
    idx = idx[:,:K]
    pred_TopK_1 = pred.gather(1, idx)
    gt_Topk_1 = gt.gather(1,idx)

    idx_pred = torch.argsort(pred,dim=1,descending=True)
    idx_pred = idx_pred[:,:K]
    try:
        gt_TopK_2 = gt.gather(1, idx_pred)
    except Exception as e:
        print(e)
        print(gt.shape)
        print(idx_pred.shape)
    pred_TopK_2 = pred.gather(1, idx_pred)

    gt_Topk_1_normed = torch.nn.functional.softmax(gt_Topk_1,dim=-1)
    pred_TopK_1_normed = torch.nn.functional.softmax(pred_TopK_1,dim=-1)
    gt_TopK_2_normed = torch.nn.functional.softmax(gt_TopK_2,dim=-1)
    pred_TopK_2_normed = torch.nn.functional.softmax(pred_TopK_2,dim=-1)

    def kl(a,b):
        return torch.nn.functional.kl_div(a.log(), b, reduction="batchmean")

    def jsd(a,b):
        loss = kl(a,b) + kl(b,a)
        loss /= 2
        return loss


    if metric == 'l1':
        loss = torch.abs((pred_TopK_1 - gt_Topk_1)) + torch.abs(gt_TopK_2 - pred_TopK_2)
        loss = loss/(2*K)
    elif metric == "l2":
        loss = torch.norm(pred_TopK_1 - gt_Topk_1, p=2) + torch.norm(gt_TopK_2 - pred_TopK_2, p=2)
        loss = loss/(2*K)
    elif metric == "kl-full":
        loss = kl(gt,pred)
    elif metric == "jsd-full":
        loss = jsd(gt,pred)
    elif metric == "kl-topk":
        loss = kl(gt_Topk_1_normed,pred_TopK_1_normed) + kl(gt_TopK_2_normed,pred_TopK_2_normed)
        loss /=2
    elif metric == "jsd-topk":
        loss = jsd(gt_Topk_1_normed, pred_TopK_1_normed) + jsd(gt_TopK_2_normed, pred_TopK_2_normed)
        loss /= 2
    return loss

if __name__ == '__main__':

    from torch.autograd import gradcheck
    import torch
    import torch.nn as nn

    # intersection_of_two_tensor(t1[i], t2[i])

    t1 = torch.tensor(
        np.array([[100, 2, 3, 4],
                  [2, 1, 3, 7]],),requires_grad=True, dtype=torch.double
    )
    print(t1.shape)
    t2 = torch.tensor(
        np.array([[1, 2, 3, 4],
                  [2, 4, 6, 7]]),requires_grad=True, dtype=torch.double
    )
    print(t2.shape)



    print(topK_overlap_true_loss(torch.argsort(t1,descending=True),torch.argsort(t2,descending=True),K=2))
