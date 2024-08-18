import os
import pickle
import numpy as np
import json
from attention.preprocess import vectorizer
import pandas as pd
from attention.Dataset.uni_dataset import UnifiedDataset

class DatasetWrapper:
    def __init__(self, data, bert=False,name=None):
        dataset_train, dataset_test = data['train'], data['test']
        self.train = dataset_train
        self.test = dataset_test
        self.output_size = 1
        self.name = name
        self.pos_weight = dataset_train.pos_weight
        self.bert=bert
        if bert is False:
            # self.hidden_size = dataset_train.vec.hidden_size
            self.vocab_size = dataset_train.vec.vocab_size
            self.word_dim = dataset_train.vec.word_dim
            self.vec = dataset_train.vec


    def __getitem__(self, item):
        if item == 'train':
            return self.train
        elif item == 'test':
            return self.test


def load_dataset_custom(dataset_name,args=None):
    if args.encoder == "bert":
        dataset_train = UnifiedDataset(bert=True, datadir='attention/preprocess/' + dataset_name + "/data.csv"\
                       , dataset_name=dataset_name, train=True, args=args, target=args.train_mode=="adv_train")
        dataset_test = UnifiedDataset(bert=True, datadir='attention/preprocess/' + dataset_name + "/data.csv"\
                       , dataset_name=dataset_name, train=False, args=args, target=args.train_mode=="adv_train")
    else:
        dataset_train = UnifiedDataset(bert=False, datadir='attention/preprocess/' + dataset_name + "/vec.p"\
                       , dataset_name=dataset_name, train=True, args=args, target=args.train_mode=="adv_train",)
        dataset_test = UnifiedDataset(bert=False, datadir='attention/preprocess/' + dataset_name + "/vec.p"\
                       , dataset_name=dataset_name, train=False, args=args, target=args.train_mode=="adv_train",)


    return DatasetWrapper({
        "train": dataset_train,
        "test": dataset_test
    },bert=args.encoder == "bert",name=dataset_name)