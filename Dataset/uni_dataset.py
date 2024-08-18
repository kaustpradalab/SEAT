import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
from datasets import Dataset as Hug_Dataset, DatasetDict
from transformers import BertTokenizer
import numpy as np
import pickle

class UnifiedDataset(Dataset):
    def __init__(self, bert=False, datadir=None, dataset_name=None, train=False,args=None, target=False):

        self.train_type = 'train' if train else 'test'
        train = 'train' if train else 'test'
        self.bert = bert
        self.target = target
        self.datadir = datadir
        self.dataset_name = dataset_name

        if bert:
            self.df = pd.read_csv(datadir, )
            self.bert = bert
            # use bert tokenizer to tokenize data, load training and testing data
            df = self.df[self.df['exp_split'] == train][['text', 'label']].reset_index(drop=True)
            print(df.head())
            # print(
            df = df.dropna(axis=0, how='any')
            # )
            newData = DatasetDict(
                {
                    train: Hug_Dataset.from_pandas(df),
                })
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

            def encode(examples):
                return tokenizer(examples['text'], truncation=True, padding='max_length', add_special_tokens=True)
            
            filename = datadir.replace('data',f'data-{train}').replace(".csv", '.ckpt')
            if os.path.exists(filename):
                with open(filename, 'rb') as f:
                    newData = pickle.load(f)    
            else:
                newData = newData.map(encode, batched=True)
                newData = newData.remove_columns(["text"])
                with open(filename,'wb') as f:
                    pickle.dump(newData, f)   
                
            self.y = newData[train]['label'].copy()
            # newData.set_format("torch")
            self.X = {}
            self.X['input_ids'] = newData[train]['input_ids']
            self.X['attention_mask'] = newData[train]['attention_mask']
            self.X['token_type_ids'] = newData[train]['token_type_ids']
            del newData
        else:
            
            self.vec = pickle.load(open(datadir, 'rb'))
            self.X, self.y = self.vec.seq_text[train], self.vec.label[train] \
                # these are lists (of lists) of num. insts-length (NOT PADDED)
            # X, y = filterbylength(X, y, min_length=min_length, max_length=max_length)
            # Xt, yt = filterbylength(Xt, yt, min_length=min_length, max_length=max_length)
            # Xt, yt = sortbylength(Xt, yt)
        # train_or_not = 'train' if train else 'test'
        num = len(self.y)
        print(f"{train} set has {num} instances")

        y = np.array(self.y)
        self.pos_weight = [len(y) / sum(y) - 1]
        import json
        self.target = target
        if target:
            self.y_attn = json.load(open(os.path.join(args.gold_label_dir, f'{train}_attentions_best_epoch.json'), 'r'))
            self.true_pred = json.load(open(os.path.join(args.gold_label_dir, f'{train}_predictions_best_epoch.json'), 'r'))
            self.true_pred = [e[0] for e in self.true_pred]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if self.bert:
            data = {
                "input_ids": self.X['input_ids'][idx],
                "attention_mask": self.X['attention_mask'][idx],
                "token_type_ids": self.X['token_type_ids'][idx],
                'label': self.y[idx],
            }
        else:
            data = {
                "input_ids": self.X[idx],
                'label': self.y[idx],
            }

        if self.target:
            data['y_attn'] = self.y_attn[idx]
            data['true_pred'] = self.true_pred[idx]

        data['bert'] = self.bert

        return data