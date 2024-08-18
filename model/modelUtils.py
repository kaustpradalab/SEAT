def isTrue(obj, attr) :
    return hasattr(obj, attr) and getattr(obj, attr)

import numpy as np
import torch

def get_sorting_index_with_noise_from_lengths(lengths, noise_frac) :
    if noise_frac > 0 :
        noisy_lengths = [x + np.random.randint(np.floor(-x*noise_frac), np.ceil(x*noise_frac)) for x in lengths]
    else :
        noisy_lengths = lengths
    return np.argsort(noisy_lengths)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#


class BatchHolder() :
    def __init__(self, data,bert=False) :
        # maxlen=512
        if bert:
            maxlen = 512
        else:
            maxlen = max([len(x) for x in data['input_ids']])
        self.maxlen = maxlen
        self.B = len(data['input_ids'])
        self.encoder = 'bert' if bert else 'non-bert'
        target_attn = True if 'y_attn' in data.keys() else False
        self.bert=bert
        if bert:
            # self.length =
            self.seq = torch.LongTensor(np.array(data['input_ids'])).to(device)
            self.attention_mask = torch.tensor(np.array(data['attention_mask'])).to(device)
            # (B, L)
            # set the attention mask of font [CLS] and the tail [SEQ] to be zero
            self.masks = self.attention_mask.clone()
            self.masks[:,0] = 0
            idx = self.masks.sum(1)
            self.masks[np.arange(0,self.masks.shape[0]),idx] = 0
            # self.masks = torch.ByteTensor(np.array(self.masks)).to(device)
            self.token_type_ids = torch.LongTensor(np.array(data['token_type_ids'])).to(device)
            self.label = torch.FloatTensor(np.array(data['label'])).to(device)
            if target_attn:
                self.target_attn = torch.FloatTensor(data['y_attn']).to(device)
        else:
            lengths = []
            expanded = []
            masks = []
            expanded_attn = []

            for i in range(self.B):
                d = data['input_ids'][i]
                rem = maxlen - len(data['input_ids'][i])
                expanded.append(d + [0]*rem)
                lengths.append(len(d))
                masks.append([1] + [0]*(len(d)-2) + [1]*(rem+1))
                # if target_attn:
                #     ta = data['y_attn'][i]
                #     print(len(data['y_attn'][i]))
                #     print(len(data['input_ids'][i]))
                #     assert len(data['y_attn'][i]) == len(data['input_ids'][i])
                #     assert len([1] + [0] * (len(d) - 2) + [1] * (rem + 1)) == len(ta + [0] * rem) == len(
                #         d + [0] * rem)
                #     # also pad target attention:
                #     expanded_attn.append(ta + [0] * rem)

            self.lengths = torch.LongTensor(np.array(lengths)).to(device)
            self.seq = torch.LongTensor(np.array(expanded, dtype='int64')).to(device)
            self.masks = torch.tensor(np.array(masks)).to(device)
            self.label = torch.FloatTensor(np.array(data['label'])).to(device)
            # if target_attn:
            #     self.target_attn = torch.FloatTensor(expanded_attn).to(device)

        # if target_attn:
        #     self.target_pred = torch.FloatTensor(data['true_pred'])
        #     if len(self.target_pred.shape) == 1:  # (B, )
        #         self.target_pred = self.target_pred.unsqueeze(
        #             -1)  # (B, 1)

        self.label = self.label.unsqueeze(1)
        self.hidden = None
        self.predict = None
        self.attn = None
        self.inv_masks = ~self.masks

    def to_cpu(self):
        if self.bert:
            self.attention_mask = self.attention_mask.cpu()
            self.token_type_ids = self.token_type_ids.cpu()
        else:
            self.lengths = self.lengths.cpu()

        self.seq = self.seq.cpu()
        self.masks = self.masks.cpu()
        self.label = self.label.cpu()

        # if self.target_attn is not None:
        #     self.target_attn = self.target_attn.cpu()

    def generate_frozen_uniform_attn(self):
        attn = np.zeros((self.B, self.maxlen))
        inv_l = 1. / (self.lengths.cpu().data.numpy() - 2)
        attn += inv_l[:, None]
        attn = torch.Tensor(attn).to(device)
        attn.masked_fill_(self.masks.bool(), 0)
        return attn

# class BatchHolder() :
#     def __init__(self, data, target_attn=None) :
#         maxlen = max([len(x) for x in data])
#         self.maxlen = maxlen
#         self.B = len(data)
#
#         lengths = []
#         expanded = []
#         masks = []
#         expanded_attn = []
#
#         if target_attn :
#             for d, ta in zip(data, target_attn) :
#                 assert len(d) == len(ta)
#                 rem = maxlen - len(d)
#                 expanded.append(d + [0]*rem)
#                 lengths.append(len(d))
#                 masks.append([1] + [0]*(len(d)-2) + [1]*(rem+1))
#                 assert len([1] + [0]*(len(d)-2) + [1]*(rem+1)) == len(ta + [0]*rem) == len(d + [0]*rem)
#                 #also pad target attention:
#                 expanded_attn.append(ta + [0]*rem)
#         else :
#             for _, d in enumerate(data) :
#                 rem = maxlen - len(d)
#                 expanded.append(d + [0]*rem)
#                 lengths.append(len(d))
#                 masks.append([1] + [0]*(len(d)-2) + [1]*(rem+1))
#
#         self.lengths = torch.LongTensor(np.array(lengths)).to(device)
#         self.seq = torch.LongTensor(np.array(expanded, dtype='int64')).to(device)
#         self.masks = torch.ByteTensor(np.array(masks)).to(device)
#
#         self.hidden = None
#         self.predict = None
#         self.attn = None
#
#         if target_attn :
#                 self.target_attn = torch.FloatTensor(expanded_attn).to(device)
#         self.inv_masks = ~self.masks
#
#     def generate_frozen_uniform_attn(self):
#         attn = np.zeros((self.B, self.maxlen))
#         inv_l = 1. / (self.lengths.cpu().data.numpy() - 2)
#         attn += inv_l[:, None]
#         attn = torch.Tensor(attn).to(device)
#         attn.masked_fill_(self.masks.bool(), 0)
#         return attn

def kld(a1, a2) :
    #(B, *, A), #(B, *, A)
    a1 = torch.clamp(a1, 0, 1)
    a2 = torch.clamp(a2, 0, 1)
    log_a1 = torch.log(a1 + 1e-10)
    log_a2 = torch.log(a2 + 1e-10)

    kld = a1 * (log_a1 - log_a2)
    kld = kld.sum(-1)

    return kld

def jsd(p, q) :
    m = 0.5 * (p + q)
    jsd = 0.5 * (kld(p, m) + kld(q, m)) #for each instance in the batch
    
    return jsd.unsqueeze(-1) #jsd.squeeze(1).sum()

if __name__ == '__main__':
    class x:
        def __init__(self):
            pass
    x_ = x()
    x_.keep = True
    print(isTrue(x_,'keep'))
