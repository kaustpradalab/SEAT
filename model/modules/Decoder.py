from allennlp.common.from_params import FromParams
import torch
import torch.nn as nn
from typing import Dict
from allennlp.common import Params

from attention.model.modules.Attention import Attention
from attention.model.modelUtils import isTrue, BatchHolder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AttnDecoder(nn.Module, FromParams) :
    def __init__(self, hidden_size:int, 
                       attention:Dict, 
                       output_size:int = 1, 
                       use_attention:bool = True) :
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.linear_1 = nn.Linear(hidden_size, output_size)

        attention['hidden_size'] = self.hidden_size
        self.attention = Attention.from_params(Params(attention))

        self.use_attention = use_attention
               
    def decode(self, predict) :
        predict = self.linear_1(predict)
        return predict

    def get_att(self,data:BatchHolder):
        if self.use_attention:
            output = data.hidden
            mask = data.masks
            attn = self.attention(output.to(device), mask.to(device))
            return attn
        else:
            assert False

    def forward(self, data:BatchHolder, revise_att=None, revise_hidden=None) :
        if self.use_attention :

            if revise_hidden:
                hidden_embed = revise_hidden
            else:
                hidden_embed = data.hidden
            mask = data.masks
            # print('mask', mask.shape)

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            if revise_att is not None:
                attn = revise_att
            else:
                attn = self.attention(hidden_embed.to(device), mask.to(device))
            # print('hidden embedding shape',hidden_embed.shape, 'mask shape',mask.shape)

            data.attn = attn
            context = (data.attn.unsqueeze(-1).to(device) * hidden_embed.to(device)).sum(1)
        else :
            context = data.last_hidden
            
        predict = self.decode(context.to(device))
        data.predict = predict


class FrozenAttnDecoder(AttnDecoder) :

    def forward(self, data:BatchHolder) :
        if self.use_attention :
            output = data.hidden
            attn = data.generate_frozen_uniform_attn()

            context = (attn.unsqueeze(-1) * output).sum(1)
            data.attn = attn
        else :
            context = data.last_hidden
            
        predict = self.decode(context)
        data.predict = predict


class PretrainedWeightsDecoder(AttnDecoder) :

    def forward(self, data:BatchHolder) :
        if self.use_attention :
            output = data.hidden
            attn = data.target_attn

            context = (attn.unsqueeze(-1) * output).sum(1)
            data.attn = attn
        else :
            context = data.last_hidden
            
        predict = self.decode(context)
        data.predict = predict
