from torch import nn
from allennlp.common import Registrable

def masked_softmax(attn_odds, masks) :
    # print("attn_odds:", attn_odds.shape)
    attn_odds.masked_fill_(masks.bool(), -float('inf'))
    attn = nn.Softmax(dim=-1)(attn_odds)
    return attn

class Attention(nn.Module, Registrable) :
    def forward(self, **kwargs) :
        raise NotImplementedError("Implement forward Model")

@Attention.register('tanh')
class TanhAttention(Attention) :
    def __init__(self, hidden_size) :
        super().__init__()
        self.attn1 = nn.Linear(hidden_size, hidden_size // 2)
        self.attn2 = nn.Linear(hidden_size // 2, 1, bias=False)
        
    def forward(self, hidden, masks) :
        assert hidden.shape[0] == masks.shape[0]
        attn1 = nn.Tanh()(self.attn1(hidden))
        attn2 = self.attn2(attn1).squeeze(-1)
        attn = masked_softmax(attn2, masks)
        
        return attn