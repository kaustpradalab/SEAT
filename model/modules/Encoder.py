import torch
import torch.nn as nn

from transformers import BertTokenizer, BertModel, BertForMaskedLM
from attention.model.modelUtils import isTrue
from allennlp.common import Registrable
from allennlp.nn.activations import Activation

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Encoder(nn.Module, Registrable) :
    def forward(self, **kwargs) :
        raise NotImplementedError("Implement forward Model")


@Encoder.register('simple-rnn')
class EncoderRNN(Encoder):
    def __init__(self, vocab_size, embed_size, hidden_size, pre_embed=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size

        if pre_embed is not None:
            # print("Setting Embedding")
            weight = torch.Tensor(pre_embed)
            weight[0, :].zero_()
            self.embedding = nn.Embedding(vocab_size, embed_size, _weight=weight, padding_idx=0)
        else:
            self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)

        self.hidden_size = hidden_size
        # self.rnn = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, batch_first=True, bidirectional=True)
        self.rnn = torch.nn.RNN(input_size=embed_size, hidden_size=hidden_size, num_layers=1, bias=True,
                           bidirectional=False)
        self.output_size = self.hidden_size

    def forward(self, data, revise_embedding=None):
        # assert data.encoder == "non-bert"
        seq = data.seq
        lengths = data.lengths
        if revise_embedding is not None:
            embedding = revise_embedding
        else:
            embedding = self.embedding(seq)  # (B, L, E)
        packseq = nn.utils.rnn.pack_padded_sequence(embedding, lengths.cpu(), batch_first=True, enforce_sorted=False)
        output, h = self.rnn(packseq)
        output, lengths = nn.utils.rnn.pad_packed_sequence(output, batch_first=True, padding_value=0)

        data.hidden = output
        data.last_hidden = h
        data.embedding = embedding


        if isTrue(data, 'keep_grads'):
            # data.embedding = embedding
            data.embedding.retain_grad()
            data.hidden.retain_grad()


@Encoder.register('lstm')
class EncoderRNN(Encoder) :
    def __init__(self, vocab_size, embed_size, hidden_size, pre_embed=None) :
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size

        if pre_embed is not None :
            print("Setting Embedding")
            weight = torch.Tensor(pre_embed)
            weight[0, :].zero_()

            self.embedding = nn.Embedding(vocab_size, embed_size, _weight=weight, padding_idx=0)
        else :
            self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)

        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, batch_first=True, bidirectional=True)

        self.output_size = self.hidden_size * 2

    def forward(self, data,revise_embedding=None) :
        # assert data.encoder == "non-bert"
        seq = data.seq
        lengths = data.lengths
        if revise_embedding is not None:
            embedding = revise_embedding
        else:
            embedding = self.embedding(seq)  # (B, L, E)
        packseq = nn.utils.rnn.pack_padded_sequence(embedding, lengths.cpu(), batch_first=True, enforce_sorted=False)
        output, (h, c) = self.rnn(packseq)
        output, lengths = nn.utils.rnn.pad_packed_sequence(output, batch_first=True, padding_value=0)

        data.hidden = output
        data.last_hidden = torch.cat([h[0], h[1]], dim=-1)
        data.embedding = embedding

        if isTrue(data, 'keep_grads') :
            # data.embedding = embedding
            data.embedding.retain_grad()
            data.hidden.retain_grad()
            

@Encoder.register("average")
class EncoderAverage(Encoder) :
    def __init__(self,  vocab_size, embed_size, projection, hidden_size=None, activation:Activation=Activation.by_name('linear'), pre_embed=None) :
        super(EncoderAverage, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size

        if pre_embed is not None :
            print("Setting Embedding")
            weight = torch.Tensor(pre_embed)
            weight[0, :].zero_()

            self.embedding = nn.Embedding(vocab_size, embed_size, _weight=weight, padding_idx=0)
        else :
            self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)

        if projection :
            self.projection = nn.Linear(embed_size, hidden_size)
            self.output_size = hidden_size
        else :
            self.projection = lambda s : s
            self.output_size = embed_size

        self.activation = activation

    def forward(self, data,revise_embedding=None) :
        # assert data.encoder == "non-bert"
        seq = data.seq
        lengths = data.lengths
        # embedding = self.embedding(seq) #(B, L, E)
        if revise_embedding is not None:
            embedding = revise_embedding
        else:
            embedding = self.embedding(seq)  # (B, L, E)

        output = self.activation(self.projection(embedding))
        h = output.mean(1)

        data.hidden = output
        data.last_hidden = h
        data.embedding = embedding

        if isTrue(data, 'keep_grads') :
            # data.embedding = embedding
            data.embedding.retain_grad()
            data.hidden.retain_grad()


@Encoder.register("bert")
class EncoderBERT(Encoder) :
    def __init__(self) :
        super(EncoderBERT, self).__init__()
        # self.pre_embed=pre_embed
        self.embed_size = 768
        self.output_size = self.embed_size
        self.model = BertModel.from_pretrained('bert-base-uncased')

    def forward(self, data, revise_embedding=None) :
        # assert data.encoder == "bert"
        if revise_embedding is not None:
            # assert revise_embedding.shape[0] == 8
            embedding = revise_embedding
            # print("Revised Embedding")
            # print(embedding.shape)
        else:
            # embedding = self.embedding(seq)  # (B, L, E)
            seq = data.seq
            token_type_ids = data.token_type_ids
            attention_mask = data.attention_mask
            inp = {
                'input_ids': seq,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids
            }
            embedding = self.model(**{k: v.to(device) for k, v in inp.items()})[0] # (B, L, E)
        output = embedding
        data.embedding = output
        data.hidden = output
        data.last_hidden = output.mean(1).cpu()