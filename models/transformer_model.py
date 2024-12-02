import torch
import torch.nn as nn
import numpy as np

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6, dim_feedforward=2048, max_seq_length=512):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = nn.Embedding(max_seq_length, d_model)
        self.encoder = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.Transformer(d_model, nhead, num_layers, num_layers, dim_feedforward)
        self.decoder = nn.Linear(d_model, vocab_size)
        self.d_model = d_model
        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask):
        seq_length = src.size(0)
        positions = torch.arange(0, seq_length, dtype=torch.long, device=src.device)
        positions = positions.unsqueeze(1).expand(seq_length, src.size(1))
        src = self.encoder(src) * np.sqrt(self.d_model) + self.pos_encoder(positions)
        output = self.transformer(src, src, src_mask)
        output = self.decoder(output)
        return output