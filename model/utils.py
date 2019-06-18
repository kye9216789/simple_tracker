import torch.nn as nn


class Flatten(nn.Module):
    def __init__(self, seq_len, batch_size):
        super(Flatten, self).__init__()
        self.seq_len = seq_len
        self.batch_size = batch_size

    def forward(self, input):
        return input.view(self.batch_size, self.seq_len, -1)