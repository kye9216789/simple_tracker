import torch.nn as nn
from model.utils import Flatten

class TrackerRnnModule(nn.Module):
    def __init__(self, cfg):
        super(TrackerRnnModule, self).__init__()

        self.input_dim = self.get_input_dim(cfg)
        self.hidden_dim = cfg.lstm.hidden_dim
        self.batch_size = cfg.tracker.batch_size
        self.num_layers = cfg.lstm.num_layers
        self.seq_len = cfg.lstm.seq_len
        self.out_dim = cfg.lstm.out_dim

        self.flatten = Flatten(self.seq_len, self.batch_size)
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)
        self.linear = nn.Linear(self.hidden_dim, self.out_dim)

    def init_hidden(self):
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim) * num_layers)

    def get_input_dim(self, cfg):
        roi_channel = cfg.mmdet_cfg.bbox_roi_extractor.out_channels
        out_size = cfg.mmdet_cfg.bbox_roi_extractor.roi_layer.out_size ** 2
        return roi_channel * out_size

    def forward(self, input):
        input = self.flatten(input)
        lstm_out, self.hidden = self.lstm(input.view(input.size(1), input.size(0), -1))
        pred = self.linear(lstm_out[-1].view(self.batch_size, -1))
        return pred.view(self.batch_size, -1)

    def train_mode(self, train_mode):
        if train_mode:
            self.model.train()
        else:
            self.model.eval()