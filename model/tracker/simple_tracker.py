import torch
import torch.nn as nn

from model.bbox.mmdet_bbox import TrackerCnnModule
from model.lstm.rnn_lstm import TrackerRnnModule


class Tracker(nn.Module):
    def __init__(self, cfg):
        super(Tracker, self).__init__()
        self.batch_size = cfg.tracker.batch_size
        self.cnn_module = TrackerCnnModule(cfg)
        self.rnn_module = TrackerRnnModule(cfg).cuda()
        self.batch_input = self.init_batch_input()
        self.softmax = nn.Softmax()

    def forward(self, imgs, bbox_pred):
        bbox_feats = self.cnn_module(imgs, bbox_pred)
        bbox_feats = torch.unsqueeze(bbox_feats, 0)
        self.batch_input = torch.cat((self.batch_input, bbox_feats))

        if self.batch_input.size(0) == self.batch_size:
            pred = self.rnn_module(self.batch_input)
            self.batch_input = self.init_batch_input()
            return self.softmax(pred)

    def init_batch_input(self):
        return torch.Tensor().cuda()