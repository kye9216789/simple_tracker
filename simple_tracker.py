from mmcv import Config

from mmdet import __version__
from mmdet.datasets import get_dataset
from mmdet.apis import (train_detector, init_dist, get_root_logger,
                        set_random_seed)
from mmcv.runner import load_checkpoint, parallel_test, obj_from_dict
from mmdet.models import build_detector, builder
from mmdet.datasets import build_dataloader
from mmcv.parallel import scatter, collate, MMDataParallel
from mmdet.core import bbox2roi, bbox2result, build_assigner, build_sampler
from mmdet import datasets

import torch
import torch.nn as nn
import sys
import os
import glob
import numpy as np

mmdet_dir = '/home/ye/github/mp_mmdetection/'
work_dir = 'work_dirs/190604_retinanet_x101_32x4d_fpn_1x'


class Flatten(nn.Module):
    def __init__(self, seq_len, batch_size):
        super(Flatten, self).__init__()
        self.seq_len = seq_len
        self.batch_size = batch_size

    def forward(self, input):
        return input.view(self.batch_size, self.seq_len, -1)


class Tracker(nn.Module):
    def __init__(self, cfg):
        super(Tracker, self).__init__()
        self.batch_size = 5
        self.cnn_module = TrackerCnnModule(cfg)
        self.rnn_module = TrackerRnnModule(256*49, 500, self.batch_size).cuda() # TODO : save variable in config
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

class TrackerRnnModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, batch_size, seq_len=3, output_dim=2,
                    num_layers=2):
        super(TrackerRnnModule, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.seq_len = seq_len

        self.flatten = Flatten(self.seq_len, self.batch_size) # [batch, 3, 256, 7, 7] => [batch, 3, 256 * 7 * 7]
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)
        self.linear = nn.Linear(self.hidden_dim, output_dim)

    def init_hidden(self):
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim) * num_layers)

    def forward(self, input):
        input = self.flatten(input)
        lstm_out, self.hidden = self.lstm(input.view(input.size(1), input.size(0), -1))
        pred = self.linear(lstm_out[-1].view(self.batch_size, -1))
        return pred.view(self.batch_size, -1)


class TrackerCnnModule(nn.Module):
    def __init__(self, cfg):
        super(TrackerCnnModule, self).__init__()
        model = build_detector(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
        checkpoint = glob.glob(os.path.join(mmdet_dir, work_dir, '*.pth'))[0]
        load_checkpoint(model, checkpoint)
        model = MMDataParallel(model, device_ids=[0])

        self.bbox_roi_extractor = builder.build_roi_extractor(cfg.bbox_roi_extractor)

        for param in model.parameters():
            param.requires_grad = False

        self.detector = model.module
        self.cfg = cfg

    def forward(self, img, bbox_pred, rescale=False):
        x = self.detector.extract_feat(img)

        rois = bbox2roi([bbox for bbox in bbox_pred])
        bbox_feats = self.bbox_roi_extractor(x[:self.bbox_roi_extractor.num_inputs], rois)
        return bbox_feats


cfg = Config.fromfile(glob.glob(os.path.join(mmdet_dir, work_dir, '*.py'))[0])
cfg.work_dir = work_dir
cfg.mmdet_dir = mmdet_dir

batch_size = 3

dataset = obj_from_dict(cfg.data.train, datasets, dict(test_mode=False))
data_loader = build_dataloader(
    dataset,
    imgs_per_gpu=batch_size,
    workers_per_gpu=cfg.data.workers_per_gpu,
    num_gpus=1,
    dist=False,
    shuffle=False)

tracker = Tracker(cfg)
dataset = data_loader.dataset
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(tracker.parameters(), lr=0.007)
for i, data in enumerate(data_loader):
    img = data['img'].data[0].cuda()
    bbox_pred = np.array([[[0, 100, 100, 50, 50]],[[1, 200, 200, 40, 40]], [[2, 100, 100, 30, 30]]])
    bbox_pred = torch.Tensor(bbox_pred).cuda()

    target = torch.empty(5, dtype=torch.long).random_(2).cuda()

    pred = tracker(img, bbox_pred)
    if pred is not None:
        output = loss(pred, target)
        output.backward()
        optimizer.step()