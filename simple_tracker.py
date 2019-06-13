from mmcv import Config

from mmdet import __version__
from mmdet.datasets import get_dataset
from mmdet.apis import (train_detector, init_dist, get_root_logger,
                        set_random_seed)
from mmcv.runner import load_checkpoint, parallel_test, obj_from_dict
from mmdet.models import build_detector
from mmdet.datasets import build_dataloader
from mmcv.parallel import scatter, collate, MMDataParallel
from mmdet.core import bbox2roi, bbox2result, build_assigner, build_sampler
from mmdet import datasets

import torch
import torch.nn as nn
import sys
import os
import glob

mmdet_dir = '/home/ye/github/mp_mmdetection/'
work_dir = 'work_dirs/190604_retinanet_x101_32x4d_fpn_1x'
sys.path.append(os.path.abspath(mmdet_dir))
from mmdet.models import builder


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(1, -1)
    

class Tracker(nn.Module):
    def __init__(self, cfg):
        super(Tracker, self).__init__()
        self.cnn_module = TrackerCnnModule(cfg)
        self.rnn_module = TrackerRnnModule(256*49, 500, 1).cuda() # TODO : save variable in config
        
    def forward(self, img, img_meta):
        bbox_feats = self.cnn_module(img, img_meta)
        pred = self.rnn_module(bbox_feats)
        return pred


class TrackerRnnModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, batch_size, output_dim=1,
                    num_layers=2):
        super(TrackerRnnModule, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)

        self.linear = nn.Linear(self.hidden_dim, output_dim)
        
    def init_hidden(self):
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))
        
    def forward(self, input):
        lstm_out, self.hidden = self.lstm(input.view(len(input), self.batch_size, -1))
        pred = self.linear(lstm_out[-1].view(self.batch_size, -1))
        return pred.view(-1)
    

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
        self.flatten = Flatten()
        self.cfg = cfg
        
    def forward(self, img, img_meta, rescale=False):
        x = self.detector.extract_feat(img)
        outs = self.detector.bbox_head(x)
        bbox_inputs = outs + (img_meta, cfg.test_cfg, rescale)
        bboxes, cls = self.detector.bbox_head.get_bboxes(*bbox_inputs)[0]
        rois = bbox2roi([bboxes])
        bbox_feats = self.bbox_roi_extractor(x[:self.bbox_roi_extractor.num_inputs], rois)[0]
        bbox_feats = self.flatten(bbox_feats)
        return bbox_feats

    
cfg = Config.fromfile(glob.glob(os.path.join(mmdet_dir, work_dir, '*.py'))[0])
cfg.work_dir = work_dir
cfg.mmdet_dir = mmdet_dir

dataset = obj_from_dict(cfg.data.train, datasets, dict(test_mode=False))    
data_loader = build_dataloader(
    dataset,
    imgs_per_gpu=1,
    workers_per_gpu=cfg.data.workers_per_gpu,
    num_gpus=1,
    dist=False,
    shuffle=False)

tracker = Tracker(cfg)
dataset = data_loader.dataset
for i, data in enumerate(data_loader):
    img_meta = data['img_meta'].data[0]
    img = data['img'].data[0].cuda()
    pred = tracker(img, img_meta)
    break
    