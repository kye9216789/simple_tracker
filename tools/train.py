import argparse
import os
import sys
import glob
import tqdm

import torch
import torch.nn as nn

from mmcv import Config
from mmcv.runner import obj_from_dict
sys.path.append(os.path.abspath('.'))
from model.tracker.simple_tracker import Tracker


class Runner(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = Tracker(cfg)
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(model.parameters(), lr=0.007)
        self.train_dataloader = load_dataloader(cfg, 'train')
        self.valid_dataloader = load_dataloader(cfg, 'valid')
        self.total_epoch = cfg.train.epoch
        self.interval = cfg.valid.interval
        self.train_loss = 0.0
        self.valid_loss = 0.0

    def load_dataloader(cfg, mode):
        if mode == 'train':
            pass
        elif mode == 'valid':
            pass

    def run(self):
        for i in range(self.total_epoch):
            self.run_epoch(self.model, self.train_dataloader, self.cfg)
            if (i + 1) % self.interval == 0:
                self.run_epoch(self.model, self.valid_dataloader, self.cfg, train_mode=False)

    def run_epoch(model, dataloader, cfg, train_mode=True):
        tbar = tqdm(dataloader)
        model.rnn_module.mode(train_mode)
        for i, sample in enumerate(tbar):
            target = sample["target"]
            pred = self.model(**sample)
            if pred is not None:
                output = self.loss(pred, target)
                if train_mode:
                    output.backward()
                    self.optimizer.step()
                    self.train_loss += output.item()
                    loss_value = self.train_loss
                else:
                    self.valid_loss += output.item()
                    loss_value = self.valid_loss
                tbar.set_description(printer(loss_value / (i + 1)))

    def printer(self, loss, train_mode=True):
        if train_mode:
            description = 'Train loss: '
        else:
            description = 'Test loss: '
        return description + f'{loss}'


def get_mmdet_dir(cfg):
    mmdet_dir = cfg.bbox.repository_dir
    work_dir = cfg.bbox.work_dir
    return os.path.join(mmdet_dir, work_dir)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='Config file path for tracker')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config).cfg
    cfg.mmdet_dir = get_mmdet_dir(cfg)
    cfg.mmdet_cfg = Config.fromfile(glob.glob(os.path.join(cfg.mmdet_dir, '*.py'))[0])
    train_runner = Runner(cfg)

    if train_runner.model is not None:
        print('model loaded!')
    train_runner.run()

if __name__ == '__main__':
    main()






