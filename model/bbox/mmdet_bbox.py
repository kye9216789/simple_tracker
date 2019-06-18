import glob
import os

import torch.nn as nn
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector, builder


class TrackerCnnModule(nn.Module):
    def __init__(self, cfg):
        super(TrackerCnnModule, self).__init__()
        self.cfg = cfg.mmdet_cfg
        model = build_detector(self.cfg.model, train_cfg=self.cfg.train_cfg, test_cfg=self.cfg.test_cfg)
        checkpoint = glob.glob(os.path.join(cfg.mmdet_dir, '*.pth'))[0]
        model = MMDataParallel(model, device_ids=[0])

        self.bbox_roi_extractor = builder.build_roi_extractor(self.cfg.bbox_roi_extractor)

        for param in model.parameters():
            param.requires_grad = False
        self.detector = model.module

    def forward(self, img, bbox_pred, rescale=False):
        x = self.detector.extract_feat(img)

        rois = bbox2roi([bbox for bbox in bbox_pred])
        bbox_feats = self.bbox_roi_extractor(x[:self.bbox_roi_extractor.num_inputs], rois)
        return bbox_feats