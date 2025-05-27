#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Label smoothing module."""

import torch
from torch import nn


class LabelSmoothingKLDLoss(nn.Module):
    """Label-smoothing loss.

    :param int size: the number of class
    :param int padding_idx: ignored class id
    :param float smoothing: smoothing rate (0.0 means the conventional CE)
    :param bool normalize_length: normalize loss by sequence length if True
    :param torch.nn.Module criterion: loss function to be smoothed
    """

    def __init__(
        self,
        size,
        padding_idx,
        smoothing,
        normalize_length=False,
        criterion=nn.KLDivLoss(reduction="none"),
    ):
        """Construct an LabelSmoothingLoss object."""
        super(LabelSmoothingLoss, self).__init__()
        self.criterion = criterion
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        self.normalize_length = normalize_length

    def forward(self, x, target):
        """Compute loss between x and target.

        :param torch.Tensor x: prediction (batch, seqlen, class)
        :param torch.Tensor target:
            target signal masked with self.padding_id (batch, seqlen)
        :return: scalar float value
        :rtype torch.Tensor
        """
        assert x.size(2) == self.size
        batch_size = x.size(0)
        x = x.view(-1, self.size)
        target = target.view(-1)
        with torch.no_grad():
            true_dist = x.clone()
            true_dist.fill_(self.smoothing / (self.size - 1))
            ignore = target == self.padding_idx  # (B,)
            total = len(target) - ignore.sum().item()
            target = target.masked_fill(ignore, 0)  # avoid -1 index
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        kl = self.criterion(torch.log_softmax(x, dim=1), true_dist)
        denom = total if self.normalize_length else batch_size
        return kl.masked_fill(ignore.unsqueeze(1), 0).sum() / denom



class LabelSmoothingCELoss(nn.Module):
    """Label smoothing loss in CrossEntropyLoss style."""

    def __init__(self, size, padding_idx, smoothing=0.1, normalize_length=False):
        super(LabelSmoothingCrossEntropyLoss, self).__init__()
        self.padding_idx = padding_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        self.size = size
        self.normalize_length = normalize_length

    def forward(self, x, target):
        """
        x: [batch, seq_len, num_classes]
        target: [batch, seq_len]
        """
        assert x.size(-1) == self.size
        batch_size = x.size(0)
        x = x.view(-1, self.size)
        target = target.view(-1)

        ignore = target == self.padding_idx
        total = len(target) - ignore.sum().item()

        # log_probs: [B*T, C]
        log_probs = F.log_softmax(x, dim=-1)

        # Create smooth targets
        with torch.no_grad():
            true_dist = torch.full_like(log_probs, self.smoothing / (self.size - 1))
            target_temp = target.clone()
            target_temp[ignore] = 0  # prevent out-of-bound index
            true_dist.scatter_(1, target_temp.unsqueeze(1), self.confidence)

        # loss: [B*T]
        loss = -(true_dist * log_probs).sum(dim=1)

        # mask out padding
        loss = loss.masked_fill(ignore, 0)

        denom = total if self.normalize_length else batch_size
        return loss.sum() / denom
