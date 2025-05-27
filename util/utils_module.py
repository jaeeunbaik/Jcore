
import torch
import numpy as np

from distutils.version import LooseVersion
from itertools import groupby

is_torch_1_2_plus = LooseVersion(torch.__version__) >= LooseVersion("1.2.0")
# LooseVersion('1.2.0') == LooseVersion(torch.__version__) can't include e.g. 1.2.0+aaa
is_torch_1_2 = (
    LooseVersion("1.3") > LooseVersion(torch.__version__) >= LooseVersion("1.2")
)
datatype = torch.bool if is_torch_1_2_plus else torch.uint8


def subsequent_mask(size, device="cpu", dtype=datatype):
    """Create mask for subsequent steps (1, size, size).

    :param int size: size of mask
    :param str device: "cpu" or "cuda" or torch.Tensor.device
    :param torch.dtype dtype: result dtype
    :rtype: torch.Tensor
    >>> subsequent_mask(3)
    [[1, 0, 0],
     [1, 1, 0],
     [1, 1, 1]]
    """
    if is_torch_1_2 and dtype == torch.bool:
        # torch=1.2 doesn't support tril for bool tensor
        ret = torch.ones(size, size, device=device, dtype=torch.uint8)
        return torch.tril(ret, out=ret).type(dtype)
    else:
        ret = torch.ones(size, size, device=device, dtype=dtype)
        return torch.tril(ret, out=ret)


def target_mask(ys_in_pad, ignore_id):
    """Create mask for decoder self-attention.

    :param torch.Tensor ys_pad: batch of padded target sequences (B, Lmax)
    :param int ignore_id: index of padding
    :param torch.dtype dtype: result dtype
    :rtype: torch.Tensor
    """
    ys_mask = ys_in_pad != ignore_id
    m = subsequent_mask(ys_mask.size(-1), device=ys_mask.device).unsqueeze(0)
    return ys_mask.unsqueeze(-2) & m


def pad_list(xs, pad_value):
    """Perform padding for the list of tensors.

    Args:
        xs (List): List of Tensors [(T_1, `*`), (T_2, `*`), ..., (T_B, `*`)].
        pad_value (float): Value for padding.

    Returns:
        Tensor: Padded tensor (B, Tmax, `*`).

    Examples:
        >>> x = [torch.ones(4), torch.ones(2), torch.ones(1)]
        >>> x
        [tensor([1., 1., 1., 1.]), tensor([1., 1.]), tensor([1.])]
        >>> pad_list(x, 0)
        tensor([[1., 1., 1., 1.],
                [1., 1., 0., 0.],
                [1., 0., 0., 0.]])

    """
    n_batch = len(xs)
    max_len = max(x.size(0) for x in xs)
    pad = xs[0].new(n_batch, max_len, *xs[0].size()[1:]).fill_(pad_value)

    for i in range(n_batch):
        pad[i, : xs[i].size(0)] = xs[i]

    return pad

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2020 Johns Hopkins University (Shinji Watanabe)
#                Northwestern Polytechnical University (Pengcheng Guo)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Swish() activation function for Conformer."""

import torch


class Swish(torch.nn.Module):
    """Construct an Swish object."""

    def forward(self, x):
        """Return Swich activation function."""
        return x * torch.sigmoid(x)

def get_activation(act):
    """Return activation function."""
    # Lazy load to avoid unused import
    # from swish import Swish

    activation_funcs = {
        "hardtanh": torch.nn.Hardtanh,
        "tanh": torch.nn.Tanh,
        "relu": torch.nn.ReLU,
        "selu": torch.nn.SELU,
        "swish": Swish,
        "glu": torch.nn.GLU,
    }

    return activation_funcs[act]()



def make_pad_mask(lengths, xs=None, length_dim=-1, maxlen=None):
    """Make mask tensor containing indices of padded part.

    Args:
        lengths (LongTensor or List): Batch of lengths (B,).
        xs (Tensor, optional): The reference tensor.
            If set, masks will be the same shape as this tensor.
        length_dim (int, optional): Dimension indicator of the above tensor.
            See the example.

    Returns:
        Tensor: Mask tensor containing indices of padded part.
                dtype=torch.uint8 in PyTorch 1.2-
                dtype=torch.bool in PyTorch 1.2+ (including 1.2)

    Examples:
        With only lengths.

        >>> lengths = [5, 3, 2]
        >>> make_pad_mask(lengths)
        masks = [[0, 0, 0, 0 ,0],
                 [0, 0, 0, 1, 1],
                 [0, 0, 1, 1, 1]]

        With the reference tensor.

        >>> xs = torch.zeros((3, 2, 4))
        >>> make_pad_mask(lengths, xs)
        tensor([[[0, 0, 0, 0],
                 [0, 0, 0, 0]],
                [[0, 0, 0, 1],
                 [0, 0, 0, 1]],
                [[0, 0, 1, 1],
                 [0, 0, 1, 1]]], dtype=torch.uint8)
        >>> xs = torch.zeros((3, 2, 6))
        >>> make_pad_mask(lengths, xs)
        tensor([[[0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1]],
                [[0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1]],
                [[0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1]]], dtype=torch.uint8)

        With the reference tensor and dimension indicator.

        >>> xs = torch.zeros((3, 6, 6))
        >>> make_pad_mask(lengths, xs, 1)
        tensor([[[0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [1, 1, 1, 1, 1, 1]],
                [[0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1]],
                [[0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1]]], dtype=torch.uint8)
        >>> make_pad_mask(lengths, xs, 2)
        tensor([[[0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1]],
                [[0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1]],
                [[0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1]]], dtype=torch.uint8)

    """
    if length_dim == 0:
        raise ValueError("length_dim cannot be 0: {}".format(length_dim))

    if not isinstance(lengths, list):
        lengths = lengths.long().tolist()
    bs = int(len(lengths))
    if maxlen is None:
        if xs is None:
            # maxlen = int(max(lengths))
            maxlen = max(lengths)
        else:
            maxlen = xs.size(length_dim)
    else:
        assert xs is None
        assert maxlen >= int(max(lengths))

    if isinstance(maxlen, (list, tuple)):
        maxlen = max(maxlen)
    elif isinstance(maxlen, torch.Tensor):
        maxlen = maxlen.item()
    seq_range = torch.arange(0, maxlen, dtype=torch.int64)
    seq_range_expand = seq_range.unsqueeze(0).expand(bs, maxlen)
    seq_length_expand = seq_range_expand.new(lengths).unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand

    if xs is not None:
        assert xs.size(0) == bs, (xs.size(0), bs)

        if length_dim < 0:
            length_dim = xs.dim() + length_dim
        # ind = (:, None, ..., None, :, , None, ..., None)
        ind = tuple(
            slice(None) if i in (0, length_dim) else None for i in range(xs.dim())
        )
        mask = mask[ind].expand_as(xs).to(xs.device)
    return mask


def make_non_pad_mask(lengths, xs=None, length_dim=-1):
    """Make mask tensor containing indices of non-padded part.

    Args:
        lengths (LongTensor or List): Batch of lengths (B,).
        xs (Tensor, optional): The reference tensor.
            If set, masks will be the same shape as this tensor.
        length_dim (int, optional): Dimension indicator of the above tensor.
            See the example.

    Returns:
        ByteTensor: mask tensor containing indices of padded part.
                    dtype=torch.uint8 in PyTorch 1.2-
                    dtype=torch.bool in PyTorch 1.2+ (including 1.2)

    Examples:
        With only lengths.

        >>> lengths = [5, 3, 2]
        >>> make_non_pad_mask(lengths)
        masks = [[1, 1, 1, 1 ,1],
                 [1, 1, 1, 0, 0],
                 [1, 1, 0, 0, 0]]

        With the reference tensor.

        >>> xs = torch.zeros((3, 2, 4))
        >>> make_non_pad_mask(lengths, xs)
        tensor([[[1, 1, 1, 1],
                 [1, 1, 1, 1]],
                [[1, 1, 1, 0],
                 [1, 1, 1, 0]],
                [[1, 1, 0, 0],
                 [1, 1, 0, 0]]], dtype=torch.uint8)
        >>> xs = torch.zeros((3, 2, 6))
        >>> make_non_pad_mask(lengths, xs)
        tensor([[[1, 1, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1, 0]],
                [[1, 1, 1, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0]],
                [[1, 1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 0, 0]]], dtype=torch.uint8)

        With the reference tensor and dimension indicator.

        >>> xs = torch.zeros((3, 6, 6))
        >>> make_non_pad_mask(lengths, xs, 1)
        tensor([[[1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [0, 0, 0, 0, 0, 0]],
                [[1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0]],
                [[1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0]]], dtype=torch.uint8)
        >>> make_non_pad_mask(lengths, xs, 2)
        tensor([[[1, 1, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1, 0]],
                [[1, 1, 1, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0]],
                [[1, 1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 0, 0]]], dtype=torch.uint8)

    """
    return ~make_pad_mask(lengths, xs, length_dim)

def target_mask(ys_in_pad, ignore_id):
    """Create mask for decoder self-attention.

    :param torch.Tensor ys_pad: batch of padded target sequences (B, Lmax)
    :param int ignore_id: index of padding
    :param torch.dtype dtype: result dtype
    :rtype: torch.Tensor (B, Lmax, Lmax)
    """
    ys_mask = ys_in_pad != ignore_id
    m = subsequent_mask(ys_mask.size(-1), device=ys_mask.device).unsqueeze(0)
    return ys_mask.unsqueeze(-2) & m


def to_device(m, x):
    """Send tensor into the device of the module.

    Args:
        m (torch.nn.Module): Torch module.
        x (Tensor): Torch tensor.

    Returns:
        Tensor: Torch tensor located in the same place as torch module.

    """
    if isinstance(m, torch.nn.Module):
        device = next(m.parameters()).device
    elif isinstance(m, torch.Tensor):
        device = m.device
    else:
        raise TypeError(
            "Expected torch.nn.Module or torch.tensor, " f"bot got: {type(m)}"
        )
    return x.to(device)


def end_detect(ended_hyps, i, M=3, D_end=np.log(1 * np.exp(-10))):
    """End detection.

    described in Eq. (50) of S. Watanabe et al
    "Hybrid CTC/Attention Architecture for End-to-End Speech Recognition"

    :param ended_hyps:
    :param i:
    :param M:
    :param D_end:
    :return:
    """
    if len(ended_hyps) == 0:
        return False
    count = 0
    best_hyp = sorted(ended_hyps, key=lambda x: x["score"], reverse=True)[0]
    for m in range(M):
        # get ended_hyps with their length is i - m
        hyp_length = i - m
        hyps_same_length = [x for x in ended_hyps if len(x["yseq"]) == hyp_length]
        if len(hyps_same_length) > 0:
            best_hyp_same_length = sorted(
                hyps_same_length, key=lambda x: x["score"], reverse=True
            )[0]
            if best_hyp_same_length["score"] - best_hyp["score"] < D_end:
                count += 1

    if count == M:
        return True
    else:
        return False



def th_accuracy(pad_outputs, pad_targets, ignore_label):
    """Calculate accuracy.

    Args:
        pad_outputs (Tensor): Prediction tensors (B * Lmax, D).
        pad_targets (LongTensor): Target label tensors (B, Lmax, D).
        ignore_label (int): Ignore label id.

    Returns:
        float: Accuracy value (0.0 - 1.0).

    """
    pad_pred = pad_outputs.view(
        pad_targets.size(0), pad_targets.size(1), pad_outputs.size(1)
    ).argmax(2)
    mask = pad_targets != ignore_label
    numerator = torch.sum(
        pad_pred.masked_select(mask) == pad_targets.masked_select(mask)
    )
    denominator = torch.sum(mask)
    return float(numerator) / float(denominator)


class AttrDict(dict):
    """
    Dictionary subclass whose keys can be accessed as attributes.
    
    Example:
        config = AttrDict({'model': {'lr': 0.01}})
        print(config.model.lr)  # Prints 0.01
    """
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
        
        # Convert nested dictionaries to AttrDict as well
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = AttrDict(value)