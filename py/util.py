"""Utilities for scoring the model."""
import torch


def get_num_input_channels(args):
    if args.dataset in ('omniglot', "quickdraw"):
        return 1
    return 3

def get_num_hidden_channels(args):
    if args.dataset in ('fungi', 'flowers'):
        return 64 #32
    return 64


def score(logits, labels):
    """Returns the mean accuracy of a model's predictions on a set of examples.

    Args:
        logits (torch.Tensor): model predicted logits
            shape (examples, classes)
        labels (torch.Tensor): classification labels from 0 to num_classes - 1
            shape (examples,)
    """

    assert logits.dim() == 2
    assert labels.dim() == 1
    assert logits.shape[0] == labels.shape[0]
    y = torch.argmax(logits, dim=-1) == labels
    y = y.type(torch.float)
    return torch.mean(y).item()
