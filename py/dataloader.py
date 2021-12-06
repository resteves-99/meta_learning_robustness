"""Dataloading for Omniglot."""
import os
import glob

import numpy as np
import torch
from torch.utils.data import dataset, sampler, dataloader

from omniglot import OmniglotDataset
from quickdraw import QuickDrawDataset
from fungi import FungiDataset
from flowers import FlowersDataset
from flowers_unsup import FlowersDatasetUnsup
from fungi_unsup import FungiDatasetUnsup
from flowers_fungi_combo import FungiFlowersUnsupDataset, FlowersFungiUnsupDataset

class DataSampler(sampler.Sampler):
    """Samples task specification keys for an OmniglotDataset."""

    def __init__(self, split_idxs, num_way, num_tasks):
        """Inits OmniglotSampler.

        Args:
            split_idxs (range): indices that comprise the
                training/validation/test split
            num_way (int): number of classes per task
            num_tasks (int): number of tasks to sample
        """
        super().__init__(None)
        self.split_idxs = split_idxs
        self._num_way = num_way
        self._num_tasks = num_tasks

    def __iter__(self):
        return (
            np.random.default_rng().choice(
                self.split_idxs,
                size=self._num_way,
                replace=False
            ) for _ in range(self._num_tasks)
        )

    def __len__(self):
        return self._num_tasks

def identity(x):
    return x

def get_dataloader(
        dataset,
        split,
        batch_size,
        num_way,
        num_support,
        num_query,
        num_tasks_per_epoch,
        num_workers=2
):
        use_dataset = None
        if dataset == "omniglot":
            use_dataset = OmniglotDataset
        elif dataset == "quickdraw":
            use_dataset = QuickDrawDataset
        elif dataset == "fungi":
            use_dataset = FungiDataset
        elif dataset == "flowers":
            use_dataset = FlowersDataset
        elif dataset == "flowers_unsup":
            if split == "train":
                use_dataset = FlowersDatasetUnsup
            else:
                use_dataset = FlowersDataset
                num_support = 1
        elif dataset == "fungi_unsup":
            if split == "train":
                use_dataset = FungiDatasetUnsup
            else:
                use_dataset = FungiDataset
                num_support = 1
                num_query = 5
        # not the best coding but its okay
        elif dataset == "fungi+flowers_unsup":
            if split == "train":
                use_dataset = FungiFlowersUnsupDataset
            else:
                use_dataset = FlowersDataset
                num_support = 1
        elif dataset == "flowers+fungi_unsup":
            if split == "train":
                use_dataset = FlowersFungiUnsupDataset
            else:
                use_dataset = FungiDataset
                num_support = 1
                num_query = 5
        

        return get_dataset_dataloader(
                split,
                batch_size,
                num_way,
                num_support,
                num_query,
                num_tasks_per_epoch,
                use_dataset,
                num_workers=num_workers 
            )

def get_dataset_dataloader(
        split,
        batch_size,
        num_way,
        num_support,
        num_query,
        num_tasks_per_epoch, 
        dataset, 
        num_workers=2,
        data_sampler=DataSampler
):
    """Returns a dataloader.DataLoader for Omniglot.

    Args:
        split (str): one of 'train', 'val', 'test'
        batch_size (int): number of tasks per batch
        num_way (int): number of classes per task
        num_support (int): number of support examples per class
        num_query (int): number of query examples per class
        num_tasks_per_epoch (int): number of tasks before DataLoader is
            exhausted
    """

    if split == 'train':
        split_idxs = range(dataset.NUM_TRAIN_CLASSES)
    elif split == 'val':
        split_idxs = range(
            dataset.NUM_TRAIN_CLASSES,
            dataset.NUM_TRAIN_CLASSES \
                + dataset.NUM_VAL_CLASSES
        )
    elif split == 'test':
        split_idxs = range(
            dataset.NUM_TRAIN_CLASSES + dataset.NUM_VAL_CLASSES,
            dataset.NUM_TRAIN_CLASSES \
                + dataset.NUM_VAL_CLASSES \
                + dataset.NUM_TEST_CLASSES
        )
    else:
        raise ValueError



    return dataloader.DataLoader(
        dataset=dataset(num_support, num_query),
        batch_size=batch_size,
        sampler=data_sampler(split_idxs, num_way, num_tasks_per_epoch),
        num_workers=num_workers,
        collate_fn=identity,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )

