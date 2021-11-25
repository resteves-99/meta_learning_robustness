"""Dataloading for Quick Draw."""
import os
import glob

import google_drive_downloader as gdd
import numpy as np
import torch
from torch.utils.data import dataset, sampler, dataloader
from functools import lru_cache
import pdb

@lru_cache(maxsize=None)
def load_npy(file_path):
    return np.load(file_path)

def load_image_npy(file_path, num_images):
    """Loads and transforms an Omniglot image.

    Args:
        file_path (str): file path of image

    Returns:
        a Tensor containing image data
            shape (1, 28, 28)
    """
    x = load_npy(file_path)
    x = x[np.random.choice(range(len(x)), size=num_images)]
    x = torch.tensor(x)
    x = x.reshape([num_images, 1, 28, 28])
    x = x.type(torch.float)
    x = x / 255.0
    return x # Note: x is not inverted, so no need for 1 - x

TOTAL_CLASSES = 345
class QuickDrawDataset(dataset.Dataset):
    """QuickDraw dataset for meta-learning.

    Each element of the dataset is a task. A task is specified with a key,
    which is a tuple of class indices (no particular order). The corresponding
    value is the instantiated task, which consists of sampled (image, label)
    pairs.
    """

    _BASE_PATH = './data/mini_quickdraw/'

    # Quickdraw Constants (total classes = 345)
    NUM_TRAIN_CLASSES = 241
    NUM_VAL_CLASSES = 52
    NUM_TEST_CLASSES = 52

    def __init__(self, num_support, num_query):
        """Inits QuickDrawDataset.

        Args:
            num_support (int): number of support examples per class
            num_query (int): number of query examples per class
        """
        super().__init__()

        # get all character folders
        self.all_file_paths = glob.glob(
            os.path.join(self._BASE_PATH, '*'))
        assert len(self.all_file_paths) == TOTAL_CLASSES

        # shuffle characters
        np.random.default_rng(0).shuffle(self.all_file_paths)

        # check problem arguments
        self._num_support = num_support
        self._num_query = num_query

    def __getitem__(self, class_idxs):
        """Constructs a task.

        Data for each class is sampled uniformly at random without replacement.
        The ordering of the labels corresponds to that of class_idxs.

        Args:
            class_idxs (tuple[int]): class indices that comprise the task

        Returns:
            images_support (Tensor): task support images
                shape (num_way * num_support, channels, height, width)
            labels_support (Tensor): task support labels
                shape (num_way * num_support,)
            images_query (Tensor): task query images
                shape (num_way * num_query, channels, height, width)
            labels_query (Tensor): task query labels
                shape (num_way * num_query,)
        """
        images_support, images_query = [], []
        labels_support, labels_query = [], []
        # print("A")

        for label, class_idx in enumerate(class_idxs):
            # get a class's examples and sample from them
            file_path = self.all_file_paths[class_idx]
            images = load_image_npy(file_path, self._num_query+self._num_support)

            # split sampled examples into support and query
            images_support.extend(images[:self._num_support])
            images_query.extend(images[self._num_support:])
            labels_support.extend([label] * self._num_support)
            labels_query.extend([label] * self._num_query)

        # aggregate into tensors
        images_support = torch.stack(images_support)  # shape (N*S, C, H, W)
        labels_support = torch.tensor(labels_support)  # shape (N*S)
        images_query = torch.stack(images_query)
        labels_query = torch.tensor(labels_query)
        
        # print("B")

        return images_support, labels_support, images_query, labels_query


# class QuickDrawSampler(sampler.Sampler):
#     """Samples task specification keys for an OmniglotDataset."""

#     def __init__(self, split_idxs, num_way, num_tasks):
#         """Inits QuickDrawSampler.

#         Args:
#             split_idxs (range): indices that comprise the
#                 training/validation/test split
#             num_way (int): number of classes per task
#             num_tasks (int): number of tasks to sample
#         """
#         super().__init__(None)
#         self._num_way = num_way
#         self._num_tasks = num_tasks
#         self.split_idxs = split_idxs

#     def __iter__(self):
#         return (
#             np.random.default_rng().choice(
#                 self.split_idxs,
#                 size=self._num_way,
#                 replace=False
#             ) for _ in range(self._num_tasks)
#         )

#     def __len__(self):
#         return self._num_tasks

if __name__ == '__main__':
    tmp = np.load('./data/quickdraw/airplane.npy')
    tmp2 = np.load('./data/quickdraw/axe.npy')
    # tmp is a list of flattened images
    print(len(tmp))
    print(len(tmp2))
