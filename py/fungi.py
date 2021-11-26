"""Dataloading for Fungi Dataset."""
import os
import glob

import numpy as np
import torch
from torch.utils.data import dataset, sampler, dataloader
import imageio
from math import floor, ceil
import pdb


def load_image(file_path):
    """Loads and transforms an Fungi image.

    Args:
        file_path (str): file path of image

    Returns:
        a Tensor containing image data
            shape (3, variable, variable)
    """
    x = imageio.imread(file_path)
    # W, H, C
    x = torch.tensor(x)
    x = x.transpose(2, 0)
    x = x.type(torch.float)
    x = x / 255.0
    x = 1 - x

    max_size = 1200 # estimate, might need to be higher
    pad_height, pad_width = max_size-x.shape[1], max_size-x.shape[2]
    pad_input = (floor(pad_width/2), ceil(pad_width/2), floor(pad_height/2), ceil(pad_height/2))
    x = torch.nn.funcional.pad(x, pad_input)
    return x


TOTAL_CLASSES = 1394


class FungiDataset(dataset.Dataset):
    """Fungi dataset for meta-learning.

    Each element of the dataset is a task. A task is specified with a key,
    which is a tuple of class indices (no particular order). The corresponding
    value is the instantiated task, which consists of sampled (image, label)
    pairs.
    """

    _BASE_PATH = './data/fungi/images/'
    _GDD_FILE_ID = '1iaSFXIYC3AB8q9K_M-oVMa4pmB7yKMtI'

    # Fungi constants
    # 1394 total classes
    NUM_TRAIN_CLASSES = 800
    NUM_VAL_CLASSES = 200
    NUM_TEST_CLASSES = 394
    NUM_SAMPLES_PER_CLASS = 20  # probably more than this

    def __init__(self, num_support, num_query):
        """Inits OmniglotDataset.

        Args:
            num_support (int): number of support examples per class
            num_query (int): number of query examples per class
        """
        super().__init__()

        # get all character folders
        self._character_folders = glob.glob(
            os.path.join(self._BASE_PATH, '*/'))
        assert len(self._character_folders) == TOTAL_CLASSES

        # shuffle characters
        np.random.default_rng(0).shuffle(self._character_folders)

        # check problem arguments
        assert num_support + num_query <= self.NUM_SAMPLES_PER_CLASS
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

        for label, class_idx in enumerate(class_idxs):
            # get a class's examples and sample from them
            all_file_paths = glob.glob(
                os.path.join(self._character_folders[class_idx], '*.JPG')
            )
            sampled_file_paths = np.random.default_rng().choice(
                all_file_paths,
                size=self._num_support + self._num_query,
                replace=False
            )
            images = [load_image(file_path) for file_path in sampled_file_paths]

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

        return images_support, labels_support, images_query, labels_query
