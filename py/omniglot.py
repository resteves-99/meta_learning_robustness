"""Dataloading for Omniglot."""
import os
import glob

import google_drive_downloader as gdd
import numpy as np
import torch
from torch.utils.data import dataset, sampler, dataloader
import imageio
import pdb

def load_image(file_path, repeat=False):
    """Loads and transforms an Omniglot image.

    Args:
        file_path (str): file path of image

    Returns:
        a Tensor containing image data
            shape (1, 28, 28)
    """
    # Not the best idea to do augmentation on this because images are already 28x28. Start with fungi or smth
    x = imageio.imread(file_path)
    x = torch.tensor(x)
    x = x.reshape([1, 28, 28])
    x = x.type(torch.float)
    x = x / 255.0

    if repeat: x = x.expand(3, -1, -1)

    return 1 - x

TOTAL_CLASSES = 1623
class OmniglotDataset(dataset.Dataset):
    """Omniglot dataset for meta-learning.

    Each element of the dataset is a task. A task is specified with a key,
    which is a tuple of class indices (no particular order). The corresponding
    value is the instantiated task, which consists of sampled (image, label)
    pairs.
    """

    _BASE_PATH = './data/omniglot_resized/'
    _GDD_FILE_ID = '1iaSFXIYC3AB8q9K_M-oVMa4pmB7yKMtI'

    # Omniglot constants
    NUM_TRAIN_CLASSES = 1100
    NUM_VAL_CLASSES = 100
    NUM_TEST_CLASSES = 423
    NUM_SAMPLES_PER_CLASS = 20

    def __init__(self, num_support, num_query):
        """Inits OmniglotDataset.

        Args:
            num_support (int): number of support examples per class
            num_query (int): number of query examples per class
        """
        super().__init__()


        # if necessary, download the Omniglot dataset
        if not os.path.isdir(self._BASE_PATH):
            gdd.GoogleDriveDownloader.download_file_from_google_drive(
                file_id=self._GDD_FILE_ID,
                dest_path=f'{self._BASE_PATH}.zip',
                unzip=True
            )

        # get all character folders
        self._character_folders = glob.glob(
            os.path.join(self._BASE_PATH, '*/*/'))
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
        # print("A")

        for label, class_idx in enumerate(class_idxs):
            # get a class's examples and sample from them
            all_file_paths = glob.glob(
                os.path.join(self._character_folders[class_idx], '*.png')
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

        # print("B")

        return images_support, labels_support, images_query, labels_query


# class OmniglotSampler(sampler.Sampler):
#     """Samples task specification keys for an OmniglotDataset."""

#     def __init__(self, split_idxs, num_way, num_tasks):
#         """Inits OmniglotSampler.

#         Args:
#             split_idxs (range): indices that comprise the
#                 training/validation/test split
#             num_way (int): number of classes per task
#             num_tasks (int): number of tasks to sample
#         """
#         super().__init__(None)
#         self._split_idxs = split_idxs
#         self._num_way = num_way
#         self._num_tasks = num_tasks

#     def __iter__(self):
#         return (
#             np.random.default_rng().choice(
#                 self._split_idxs,
#                 size=self._num_way,
#                 replace=False
#             ) for _ in range(self._num_tasks)
#         )

#     def __len__(self):
#         return self._num_tasks
