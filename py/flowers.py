import scipy.io
import numpy as np
import os
import os
import glob

import google_drive_downloader as gdd
import numpy as np
import torch
from torch.utils.data import dataset, sampler, dataloader
import imageio
import pdb

def load_image(file_path):
    """Loads and transforms an Fungi image.

    Args:
        file_path (str): file path of image

    Returns:
        a Tensor containing image data
            shape (3, variable, 500)
    """
    x = imageio.imread(file_path)
    x = torch.tensor(x)
    # x = x.reshape([3, -1, 500])
    x = x.type(torch.float)
    x = x / 255.0
    return 1 - x

TOTAL_CLASSES = 102
class FlowersDataset(dataset.Dataset):
    """Flowers dataset for meta-learning.

    Each element of the dataset is a task. A task is specified with a key,
    which is a tuple of class indices (no particular order). The corresponding
    value is the instantiated task, which consists of sampled (image, label)
    pairs.
    """

    _BASE_PATH = './data/flowers/chars/'
    _GDD_FILE_ID = '1iaSFXIYC3AB8q9K_M-oVMa4pmB7yKMtI'

    # Fungi constants
    # 102 total classes
    NUM_TRAIN_CLASSES = 62
    NUM_VAL_CLASSES = 20
    NUM_TEST_CLASSES = 20
    NUM_SAMPLES_PER_CLASS = 20 # probably more than this

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


def rearrange_folders():
    # read files
    mat = scipy.io.loadmat('./data/flowers/imagelabels.mat')
    flower_labels = mat["labels"].tolist()
    flower_labels = flower_labels[0]
    unique_labels = np.unique(flower_labels).tolist()

    # create directories
    flower_path = './data/flowers/chars/'
    if not os.path.isdir(flower_path):
        os.mkdir(flower_path)
    for label_idx in range(len(unique_labels)):
        label_id = unique_labels[label_idx]
        curr_path = flower_path + f'{label_id}/'
        if not os.path.isdir(curr_path):
            os.mkdir(curr_path)

    # fill directories
    jpg_path = './data/flowers/images/'
    all_paths = os.listdir(jpg_path)
    all_paths.sort()
    for idx in range(len(all_paths)):
        curr_img_name = all_paths[idx]
        curr_img_path = jpg_path + curr_img_name
        curr_label = flower_labels[idx]
        new_img_path = flower_path + f'{curr_label}/{curr_img_name}'
        os.rename(curr_img_path, new_img_path)


if __name__ == '__main__':
    rearrange_folders()

