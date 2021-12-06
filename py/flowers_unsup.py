import scipy.io
import numpy as np
import os
import os
import glob

import numpy as np
import torch
from torch.utils.data import dataset, sampler, dataloader
import imageio
import pdb
import torchvision.transforms as transforms
from math import floor, ceil

img_dim = 224
div_fac = 8
flowers_transform_augmented = transforms.Compose([
    transforms.RandomResizedCrop(img_dim//div_fac),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
    transforms.RandomGrayscale(p=0.1),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_image(file_path, num_augs=1):
    """Loads and transforms an Fungi image.

    Args:
        file_path (str): file path of image

    Returns:
        a Tensor containing image data
            shape (3, variable, 500)
    """
    x = imageio.imread(file_path)
    # W, H, C
    x = torch.tensor(x)
    x = x.transpose(2, 0)
    x = x.type(torch.float)
    # x = x / 255.0

    if num_augs == 1:
        return flowers_transform_augmented(x)
    return [flowers_transform_augmented(x) for _ in range(num_augs)]

TOTAL_CLASSES = 102
TOTAL_SAMPLES = 8189
class FlowersDatasetUnsup(dataset.Dataset):
    """Flowers dataset for meta-learning.

    Each element of the dataset is a task. A task is specified with a key,
    which is a tuple of class indices (no particular order). The corresponding
    value is the instantiated task, which consists of sampled (image, label)
    pairs.
    """

    _BASE_PATH = './data/flowers/chars/'

    # Fungi constants
    # 102 total classes
    NUM_TRAIN_CLASSES = 62
    NUM_VAL_CLASSES = 20
    NUM_TEST_CLASSES = 20
    NUM_SAMPLES_PER_CLASS = 40 # probably more than this

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
        # Sanity check
        assert len(self._character_folders) == TOTAL_CLASSES
        
        # get all character folders
        self._image_paths = glob.glob(
            os.path.join(self._BASE_PATH, '*/*.jpg'))

        assert len(self._image_paths) == TOTAL_SAMPLES

        # shuffle characters
        np.random.default_rng(0).shuffle(self._image_paths)

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

        sampled_file_paths = np.random.default_rng().choice(
            self._image_paths,
            size=len(class_idxs),
            replace=False
        )

        for label, file_path in enumerate(sampled_file_paths):
            # get a sample augment it num_query + num_support times
            images = load_image(file_path, num_augs = self._num_support + self._num_query)

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

