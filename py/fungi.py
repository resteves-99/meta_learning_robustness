"""Dataloading for Fungi Dataset."""
import os
import glob

import numpy as np
import torch
from torch.utils.data import dataset, sampler, dataloader
import imageio
from math import floor, ceil
import torchvision.transforms as transforms
import pdb

img_dim = 224
div_fac = 8
fungi_transform = transforms.Compose([
    transforms.Resize(256//div_fac),
    transforms.CenterCrop(img_dim//div_fac),
])

def load_image(file_path):
    """Loads and transforms an Fungi image.

    Args:
        file_path (str): file path of image

    Returns:
        a Tensor containing image data
            shape (3, variable, variable)
    """
    try:
        x = imageio.imread(file_path)
    except ValueError: 
        print(file_path)
        pdb.set_trace()
    
    # W, H, C
    try:
        x = torch.tensor(x)
    except ValueError:
        x = torch.tensor(x.copy())
    x = x.transpose(2, 0)
    x = x.type(torch.float)
    x = fungi_transform(x)
    x = x / 255.0

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

    # Fungi constants
    # 1394 total classes
    NUM_TRAIN_CLASSES = 800
    NUM_VAL_CLASSES = 200
    NUM_TEST_CLASSES = 394
    NUM_SAMPLES_PER_CLASS = 16  # probably more than this

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
            try:
                sampled_file_paths = np.random.default_rng().choice(
                    all_file_paths,
                    size=self._num_support + self._num_query,
                    replace=False
                )
            except ValueError:
                print(len(all_file_paths))
                print(all_file_paths)
                print(self._num_support + self._num_query)
                exit()
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

def clean_bad_images():
    from tqdm import tqdm
    _BASE_PATH = './data/fungi/images/'
    jpg_files = glob.glob(os.path.join(_BASE_PATH, '*/*.JPG'))
    for jpg_path in tqdm(jpg_files):
        
        try:
            _ = imageio.imread(jpg_path)
        except ValueError: 
            print(f"bad file {jpg_path}")
            new_path = jpg_path + "_BAD"
            os.rename(jpg_path, new_path)
            # new_path = jpg_path.replace("images", "bad_images")

    
if __name__ == "__main__":
    clean_bad_images()