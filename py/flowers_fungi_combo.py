import os
import glob

import pdb
from math import floor, ceil
import random
from torch.utils.data import dataset
from fungi import FungiDataset
from flowers import FlowersDataset
from flowers_unsup import FlowersDatasetUnsup
from fungi_unsup import FungiDatasetUnsup

class FlowersFungiUnsupDataset(dataset.Dataset):
    NUM_TRAIN_CLASSES = 62
    NUM_VAL_CLASSES = 20
    NUM_TEST_CLASSES = 20
    def __init__(self, num_support, num_query):
        super().__init__()
        
        self.threshold = 0.5
        self.flowers = FlowersDataset(1, num_query)
        self.fungi_unsup = FungiDatasetUnsup(num_support, num_query)

    def __getitem__(self, class_idxs):

        if random.random() > self.threshold:
            return self.flowers[class_idxs]
        return self.fungi_unsup[class_idxs]

class FungiFlowersUnsupDataset(dataset.Dataset):
    NUM_TRAIN_CLASSES = 800
    NUM_VAL_CLASSES = 200
    NUM_TEST_CLASSES = 394
    def __init__(self, num_support, num_query):
        super().__init__()
        
        self.threshold = 0.5
        self.flowers_unsup = FlowersDatasetUnsup(num_support, num_query)
        self.fungi = FungiDataset(1, 5)

    def __getitem__(self, class_idxs):

        if random.random() > self.threshold:
            return self.fungi[class_idxs]
        return self.flowers_unsup[class_idxs]