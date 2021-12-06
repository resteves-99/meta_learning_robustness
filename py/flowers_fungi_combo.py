import os
import glob

import pdb
from math import floor, ceil
import random
from fungi import FungiDataset
from flowers import FlowersDataset
from flowers_unsup import FlowersDatasetUnsup
from fungi_unsup import FungiDatasetUnsup

class FlowersFungiUnsupDataset(dataset.Dataset):
    def __init__(self, num_support, num_query):
        super().__init__()
        
        self.threshold = 0.5
        self.flowers = FlowersDataset(num_support, num_query)
        self.fungi_unsup = FungiDatasetUnsup(num_support, num_query)

    def __getitem__(self, class_idxs):

        if random.random() > self.threshold:
            return self.flowers[class_idxs]
        return self.fungi_unsup[class_idxs]

class FungiFlowersUnsupDataset(dataset.Dataset):
    def __init__(self, num_support, num_query):
        super().__init__()
        
        self.threshold = 0.5
        self.flowers_unsup = FlowersDatasetUnsup(num_support, num_query)
        self.fungi = FungiDataset(num_support, num_query)

    def __getitem__(self, class_idxs):

        if random.random() > self.threshold:
            return self._unsup[class_idxs]
        return self.fungi[class_idxs]