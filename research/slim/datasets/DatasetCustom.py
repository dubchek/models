from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

class custom_dataset:
    def __init__(self, dataset_dir):    # Constructor of the class
        self.dataset_dir = dataset_dir
        self.classnames_to_labels = {}
        self.labels_to_classnames = {}
        self.num_classes = 0
    def get_paths_and_labels(self, split_name):              # Abstract method, defined by convention only
        raise NotImplementedError("Subclass must implement abstract method")