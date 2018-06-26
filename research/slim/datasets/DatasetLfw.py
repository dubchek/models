from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import DatasetCustom

from datasets import dataset_utils
from tensorflow.python.framework import ops


class DatasetLfw(DatasetCustom):
    def get_people(self, split_name):
        people_list_file = os.path.join(self.dataset_dir, + 'people_%s.txt' % split_name)
        people = []
        with open(people_list_file, 'r') as f:
            for line in f.readlines()[1:]:
                sample = line.strip().split()
                people.append(sample)
        return people

    def get_paths_and_labels(self, split_name):
        people = self.get_people(split_name)
        self.labels_to_classnames = {}
        self.classnames_to_labels = {}
        nrof_skipped_pairs = 0
        path_list = []
        labels_list = []
        for person in people:
            if len(person) == 2:
                path = os.path.join(self.dataset_dir, person[0], person[0] + '_' + '%04d' % int(person[1]) + '.jpg')
            else:
                print('WARN: length of the record is longer than expected')

            if os.path.exists(path):  # Only add the pair if both paths exist
                path_list.append(path)
                if person not in self.classnames_to_labels:
                    self.classnames_to_labels[person] = len(self.labels_to_classnames + 1)
                    self.labels_to_classnames[self.classnames_to_labels[person]] = person
                labels_list.append(self.classnames_to_labels[person])
            else:
                nrof_skipped_pairs += 1
        if nrof_skipped_pairs > 0:
            print('Skipped %d image pairs' % nrof_skipped_pairs)
        self.num_classes = len(self.classnames_to_labels)
        return path_list, labels_list