"""Provides data for the faces dataset.

The dataset scripts used to create the dataset can be found at:
tensorflow/models/slim/datasets/download_and_convert_faces.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

from datasets import dataset_utils

slim = tf.contrib.slim

_FILE_PATTERN = 'faces_%s_*.tfrecord'

SPLITS_TO_SIZES = {}#{'train': 3320, 'validation': 350}

_NUM_CLASSES = 5

_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying size.',
    'label': 'A single integer between 0 and 4',
}


def get_split(split_name, dataset_dir, file_pattern=None, reader=None):
  """Gets a dataset tuple with instructions for reading faces.

  Args:
    split_name: A train/validation split name.
    dataset_dir: The base directory of the dataset sources.
    file_pattern: The file pattern to use when matching the dataset sources.
      It is assumed that the pattern contains a '%s' string so that the split
      name can be inserted.
    reader: The TensorFlow reader type.

  Returns:
    A `Dataset` namedtuple.

  Raises:
    ValueError: if `split_name` is not a valid train/validation split.
  """

  trainset_file = open(os.path.join(dataset_dir, 'train_set.txt'), 'r')
  validationset_file = open(os.path.join(dataset_dir, 'validation_set.txt'), 'r')
  SPLITS_TO_SIZES['train'] = sum(1 for line in trainset_file)
  SPLITS_TO_SIZES['validation'] = sum(1 for line in validationset_file)

  print('Faces: Train set size is %s, validation set size is %s' %
        (SPLITS_TO_SIZES['train'], SPLITS_TO_SIZES['validation']))

  if split_name not in SPLITS_TO_SIZES:
    raise ValueError('split name %s was not recognized.' % split_name)

  if not file_pattern:
    file_pattern = _FILE_PATTERN
  file_pattern = os.path.join(dataset_dir, file_pattern % split_name)

  # Allowing None in the signature so that dataset_factory can use the default.
  if reader is None:
    reader = tf.TFRecordReader

  keys_to_features = {
      'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
      'image/format': tf.FixedLenFeature((), tf.string, default_value='png'),
      'image/class/label': tf.FixedLenFeature(
          [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
  }

  items_to_handlers = {
      'image': slim.tfexample_decoder.Image(),
      'label': slim.tfexample_decoder.Tensor('image/class/label'),
  }

  decoder = slim.tfexample_decoder.TFExampleDecoder(
      keys_to_features, items_to_handlers)

  labels_to_names = None
  if dataset_utils.has_labels(dataset_dir):
    labels_to_names = dataset_utils.read_label_file(dataset_dir)

  _NUM_CLASSES = len(labels_to_names)

  print("Faces: Call slim.dataset.Dataset() with parameters:"
        "\r\n\tdata_sources=%s"
        "\r\n\treader=reader"
        "\r\n\tdecoder=decoder"
        "\r\n\tnum_samples=%s"
        "\r\n\titems_to_descriptions=%s"
        "\r\n\tnum_classes=%s"
        "\r\n\tlabels_to_names=%s)" %
        (file_pattern,
      SPLITS_TO_SIZES[split_name],
      _ITEMS_TO_DESCRIPTIONS,
      _NUM_CLASSES,
      labels_to_names))

  return slim.dataset.Dataset(
      data_sources=file_pattern,
      reader=reader,
      decoder=decoder,
      num_samples=SPLITS_TO_SIZES[split_name],
      items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
      num_classes=_NUM_CLASSES,
      labels_to_names=labels_to_names)

