# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Downloads and converts Faces data to TFRecords of TF-Example protos.

This module downloads the Faces data, uncompresses it, reads the files
that make up the Faces data and creates two TFRecord datasets: one for train
and one for test. Each TFRecord dataset is comprised of a set of TF-Example
protocol buffers, each of which contain a single image and label.

The script should take about a minute to run.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys

import tensorflow as tf

from datasets import dataset_utils

# The URL where the Faces data can be downloaded.
_DATA_URLs = \
    {
        'lfw': 'http://vis-www.cs.umass.edu/lfw/lfw.tgz',
        'lfw_funneled': 'http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz',
        'lfw_deepfunneled': 'http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz'
    }

# The number of images in the validation set.
# _NUM_VALIDATION = 350

# Seed for repeatability.
_RANDOM_SEED = 0

# The number of shards per dataset split.
_NUM_SHARDS = 4


class ImageReader(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self):
        # Initializes function that decodes RGB JPEG data.
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(
            self._decode_jpeg_data, channels=3)

    def read_image_dims(self, sess, image_data):
        image = self.decode_jpeg(sess, image_data)
        return image.shape[0], image.shape[1]

    def decode_jpeg(self, sess, image_data):
        image = sess.run(self._decode_jpeg,
                         feed_dict={self._decode_jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image


# def _get_filenames_and_classes(dataset_dir):
#     """Returns a list of filenames and inferred class names.
#
#     Args:
#       dataset_dir: A directory containing a set of subdirectories representing
#         class names. Each subdirectory should contain PNG or JPG encoded images.
#
#     Returns:
#       A list of image file paths, relative to `dataset_dir` and the list of
#       subdirectories, representing class names.
#     """
#     face_root = os.path.join(dataset_dir, 'face_photos')
#     directories = []
#     class_names = []
#     for filename in os.listdir(face_root):
#         path = os.path.join(face_root, filename)
#         if os.path.isdir(path):
#             directories.append(path)
#             class_names.append(filename)
#
#     photo_filenames = []
#     for directory in directories:
#         for filename in os.listdir(directory):
#             path = os.path.join(directory, filename)
#             photo_filenames.append(path)
#
#     return photo_filenames, sorted(class_names)


def _get_train_test_and_classes(dataset_dir, min_images_per_class, test_set_size, random_seed):
    directories = []
    class_names = []
    for filename in os.listdir(dataset_dir):
        path = os.path.join(dataset_dir, filename)
        if os.path.isdir(path):
            directories.append(path)
            #class_names.append(filename)

    trainset = []
    testset = []

    for directory in directories:
        photo_filenames = []
        for filename in os.listdir(directory):
            path = os.path.join(directory, filename)
            photo_filenames.append(path)
        if len(photo_filenames) < min_images_per_class:
            continue
        class_names.append(os.path.basename(directory))
        split_index = round(len(photo_filenames) * test_set_size)
        random.seed(random_seed)
        random.shuffle(photo_filenames)
        testset += photo_filenames[:split_index]
        trainset += photo_filenames[split_index:]

    return trainset, testset, sorted(class_names)


def _get_dataset_filename(dataset_dir, split_name, shard_id):
    output_filename = 'faces_%s_%05d-of-%05d.tfrecord' % (
        split_name, shard_id, _NUM_SHARDS)
    return os.path.join(dataset_dir, output_filename)


def _convert_dataset(split_name, filenames, class_names_to_ids, dataset_dir):
    """Converts the given filenames to a TFRecord dataset.

    Args:
      split_name: The name of the dataset, either 'train' or 'validation'.
      filenames: A list of absolute paths to png or jpg images.
      class_names_to_ids: A dictionary from class names (strings) to ids
        (integers).
      dataset_dir: The directory where the converted datasets are stored.
    """
    assert split_name in ['train', 'validation']

    num_per_shard = int(math.ceil(len(filenames) / float(_NUM_SHARDS)))

    with tf.Graph().as_default():
        image_reader = ImageReader()

        with tf.Session('') as sess:

            for shard_id in range(_NUM_SHARDS):
                output_filename = _get_dataset_filename(
                    dataset_dir, split_name, shard_id)

                with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
                    start_ndx = shard_id * num_per_shard
                    end_ndx = min((shard_id + 1) *
                                  num_per_shard, len(filenames))
                    for i in range(start_ndx, end_ndx):
                        sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                            i + 1, len(filenames), shard_id))
                        sys.stdout.flush()

                        # Read the filename:
                        image_data = tf.gfile.FastGFile(
                            filenames[i], 'rb').read()
                        height, width = image_reader.read_image_dims(
                            sess, image_data)

                        class_name = os.path.basename(
                            os.path.dirname(filenames[i]))
                        class_id = class_names_to_ids[class_name]

                        example = dataset_utils.image_to_tfexample(
                            image_data, b'jpg', height, width, class_id)
                        tfrecord_writer.write(example.SerializeToString())

    sys.stdout.write('\n')
    sys.stdout.flush()


def _clean_up_temporary_files(dataset_dir):
    """Removes temporary files used to create the dataset.

    Args:
      dataset_dir: The directory where the temporary files are stored.
    """
    filename = _DATA_URL.split('/')[-1]
    filepath = os.path.join(dataset_dir, filename)
    tf.gfile.Remove(filepath)

    tmp_dir = os.path.join(dataset_dir, 'face_photos')
    tf.gfile.DeleteRecursively(tmp_dir)


def _dataset_exists(dataset_dir):
    for split_name in ['train', 'validation']:
        for shard_id in range(_NUM_SHARDS):
            output_filename = _get_dataset_filename(
                dataset_dir, split_name, shard_id)
            if not tf.gfile.Exists(output_filename):
                return False
    return True


def run(dataset_dir, dataset_home_dir, dataset_type='lfw', force_download=False, min_images_per_class=40,
        validation_set_fraction=0.25):
    """Runs the download and conversion operation.

    Args:
      dataset_dir: The dataset directory where the dataset is stored.
    """

    # Set default values for input parameters
    dataset_home_dir = dataset_home_dir if dataset_home_dir else dataset_dir
    dataset_type = dataset_type if dataset_type else 'lfw'
    min_images_per_class = min_images_per_class if min_images_per_class else 40
    validation_set_fraction = validation_set_fraction if validation_set_fraction else 0.25

    print('Input arguments:\r\n\tdataset_dir=%s\r\n\tdataset_home_dir=%s\r\n\tdataset_type=%s\r\n\tforce_download=%s'
          '\r\n\tmin_images_per_class=%s\r\n\tvalidation_set_fraction=%s'
          % (dataset_dir, dataset_home_dir, dataset_type, force_download, min_images_per_class,
        validation_set_fraction))

    if not tf.gfile.Exists(dataset_dir):
        tf.gfile.MakeDirs(dataset_dir)

    if _dataset_exists(dataset_dir):
        print('Dataset files already exist. Exiting without re-creating them.')
        return

    dataset_home_dir = os.path.join(dataset_home_dir, dataset_type)
    if  force_download or not os.path.exists(dataset_home_dir):
        dataset_utils.download_and_uncompress_tarball(_DATA_URLs[dataset_type], dataset_home_dir)

    # photo_filenames, class_names = _get_filenames_and_classes(dataset_dir)
    training_filenames, validation_filenames, class_names = \
        _get_train_test_and_classes(dataset_home_dir, min_images_per_class, validation_set_fraction, _RANDOM_SEED)

    # log train and validation sets
    f1 = open(os.path.join(dataset_dir, 'train_set.txt'), 'w')
    f1.write("\n".join(training_filenames))
    f1.close()

    f1 = open(os.path.join(dataset_dir, 'validation_set.txt'), 'w')
    f1.write("\n".join(validation_filenames))
    f1.close()

    class_names_to_ids = dict(zip(class_names, range(len(class_names))))

    # First, convert the training and validation sets.
    _convert_dataset('train', training_filenames, class_names_to_ids,
                     dataset_dir)
    _convert_dataset('validation', validation_filenames, class_names_to_ids,
                     dataset_dir)

    # Finally, write the labels file:
    labels_to_class_names = dict(zip(range(len(class_names)), class_names))
    dataset_utils.write_label_file(labels_to_class_names, dataset_dir)

    # _clean_up_temporary_files(dataset_dir)
    print('\nFinished converting the Faces dataset!')
