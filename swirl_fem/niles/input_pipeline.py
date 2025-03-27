# Copyright 2024 The swirl_fem Authors.
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

"""Navier-Stokes filtered DNS input pipeline."""

from functools import partial

from absl import logging
from etils import epath
from absl import flags
import h5py
import jax
import ml_collections
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


NUM_SPLIT_EXAMPLES = dict([
    (tfds.Split.TRAIN, 6400 * 24),
    (tfds.Split.VALIDATION, 6400 * 4),
])

_DATASET_DIR = flags.DEFINE_string(
    'dataset_dir',
    '',
    'This flag is used for specifying the dataset directory.',
)

_TRAIN_SHARDS = list(range(0, 24))
_VALID_SHARDS = list(range(24, 28))

_DATASET_FILENAME = 'filtered_dns_shard_{:02}.hdf5'


def get_num_examples(
    split_name: tfds.Split, window_size: int, window_stride: int, debug: bool
):
  """Returns the total number of examples."""
  num_examples = NUM_SPLIT_EXAMPLES[split_name]
  if debug:
    if split_name == tfds.Split.TRAIN:
      num_examples //= 24
    else:
      num_examples //= 4

  return (num_examples - window_size) // window_stride + 1


def read_snapshots_file(file_path: epath.PathLike) -> dict[str, np.ndarray]:
  path = epath.Path(file_path)
  snapshots = {}
  with path.open('rb') as f:
    with h5py.File(f, 'r') as dataset_file:
      for key, data in dataset_file.items():
        snapshots[key] = np.array(data, dtype=np.float64)
  return snapshots


def _read_and_parse(
    file_path: epath.PathLike,
    window_size: int,
    window_stride: int,
    window_step: int,
):
  """Reads in snapshots from the filepath and windows them."""
  snapshots = read_snapshots_file(file_path)

  def _windowed(arr):
    windows = []
    for i in range(0, len(arr), window_stride):
      if i + window_size > len(arr):
        break
      windows.append(arr[i:i + window_size:window_step])
    return np.stack(windows)

  return jax.tree_map(_windowed, snapshots)  # pytype: disable=module-attr


def create_split(
    batch_size: int, train: bool, config: ml_collections.FrozenConfigDict):
  """Creates a split from the dataset.

  Args:
    batch_size: the batch size returned by the data pipeline.
    train: Whether to load the train or evaluation split.
    config: The training configuration.

  Returns:
    A `tf.data.Dataset`.
  """
  dataset_dir = epath.Path(_DATASET_DIR.value)
  logging.info('input pipeline creating split. debug: %s', str(config.debug))

  if train:
    split_name = tfds.Split.TRAIN
    shards = _TRAIN_SHARDS
    window_size = config.train_window_size
    window_stride = config.train_window_stride
  else:
    split_name = tfds.Split.VALIDATION
    shards = _VALID_SHARDS
    window_size = config.eval_window_size
    window_stride = config.eval_window_stride

  filenames = [dataset_dir / _DATASET_FILENAME.format(i) for i in shards]
  # in debug mode, just read from one file to reduce the data volume
  if config.debug:
    filenames = [filenames[0]]

  snapshots_list = list(map(
      partial(_read_and_parse,
              window_size=window_size, window_stride=window_stride,
              window_step=config.window_step), filenames))

  snapshots = {}
  for key in ['u', 'p', 't']:
    snapshots[key] = np.concatenate([s[key] for s in snapshots_list])

  logging.info('snapshots timestamps: %s', snapshots['t'][:100])

  num_examples = get_num_examples(
      split_name, window_size=window_size, window_stride=window_stride,
      debug=config.debug)
  logging.info('read num_examples %d for split %s', num_examples, split_name)
  assert num_examples % jax.process_count() == 0

  split_size = num_examples // jax.process_count()
  split_start = jax.process_index() * split_size
  split_snapshots = {}
  for key, value in snapshots.items():
    split_snapshots[key] = value[split_start:split_start + split_size]

  ds = tf.data.Dataset.from_tensor_slices(split_snapshots)
  options = tf.data.Options()
  options.experimental_threading.private_threadpool_size = 48
  ds = ds.with_options(options)

  if config.cache:
    ds = ds.cache()

  # shuffle and repeat both train and eval datasets
  ds = ds.shuffle(8 * batch_size, seed=0)
  ds = ds.batch(batch_size, drop_remainder=True)
  ds = ds.repeat()

  ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
  return ds
