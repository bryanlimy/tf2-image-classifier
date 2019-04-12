import os
import pathlib
import tensorflow as tf
from sklearn.model_selection import train_test_split

from utils import get_hparams


def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy(
    )  # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def get_paths(data_dir):
  return pathlib.Path(data_dir)


def get_images(paths):
  return [
      str(path)
      for path in list(paths.glob('*/*'))
      if str(path).lower().endswith('.jpg') and os.stat(str(path)).st_size > 0
  ]


def get_labels(paths, images_path):
  label_names = sorted(item.name for item in paths.glob('*/') if item.is_dir())

  label_to_index = dict((name, index) for index, name in enumerate(label_names))

  return [
      label_to_index[pathlib.Path(path).parent.name] for path in images_path
  ]


def create_tfrecord(images, labels, hparams, train=True):
  record_path = hparams.train_record if train else hparams.test_record

  if tf.io.gfile.exists(record_path):
    tf.io.gfile.remove(record_path)

  def create_tf_example(filename, label):
    feature = {
        'image': _bytes_feature(tf.io.read_file(filename)),
        'label': _int64_feature(label)
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

  count = 0
  with tf.io.TFRecordWriter(record_path) as writer:
    for filename, label in zip(images, labels):
      tf_example = create_tf_example(filename, label)
      writer.write(tf_example.SerializeToString())
      count += 1

  print('%d data points stored to %s' % (count, record_path))


def main():
  hparams = get_hparams()

  paths = get_paths(hparams.data_dir)

  images_path = get_images(paths)
  labels = get_labels(paths, images_path)

  X_train, X_test, y_train, y_test = train_test_split(
      images_path, labels, test_size=0.3, shuffle=True)

  create_tfrecord(X_train, y_train, hparams, train=True)
  create_tfrecord(X_test, y_test, hparams, train=False)


if __name__ == "__main__":
  main()
