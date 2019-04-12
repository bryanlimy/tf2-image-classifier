import tensorflow as tf
import pathlib

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
  return [str(path) for path in list(paths.glob('*/*'))]


def get_labels(paths, images_path):
  label_names = sorted(item.name for item in paths.glob('*/') if item.is_dir())

  label_to_index = dict((name, index) for index, name in enumerate(label_names))

  return [
      label_to_index[pathlib.Path(path).parent.name] for path in images_path
  ]


def main(train=True):
  hparams = get_hparams()

  record_path = hparams.train_record if train else hparams.test_record

  if tf.io.gfile.exists(record_path):
    tf.io.gfile.remove(record_path)

  paths = get_paths(hparams.data_dir)

  images_path = get_images(paths)
  labels = get_labels(paths, images_path)

  def create_tf_example(filename, label):
    feature = {
        'image': _bytes_feature(tf.io.read_file(filename)),
        'label': _int64_feature(label)
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

  count = 0
  with tf.io.TFRecordWriter(record_path) as writer:
    for filename, label in zip(images_path, labels):
      tf_example = create_tf_example(filename, label)
      writer.write(tf_example.SerializeToString())
      count += 1

  print('%d data points stored to %s' % (count, record_path))


if __name__ == "__main__":
  main()
