import tensorflow as tf
import pathlib

from utils import get_image_paths, get_hparams


def preprocess_image(image):
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.image.resize(image, [192, 192])
  # normalize image to [-1, -1]
  image = (image / 127.5) - 1
  return image


def load_and_preprocess_image(path):
  image = tf.io.read_file(path)
  return preprocess_image(image)


def get_labels_ds(hparams):
  label_names = sorted(
      item.name
      for item in pathlib.Path(hparams.data_dir).glob('*/')
      if item.is_dir())

  label_to_index = dict((name, index) for index, name in enumerate(label_names))

  all_image_labels = [
      label_to_index[pathlib.Path(path).parent.name]
      for path in get_image_paths(hparams)
  ]

  return tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))


def main():
  hparams = get_hparams()

  if tf.io.gfile.exists(hparams.train_record):
    tf.io.gfile.remove(hparams.train_record)

  all_image_paths = get_image_paths(hparams)
  paths_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
  images_ds = paths_ds.map(load_and_preprocess_image)
  images_ds = images_ds.map(tf.io.serialize_tensor)

  labels_ds = get_labels_ds(hparams)
  labels_ds = labels_ds.map(tf.io.serialize_tensor)

  ds = tf.data.Dataset.zip((images_ds, labels_ds))

  #ds = ds.map(tf.io.serialize_tensor)

  tfrecord = tf.data.experimental.TFRecordWriter(hparams.train_record)
  tfrecord.write(ds)

  print(
      'written %d images to %s' % (len(all_image_paths), hparams.train_record))


if __name__ == "__main__":
  main()
