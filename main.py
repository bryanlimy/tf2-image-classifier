import time
import pathlib
import tensorflow as tf
import tensorflow.keras as keras

from utils import Logger, get_hparams, get_image_paths


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


def get_images_ds(hparams):

  def parse(x):
    result = tf.io.parse_tensor(x, out_type=tf.float32)
    result = tf.reshape(result, [192, 192, 3])
    return result

  images_ds = tf.data.TFRecordDataset(hparams.train_record)
  return images_ds.map(parse, num_parallel_calls=tf.data.experimental.AUTOTUNE)


def get_ds(hparams):
  images_ds, labels_ds = get_images_ds(hparams), get_labels_ds(hparams)
  ds = tf.data.Dataset.zip((images_ds, labels_ds))
  ds = ds.shuffle(buffer_size=hparams.num_images)
  ds = ds.batch(hparams.batch_size)
  ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
  return ds


def get_model(hparams):
  mobile_net = tf.keras.applications.MobileNetV2(
      input_shape=(192, 192, 3), include_top=False)

  mobile_net.trainable = False

  model = tf.keras.Sequential([
      mobile_net,
      tf.keras.layers.GlobalAveragePooling2D(),
      tf.keras.layers.Dense(hparams.num_classes, activation='softmax')
  ])

  return model


#@tf.function
def train_step(features, labels, model, optimizer, loss_fn):
  with tf.GradientTape() as tape:
    predictions = model(features, training=True)
    loss = loss_fn(labels, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  return loss, predictions


#@tf.function
def test_step(features, labels, model, loss_fn):
  predictions = model(features, training=False)
  loss = loss_fn(labels, predictions)
  return loss, predictions


def train_and_test(hparams):
  model = get_model(hparams)
  optimizer = keras.optimizers.Adam(lr=hparams.learning_rate)
  loss_fn = keras.losses.SparseCategoricalCrossentropy()
  logger = Logger(hparams, optimizer)

  dataset = get_ds(hparams)

  for epoch in range(hparams.epochs):

    start = time.time()

    for images, labels in dataset:
      loss, predictions = train_step(images, labels, model, optimizer, loss_fn)
      logger.log_progress(loss, labels, predictions, mode='train')

    elapse = time.time() - start

    logger.write_scalars(mode='train')

    # for images, labels in datasets['test']:
    #   loss, predictions = test_step(images, labels, model, loss_fn)
    #   logger.log_progress(loss, labels, predictions, mode='test')

    # logger.write_scalars(mode='test', elapse=elapse)
    logger.print_progress(epoch, elapse)

  tf.keras.models.save_model(model, filepath=hparams.output_dir)


def main():
  hparams = get_hparams()
  hparams.num_images = len(get_image_paths(hparams))
  hparams.num_classes = 5

  train_and_test(hparams)


if __name__ == "__main__":
  main()
