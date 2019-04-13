import time
import tensorflow as tf
import tensorflow.keras as keras

from utils import Logger, Checkpoint, get_hparams

AUTOTUNE = tf.data.experimental.AUTOTUNE


def get_dataset(hparams, train=False):
  record_file = hparams.train_record if train else hparams.eval_record
  ds = tf.data.TFRecordDataset(record_file)

  feature_description = {
      'image': tf.io.FixedLenFeature([], tf.string),
      'label': tf.io.FixedLenFeature([], tf.int64)
  }

  def parse_example(example_proto):
    example = tf.io.parse_single_example(example_proto, feature_description)
    image = tf.image.decode_jpeg(example['image'], channels=3)
    image = tf.image.resize(image, [192, 192])
    # scale image to [-1, 1]
    image = (image / 127.5) - 1
    return image, example['label']

  ds = ds.map(parse_example, num_parallel_calls=AUTOTUNE)
  ds = ds.shuffle(buffer_size=1024)
  ds = ds.batch(hparams.batch_size)
  ds = ds.prefetch(AUTOTUNE)
  return ds


def get_model(hparams):
  mobile_net = tf.keras.applications.MobileNetV2(
      input_shape=(192, 192, 3), include_top=False)

  mobile_net.trainable = False

  model = keras.Sequential([
      mobile_net,
      keras.layers.GlobalAveragePooling2D(),
      keras.layers.Dropout(rate=hparams.dropout),
      keras.layers.Dense(hparams.num_classes, activation='softmax')
  ])

  return model


@tf.function
def train_step(features, labels, model, optimizer, loss_fn):
  with tf.GradientTape() as tape:
    predictions = model(features, training=True)
    loss = loss_fn(labels, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  return loss, predictions


@tf.function
def eval_step(features, labels, model, loss_fn):
  predictions = model(features, training=False)
  loss = loss_fn(labels, predictions)
  return loss, predictions


def train_and_eval(hparams):
  model = get_model(hparams)
  optimizer = keras.optimizers.Adam(lr=hparams.learning_rate)
  loss_fn = keras.losses.SparseCategoricalCrossentropy()
  logger = Logger(hparams, optimizer)

  train_dataset = get_dataset(hparams, train=True)
  eval_dataset = get_dataset(hparams, train=False)

  checkpoint = Checkpoint(hparams, optimizer, model)

  checkpoint.restore()

  for epoch in range(hparams.epochs):

    start = time.time()

    for images, labels in train_dataset:
      loss, predictions = train_step(images, labels, model, optimizer, loss_fn)
      logger.log_progress(loss, labels, predictions, mode='train')

    elapse = time.time() - start

    logger.write_scalars(mode='train')

    for images, labels in eval_dataset:
      logger.write_images(images, mode='eval')
      loss, predictions = eval_step(images, labels, model, loss_fn)
      logger.log_progress(loss, labels, predictions, mode='eval')

    logger.write_scalars(mode='eval', elapse=elapse)

    logger.print_progress(epoch, elapse)

    if epoch % 5 == 0 or epoch == hparams.epochs - 1:
      checkpoint.save()

  tf.keras.models.save_model(model, filepath=hparams.save_model)
  print('model saved at %s' % hparams.save_model)


def main():
  hparams = get_hparams()
  train_and_eval(hparams)


if __name__ == "__main__":
  main()
