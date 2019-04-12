import os
import tensorflow as tf
import tensorflow.keras as keras


def get_hparams(epochs=40,
                batch_size=64,
                learning_rate=0.001,
                num_units=128,
                dropout=0.4,
                output_dir='runs/'):
  hparams = HParams()
  hparams.epochs = epochs
  hparams.batch_size = batch_size
  hparams.learning_rate = learning_rate
  hparams.num_units = num_units
  hparams.dropout = dropout
  hparams.output_dir = output_dir
  hparams.save_model = os.path.join(output_dir, 'model.h5')
  hparams.data_dir = 'data'
  hparams.train_record = 'train.tfrecord'
  hparams.test_record = 'test.tfrecord'
  hparams.num_classes = 5
  return hparams


class HParams(object):
  """Empty object hyper-parameters """

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)


class Logger(object):

  def __init__(self, hparams, optimizer):
    self.train_loss = keras.metrics.Mean(name="train_loss")
    self.train_accuracy = keras.metrics.SparseCategoricalAccuracy(
        name="train_accuracy")

    self.test_loss = keras.metrics.Mean(name="test_loss")
    self.test_accuracy = keras.metrics.SparseCategoricalAccuracy(
        name="test_accuracy")

    self.train_summary = tf.summary.create_file_writer(hparams.output_dir)
    self.test_summary = tf.summary.create_file_writer(
        os.path.join(hparams.output_dir, 'eval'))

    self.optimizer = optimizer

  def _step(self):
    return self.optimizer.iterations

  def log_progress(self, loss, labels, predictions, mode):
    if mode == 'train':
      self.train_loss(loss)
      self.train_accuracy(labels, predictions)
    else:
      self.test_loss(loss)
      self.test_accuracy(labels, predictions)

  def write_images(self, images, mode):
    summary = self.train_summary if mode == 'train' else self.test_summary
    with summary.as_default():
      tf.summary.image('features', images, step=self._step(), max_outputs=3)

  def write_scalars(self, mode, elapse=None):
    if mode == 'train':
      loss_metric = self.train_loss
      accuracy_metric = self.train_accuracy
      summary = self.train_summary
    else:
      loss_metric = self.test_loss
      accuracy_metric = self.test_accuracy
      summary = self.test_summary

    with summary.as_default():
      tf.summary.scalar('loss', loss_metric.result(), step=self._step())
      tf.summary.scalar('accuracy', accuracy_metric.result(), step=self._step())
      if elapse is not None:
        tf.summary.scalar(
            'elapse', elapse, step=self._step(), description='sec per epoch')

  def print_progress(self, epoch, elapse):
    template = 'Epoch {}, Loss {:.4f}, Accuracy: {:.2f}, Test Loss {:.4f}, ' \
               'Test Accuracy {:.2f}, Time: {:.2f}s'
    print(
        template.format(
            epoch,
            self.train_loss.result(),
            self.train_accuracy.result() * 100,
            self.test_loss.result(),
            self.test_accuracy.result() * 100,
            elapse,
        ))


def preprocess_image(image_path):
  image = tf.image.decode_jpeg(image_path, channels=3)
  image = tf.image.resize(image, [192, 192])
  # normalize image to [-1, -1]
  return (image / 127.5) - 1
