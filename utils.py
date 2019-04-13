import os
import tensorflow as tf
import tensorflow.keras as keras

import numpy as np


def get_hparams(epochs=20,
                batch_size=64,
                learning_rate=0.001,
                num_units=128,
                dropout=0.4,
                output_dir='runs/',
                data_dir='data'):
  hparams = HParams()
  hparams.epochs = epochs
  hparams.batch_size = batch_size
  hparams.learning_rate = learning_rate
  hparams.num_units = num_units
  hparams.dropout = dropout
  hparams.output_dir = output_dir
  hparams.save_model = os.path.join(output_dir, 'model.h5')
  hparams.data_dir = data_dir
  hparams.train_record = 'train.tfrecord'
  hparams.eval_record = 'eval.tfrecord'
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

    self.eval_loss = keras.metrics.Mean(name="eval_loss")
    self.eval_accuracy = keras.metrics.SparseCategoricalAccuracy(
        name="eval_accuracy")

    self.train_summary = tf.summary.create_file_writer(hparams.output_dir)
    self.eval_summary = tf.summary.create_file_writer(
        os.path.join(hparams.output_dir, 'eval'))

    self.optimizer = optimizer

  def _step(self):
    return self.optimizer.iterations

  def log_progress(self, loss, labels, predictions, mode):
    if mode == 'train':
      self.train_loss(loss)
      self.train_accuracy(labels, predictions)
    else:
      self.eval_loss(loss)
      self.eval_accuracy(labels, predictions)

  def write_images(self, images, mode):
    summary = self.train_summary if mode == 'train' else self.eval_summary

    # scale image to [0, 1]
    images = (images + 1) / 2
    with summary.as_default():
      tf.summary.image('features', images, step=self._step(), max_outputs=3)

  def write_scalars(self, mode, elapse=None):
    if mode == 'train':
      loss_metric = self.train_loss
      accuracy_metric = self.train_accuracy
      summary = self.train_summary
    else:
      loss_metric = self.eval_loss
      accuracy_metric = self.eval_accuracy
      summary = self.eval_summary

    with summary.as_default():
      tf.summary.scalar('loss', loss_metric.result(), step=self._step())
      tf.summary.scalar('accuracy', accuracy_metric.result(), step=self._step())
      if elapse is not None:
        tf.summary.scalar(
            'elapse', elapse, step=self._step(), description='sec per epoch')

  def print_progress(self, epoch, elapse):
    template = 'Epoch {}, Loss {:.4f}, Accuracy: {:.2f}, Eval Loss {:.4f}, ' \
               'Eval Accuracy {:.2f}, Time: {:.2f}s'
    print(
        template.format(
            epoch,
            self.train_loss.result(),
            self.train_accuracy.result() * 100,
            self.eval_loss.result(),
            self.eval_accuracy.result() * 100,
            elapse,
        ))


class Checkpoint(object):

  def __init__(self, hparams, optimizer, model):
    self.checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    self.output_dir = hparams.output_dir
    self.file_prefix = os.path.join(self.output_dir, 'ckpt')

  def save(self):
    path = self.checkpoint.save(file_prefix=self.file_prefix)
    print('saved checkpoint %s' % path)

  def restore(self):
    latest_checkpoint = tf.train.latest_checkpoint(self.output_dir)
    if latest_checkpoint is not None:
      self.checkpoint.restore(latest_checkpoint)
      print('restore checkpoint %s' % latest_checkpoint)
