from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from six.moves import range
import matplotlib.pyplot as plt
import numpy as np

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ''
tf.logging.set_verbosity(tf.logging.INFO)

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float('mean', 6.5, '')
tf.app.flags.DEFINE_float('std', 2.25, '')
tf.app.flags.DEFINE_integer('batch_size', 100, '')
tf.app.flags.DEFINE_integer('max_steps', 50000,
                            'the number of iterations to train')
tf.app.flags.DEFINE_integer('g_step', 1, '')
tf.app.flags.DEFINE_integer('d_step', 1, '')
tf.app.flags.DEFINE_float('base_lr', 2e-4, 'learning rate')
tf.app.flags.DEFINE_integer('lr_decay_step', 50000,
                            'decay learning rate every lr_decay_step')
tf.app.flags.DEFINE_float('lr_decay_rate', 0.1, 'learning decay rate')
tf.app.flags.DEFINE_integer('test_interval', 100, '')
tf.app.flags.DEFINE_integer('test_iter', 200, '')
tf.app.flags.DEFINE_string('checkpoint_dir', '~/tmp/tmp/tf-gan/',
                           'directory to save checkpoint')
tf.app.flags.DEFINE_integer('snapshot_interval', 5000,
                            'save checkpoint every snapshot_interval steps')
tf.app.flags.DEFINE_string('snapshot_prefix', 'gan-model',
                           'prefix added to checkpoint files')

# Discriminator
class DNet:
  def __init__(self, hidden_size=50, output_size=1):
    self._hidden_size = hidden_size
    self._output_size = output_size

  def infer_fn(self, data):
      # our fully connected layers
    fc1 = tf.layers.dense(
      inputs = data,
      units = self._hidden_size,
      activation = tf.nn.elu,
      use_bias = True,
      kernel_initializer = tf.random_uniform_initializer(
        minval=-0.141, maxval=0.141),
      bias_initializer =tf.random_uniform_initializer(
        minval=-0.141, maxval=0.141),
      name = 'fc1'
    )
    fc2 = tf.layers.dense(
      inputs = fc1,
      units = self._hidden_size,
      activation = tf.nn.elu,
      use_bias = True,
      kernel_initializer = tf.random_uniform_initializer(
        minval=-0.141, maxval=0.141),
      bias_initializer = tf.random_uniform_initializer(
        minval=-0.141, maxval=0.141),
      name = 'fc2'
    )
    out = tf.layers.dense(
      inputs = fc2,
      units = self._output_size,
      use_bias = True,
      kernel_initializer = tf.random_uniform_initializer(
        minval=-1, maxval=1),
      bias_initializer = tf.random_uniform_initializer(
        minval=-1, maxval=1),
      name = 'out'
    )
    # Out is a scalar between -1,1 to be interpreted as fake vs. real. This is about as milquetoast as a neural net can get.
    return out

  def loss_fn(self, logits, labels):
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
      labels = labels,
      logits = logits,
      name = 'dnet_cross_entropy_per_example'
    )
    cross_entropy_mean = tf.reduce_mean(
      input_tensor = cross_entropy,
      name = 'dnet_cross_entropy'
    )
    # The total loss is defined as the cross entropy loss.
    return cross_entropy_mean

# Generating samples now. Going to get the uniformly distributed data samples from input and somehow mimic the
# normally distributed samples from the real data.
class GNet:
  def __init__(self, hidden_size=50, output_size=1):
    self._hidden_size = hidden_size
    self._output_size = output_size

  def infer_fn(self, data):
    fc1 = tf.layers.dense(
      inputs = data,
      units = self._hidden_size,
      activation = tf.nn.elu,
      use_bias = True,
      kernel_initializer = tf.random_uniform_initializer(
        minval=-0.141, maxval=0.141),
      bias_initializer = tf.random_uniform_initializer(
        minval=-0.141, maxval=0.141),
      name = 'fc1'
    )
    fc2 = tf.layers.dense(
      inputs = fc1,
      units = self._hidden_size,
      activation = tf.sigmoid,
      use_bias = True,
      kernel_initializer = tf.random_uniform_initializer(
        minval=-0.141, maxval=0.141),
      bias_initializer = tf.random_uniform_initializer(
        minval=-0.141, maxval=0.141),
      name = 'fc2'
    )
    out = tf.layers.dense(
      inputs = fc2,
      units = self._output_size,
      use_bias = True,
      kernel_initializer = tf.random_uniform_initializer(
        minval=-1, maxval=1),
      bias_initializer = tf.random_uniform_initializer(
        minval=-1, maxval=1),
      name = 'out'
    )
    print(out)
    return out


g_net = GNet(
  hidden_size=50,
  output_size=1
)
d_net = DNet(
  hidden_size=50,
  output_size=1
)
g_net_infer = tf.make_template(
  'GNet_infer',
  g_net.infer_fn
)
d_net_infer = tf.make_template(
  'DNet_infer',
  d_net.infer_fn
)
d_net_loss = tf.make_template(
  'DNet_loss',
  d_net.loss_fn
)

def preprocess(data):
  mean = tf.reduce_mean(
    input_tensor = data,
    axis = 1,
    keepdims = True
  )
  diffs = tf.pow(
    data - mean,
    2
  )
  return tf.concat(
    values = [data, diffs],
    axis = 1
  )

with tf.Graph().as_default():
  global_step = tf.train.get_or_create_global_step()
  lr = tf.train.exponential_decay(
    learning_rate = FLAGS.base_lr,
    global_step = global_step,
    decay_steps = FLAGS.lr_decay_step,
    decay_rate = FLAGS.lr_decay_rate,
    staircase = True
  )
  opt = tf.train.AdamOptimizer(
    learning_rate = lr
  )

  with tf.name_scope('DNet'):
    # data
    d_real_data = tf.random_normal(
      shape = (1, FLAGS.batch_size),
      mean = FLAGS.mean,
      stddev = FLAGS.std
    )

    d_gen_input = tf.random_uniform(
      shape = (FLAGS.batch_size, 1),
      minval = 0,
      maxval = 1
    )

    d_fake_data = g_net_infer(
      data = d_gen_input
    )
    d_data = tf.concat(
      values = [
        d_real_data,
        tf.transpose(d_fake_data)
      ],
      axis = 0
    )
    d_pred = d_net_infer(
      data = preprocess(
        d_data
      )
    )
    # label
    d_label = tf.constant(
      value = [[1], [0]],
      dtype = tf.float32
    )
    # loss
    d_loss = d_net_loss(
      logits = d_pred,
      labels = d_label
    )
    d_step_op = opt.minimize(
      loss = d_loss,
      var_list = [_ for _ in tf.trainable_variables() if _.name.startswith('DNet')]
    )

  with tf.name_scope('GNet'):
    # data
    g_gen_input = tf.random_uniform(
      shape = (FLAGS.batch_size, 1),
      minval = 0,
      maxval = 1
    )
    g_fake_data = g_net_infer(
      data = g_gen_input
    )
    dg_pred = d_net_infer(
      data = preprocess(
        tf.transpose(g_fake_data)
      )
    )
    # label
    g_label = tf.ones(
      shape = (1, 1),
      dtype = tf.float32
    )
    # loss
    g_loss = d_net_loss(
      logits = dg_pred,
      labels = g_label
    )
    g_step_op = opt.minimize(
      loss = g_loss,
      var_list = [_ for _ in tf.trainable_variables() if _.name.startswith('GNet')]
    )


  inc_global_step_op = tf.assign_add(global_step, 1)
  scaffold = tf.train.Scaffold(
    saver=tf.train.Saver(
      allow_empty=False,
      max_to_keep=0
    )
  )
  summary_writer = tf.summary.FileWriterCache.get(FLAGS.checkpoint_dir)

  with tf.train.MonitoredTrainingSession(
    scaffold = scaffold,
    checkpoint_dir = FLAGS.checkpoint_dir,
    save_checkpoint_secs = None,
    hooks=[
      tf.train.CheckpointSaverHook(
        scaffold = scaffold,
        checkpoint_dir = FLAGS.checkpoint_dir,
        save_steps = FLAGS.snapshot_interval,
        checkpoint_basename = FLAGS.snapshot_prefix
      )
    ]
  ) as sess:
    global_step_value = sess.run(global_step)
    while global_step_value < FLAGS.max_steps:
      global_step_value = sess.run(global_step)
      # Train D
      for d_step in range(FLAGS.d_step):
        _, d_loss_value = sess.run([d_step_op, d_loss])
      # Train G
      for g_step in range(FLAGS.g_step):
        _, g_loss_value = sess.run([g_step_op, g_loss])
      # Test G
      if global_step_value % FLAGS.test_interval == 0:
        gen_data = []
        for idx in range(FLAGS.test_iter):
          gen_data.append(
            sess.run(g_fake_data)
          )
        gen_data = np.vstack(gen_data)
        mean = np.mean(gen_data)
        std = np.std(gen_data)
        mean_proto = tf.Summary(
          value = [
            tf.Summary.Value(
              tag = 'mean',
              simple_value = mean
            )
          ]
        )
        summary_writer.add_summary(
          summary = mean_proto,
          global_step = global_step_value
        )
        std_proto = tf.Summary(
          value = [
            tf.Summary.Value(
              tag = 'std',
              simple_value = std
            )
          ]
        )
        summary_writer.add_summary(
          summary = std_proto,
          global_step = global_step_value
        )
        print('Gen Data:',
              ' step->', global_step_value,
              ' lr->', sess.run(lr),
              ' d_loss->', d_loss_value,
              ' g_loss->', g_loss_value,
              ' mean->', mean,
              ' std->', std
              )
        sess.run(inc_global_step_op)
      else:
        sess.run(inc_global_step_op)
    gen_data = []
    for idx in range(FLAGS.test_iter):
      gen_data.append(
        sess.run(g_fake_data)
      )
    gen_data = np.vstack(gen_data)
    plt.hist(gen_data, bins=50)
    plt.show()