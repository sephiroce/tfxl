#-*- coding:utf-8 -*-
# pylint: disable=import-error, too-many-locals, too-many-arguments
# pylint: disable=too-many-statements

# Copyright 2019 The Transformer-xl Authors.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import copy

import numpy as np
from absl import flags

import tensorflow as tf

import tfxl.model as model
import tfxl.data_utils as data_utils

# GPU config
flags.DEFINE_integer("num_core_per_host", default=8,
                     help="Number of cores per host")

# Experiment (data/checkpoint/directory) config
flags.DEFINE_string("data_dir", default="", help="Path to tf-records "
                                                 "directory.")
flags.DEFINE_string("record_info_dir", default="", help="Path to local "
                                                        "directory containing "
                                                        "filenames.txt.")
flags.DEFINE_string("corpus_info_path", default="",
                    help="Path to corpus-info.json file.")
flags.DEFINE_string("model_dir", default=None, help="Estimator model_dir.")
flags.DEFINE_bool("do_train", default=True, help="Whether to run training.")
flags.DEFINE_bool("do_eval", default=False, help="Whether to run eval on the "
                                                 "dev set.")
flags.DEFINE_string("eval_ckpt_path", None,
                    help="Checkpoint path for do_test evaluation. If set, "
                         "model_dir will be ignored. If unset, will use the "
                         "latest ckpt in model_dir.")
flags.DEFINE_string("warm_start_path", None,
                    help="Checkpoint path for warm start. If set, will clear "
                         "Adam states. Note that the new model_dir should be "
                         "different from warm_start_path.")

# Optimization config
flags.DEFINE_float("learning_rate", default=2.5e-4, help="Maximum learning "
                                                         "rate.")
flags.DEFINE_float("clip", default=0.25, help="Gradient clipping value.")

# for cosine decay
flags.DEFINE_float("min_lr_ratio", default=0.004, help="Minimum ratio "
                                                       "learning rate.")
flags.DEFINE_integer("warmup_steps", default=0, help="Number of steps for "
                                                     "linear lr warmup.")

# Training config
flags.DEFINE_integer("train_batch_size", default=60, help="Size of train "
                                                          "batch.")
flags.DEFINE_integer("eval_batch_size", default=60, help="Size of valid batch.")
flags.DEFINE_integer("train_steps", default=100000, help="Total number of "
                                                         "training steps.")
flags.DEFINE_integer("iterations", default=500, help="Number of iterations "
                                                     "per repeat loop.")
flags.DEFINE_integer("save_steps", default=10000, help="number of steps for "
                                                       "model checkpointing.")

# Evaluation config
flags.DEFINE_bool("do_test", default=False, help="Run on the test set.")
flags.DEFINE_bool("do_decode", default=False, help="Get next distribution")
flags.DEFINE_integer("max_eval_batch", default=-1,
                     help="Set -1 to turn off. Only used in test mode.")
flags.DEFINE_bool("do_eval_only", default=False, help="Run evaluation only.")
flags.DEFINE_integer("start_eval_steps", default=10000,
                     help="Which checkpoint to start with in `do_eval_only`"
                          "mode.")
flags.DEFINE_string("eval_split", "valid", help="Which data split to evaluate.")

# Model config
flags.DEFINE_integer("tgt_len", default=70, help="Number of steps to predict")
flags.DEFINE_integer("mem_len", default=70, help="Number of steps to cache")
flags.DEFINE_bool("same_length", default=False, help="Same length attention")
flags.DEFINE_integer("clamp_len", default=-1, help="Clamp length")

flags.DEFINE_integer("n_layer", default=6, help="Number of layers.")
flags.DEFINE_integer("d_model", default=500, help="Dimension of the model.")
flags.DEFINE_integer("d_embed", default=500, help="Dimension of the "
                                                  "embeddings.")
flags.DEFINE_integer("n_head", default=10, help="Number of attention heads.")
flags.DEFINE_integer("d_head", default=50, help="Dimension of each attention "
                                                "head.")
flags.DEFINE_integer("d_inner", default=1000,
                     help="Dimension of inner hidden size in position-wise "
                          "feed-forward.")
flags.DEFINE_float("dropout", default=0.1, help="Dropout rate.")
flags.DEFINE_float("dropatt", default=0.1, help="Attention dropout rate.")
flags.DEFINE_bool("untie_r", default=False, help="untie r_w_bias and r_r_bias")

# Adaptive Softmax / Embedding
flags.DEFINE_bool("tie_weight", default=True, help="Tie embedding and softmax"
                                                   "weight.")
flags.DEFINE_integer("div_val", default=1, help="Divide the embedding size by"
                                                "this val for each bin")
flags.DEFINE_bool("proj_share_all_but_first", default=False,
                  help="True to share all but first projs, False not to share.")
flags.DEFINE_bool("proj_same_dim", default=True,
                  help="Project the bin with the same dimension.")

# Parameter initialization
flags.DEFINE_enum("init", default="normal", enum_values=["normal", "uniform"],
                  help="Initialization method.")
flags.DEFINE_float("init_std", default=0.02, help="Initialization std when "
                                                  "init is normal.")
flags.DEFINE_float("proj_init_std", default=0.01,
                   help="Initialization std for embedding projection.")
flags.DEFINE_float("init_range", default=0.1,
                   help="Initialization std when init is uniform.")

flags.DEFINE_string("vocab", None, help="vocab filename")

FLAGS = flags.FLAGS


def assign_to_gpu(gpu=0, ps_dev="/device:CPU:0"):
  def _assign(operation):
    node_def = \
      operation if isinstance(operation, tf.compat.v1.NodeDef) else \
        operation.node_def
    if node_def.op == "Variable":
      return ps_dev
    return "/gpu:%d" % gpu
  return _assign


def average_grads_and_vars(tower_grads_and_vars):
  def average_dense(grad_and_vars_avg_dense):
    if len(grad_and_vars_avg_dense) == 1:
      return grad_and_vars_avg_dense[0][0]

    grad_avg_dense = grad_and_vars_avg_dense[0][0]
    for curr_grad, _ in grad_and_vars_avg_dense[1:]:
      grad_avg_dense += curr_grad
    return grad_avg_dense / len(grad_and_vars_avg_dense)

  def average_sparse(grad_and_vars_avg_sparse):
    if len(grad_and_vars_avg_sparse) == 1:
      return grad_and_vars_avg_sparse[0][0]

    indices = []
    values = []
    for grad_and_val_idx, _ in grad_and_vars_avg_sparse:
      indices += [grad_and_val_idx.indices]
      values += [grad_and_val_idx.values]
    indices = tf.concat(indices, 0)
    values = tf.concat(values, 0) / len(grad_and_vars_avg_sparse)
    return tf.IndexedSlices(values, indices,
                            grad_and_vars_avg_sparse[0][0].dense_shape)

  avg_grads_and_vars = []
  for grad_and_vars in zip(*tower_grads_and_vars):
    if grad_and_vars[0][0] is None:
      grad = None
    elif isinstance(grad_and_vars[0][0], tf.IndexedSlices):
      grad = average_sparse(grad_and_vars)
    else:
      grad = average_dense(grad_and_vars)
    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    variances = grad_and_vars[0][1]
    grad_and_var = (grad, variances)
    avg_grads_and_vars.append(grad_and_var)
  return avg_grads_and_vars


def single_core_graph(inp, tgt, mems, is_training, n_token):
  # batch major to time major?
  inp = tf.transpose(inp, [1, 0])
  tgt = tf.transpose(tgt, [1, 0])

  initializer = None
  proj_initializer = None

  if FLAGS.init == "uniform":
    initializer =\
      tf.initializers.random_uniform(minval=-FLAGS.init_range,
                                     maxval=FLAGS.init_range,
                                     seed=None)
  elif FLAGS.init == "normal":
    initializer =\
      tf.initializers.random_normal(stddev=FLAGS.init_std, seed=None)
    proj_initializer =\
      tf.initializers.random_normal(stddev=FLAGS.proj_init_std, seed=None)

  loss, new_mems, output, att_prob =\
    model.transformer(dec_inp=inp, # [1:-1], why not zero?
                      target=tgt,  # [2:]
                      mems=mems, # 3.2 Segment-Level Recurrent with State
                      # Reuse ?
                      n_token=n_token,
                      n_layer=FLAGS.n_layer,
                      d_model=FLAGS.d_model, # dimension of hidden
                      d_embed=FLAGS.d_embed, # dimension of embedded
                      n_head=FLAGS.n_head, # number of heads in multi-head att.
                      d_head=FLAGS.d_head, # dimension of head
                      d_inner=FLAGS.d_inner, # dim of inner ??
                      dropout=FLAGS.dropout, # dropout rate
                      dropatt=FLAGS.dropatt, # dropout attention ??
                      initializer=initializer,
                      proj_initializer=proj_initializer,
                      is_training=is_training,
                      mem_len=FLAGS.mem_len, # how many segments
                      same_length=FLAGS.same_length, # ??
                      clamp_len=FLAGS.clamp_len, # clamp length..?
                      untie_r=FLAGS.untie_r, # untie ??
                     )

  # number of parameters
  num_params =\
    sum([np.prod(v.shape) for v in tf.compat.v1.trainable_variables()])
  tf.compat.v1.logging.info('#params: {}'.format(num_params))

  if is_training:
    all_vars = tf.compat.v1.trainable_variables()
    grads = tf.gradients(loss, all_vars)
    grads_and_vars = list(zip(grads, all_vars))
    return loss, new_mems, grads_and_vars

  return loss, new_mems, output, att_prob


def train(n_token, ps_device):
  ##### Get input function and model function
  train_input_fn, train_record_info = data_utils.get_input_fn(
      record_info_dir=FLAGS.record_info_dir,
      split="train",
      per_host_bsz=FLAGS.train_batch_size,
      tgt_len=FLAGS.tgt_len,
      num_core_per_host=FLAGS.num_core_per_host)

  tf.compat.v1.logging.info("num of batches {}".format(train_record_info["num_batch"]))

  ##### Create computational graph
  train_set = train_input_fn({
      "batch_size": FLAGS.train_batch_size,
      "data_dir": FLAGS.data_dir})

  input_feed, label_feed = train_set.make_one_shot_iterator().get_next()

  inputs = tf.split(input_feed, FLAGS.num_core_per_host, 0)
  labels = tf.split(label_feed, FLAGS.num_core_per_host, 0)

  per_core_bsz = FLAGS.train_batch_size // FLAGS.num_core_per_host

  # initializing variable for each gpu
  tower_mems, tower_losses, tower_new_mems, tower_grads_and_vars = [], [], [], []

  # building up towers
  for i in range(FLAGS.num_core_per_host):
    reuse = True if i > 0 else None
    with tf.device(assign_to_gpu(i, ps_device)), \
        tf.compat.v1.variable_scope(tf.compat.v1.get_variable_scope(), reuse=reuse):

      # previous segment states, mem_len is the number of segments per core
      mems_i = [tf.compat.v1.placeholder(tf.float32,
                                         [FLAGS.mem_len, per_core_bsz,
                                          FLAGS.d_model])
                for _ in range(FLAGS.n_layer)]

      loss_i, new_mems_i, grads_and_vars_i =\
        single_core_graph(inp=inputs[i],
                          tgt=labels[i],
                          mems=mems_i,
                          is_training=True,
                          n_token=n_token)

      tower_mems.append(mems_i)
      tower_losses.append(loss_i)
      tower_new_mems.append(new_mems_i)
      tower_grads_and_vars.append(grads_and_vars_i)

  ## average losses and gradients across towers
  if len(tower_losses) > 1:
    loss = tf.add_n(tower_losses) / len(tower_losses)
    grads_and_vars = average_grads_and_vars(tower_grads_and_vars)
  else:
    loss = tower_losses[0]
    grads_and_vars = tower_grads_and_vars[0]
  grads, all_vars = zip(*grads_and_vars)

  ## clip gradient
  clipped, gnorm = tf.clip_by_global_norm(grads, FLAGS.clip)
  grads_and_vars = list(zip(clipped, all_vars))

  ## configure the optimizer
  global_step = tf.compat.v1.train.get_or_create_global_step()

  # warm-up stage: increase the learning rate linearly
  if FLAGS.warmup_steps > 0:
    warmup_lr = tf.to_float(global_step) / tf.to_float(FLAGS.warmup_steps) \
                * FLAGS.learning_rate
  else:
    warmup_lr = 0.0

  # decay stage: decay the learning rate using the cosine schedule
  decay_lr = tf.compat.v1.train.cosine_decay(
      FLAGS.learning_rate,
      global_step=global_step-FLAGS.warmup_steps,
      decay_steps=FLAGS.train_steps-FLAGS.warmup_steps,
      alpha=FLAGS.min_lr_ratio)

  # choose warmup or decay
  learning_rate = tf.where(global_step < FLAGS.warmup_steps,
                           warmup_lr, decay_lr)

  # get the train op
  optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
  train_op = optimizer.apply_gradients(grads_and_vars, global_step)

  ##### Training loop
  tower_mems_np = [
      [np.zeros([FLAGS.mem_len, per_core_bsz, FLAGS.d_model],
                dtype=np.float32) # pylint: disable=no-member
       for _ in range(FLAGS.n_layer)] # pylint: disable=unused-variable
      for _ in range(FLAGS.num_core_per_host) # pylint: disable=unused-variable
  ]

  saver = tf.compat.v1.train.Saver()

  with tf.compat.v1.Session(config=\
                            tf.compat.v1.ConfigProto(allow_soft_placement=True)
                           )as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    if FLAGS.warm_start_path is not None:
      tf.compat.v1.logging.info("warm start from {}".format(FLAGS.warm_start_path))
      saver.restore(sess, FLAGS.warm_start_path)

    fetches = [loss, tower_new_mems, global_step, gnorm, learning_rate,
               train_op]

    total_loss, prev_step = 0., -1

    while True:
      feed_dict = {}
      for i in range(FLAGS.num_core_per_host):
        for tower_mem_idx, m_np in zip(tower_mems[i], tower_mems_np[i]):
          feed_dict[tower_mem_idx] = m_np

      fetched = sess.run(fetches, feed_dict=feed_dict)

      loss_np, tower_mems_np, curr_step = fetched[:3]

      total_loss += loss_np

      if curr_step > 0 and curr_step % FLAGS.iterations == 0:
        curr_loss = total_loss / (curr_step - prev_step)
        tf.compat.v1.logging.info("[{}] | gnorm {:.2f} lr {:8.6f} | loss {:.2f}"
                                  " | pplx {:>7.2f}, bpc {:>7.4f}"
                                  .format(curr_step, fetched[-3], fetched[-2],
                                          curr_loss, math.exp(curr_loss),
                                          curr_loss / math.log(2)))
        total_loss, prev_step = 0., curr_step

      if curr_step > 0 and curr_step % FLAGS.save_steps == 0:
        save_path = os.path.join(FLAGS.model_dir, "model.ckpt")
        saver.save(sess, save_path)
        tf.compat.v1.logging.info("Model saved in path: {}".format(save_path))

      if curr_step == FLAGS.train_steps:
        break


def evaluate(n_token, ps_device):
  ##### Get input function and model function
  eval_input_fn, eval_record_info = data_utils.get_input_fn(
      record_info_dir=FLAGS.record_info_dir,
      split=FLAGS.eval_split,
      per_host_bsz=FLAGS.eval_batch_size,
      tgt_len=FLAGS.tgt_len,
      num_core_per_host=FLAGS.num_core_per_host)

  num_batch = eval_record_info["num_batch"]
  if FLAGS.max_eval_batch > 0:
    num_batch = FLAGS.max_eval_batch
  tf.compat.v1.logging.info("num of batches {}".format(num_batch))

  ##### Create computational graph
  eval_set = eval_input_fn({
      "batch_size": FLAGS.eval_batch_size,
      "data_dir": FLAGS.data_dir})

  input_feed, label_feed = eval_set.make_one_shot_iterator().get_next()

  inputs = tf.split(input_feed, FLAGS.num_core_per_host, 0)
  labels = tf.split(label_feed, FLAGS.num_core_per_host, 0)

  per_core_bsz = FLAGS.eval_batch_size // FLAGS.num_core_per_host
  tower_mems, tower_losses, tower_new_mems = [], [], []

  for i in range(FLAGS.num_core_per_host):
    with tf.device(assign_to_gpu(i, ps_device)), \
        tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):

      mems_i = [tf.placeholder(tf.float32,
                               [FLAGS.mem_len, per_core_bsz, FLAGS.d_model])
                for _ in range(FLAGS.n_layer)]

      loss_i, new_mems_i, _, _ = \
        single_core_graph(inp=inputs[i],
                          tgt=labels[i],
                          mems=mems_i,
                          is_training=False,
                          n_token=n_token)

      tower_mems.append(mems_i)
      tower_losses.append(loss_i)
      tower_new_mems.append(new_mems_i)

  ## sum losses across towers
  if len(tower_losses) > 1:
    loss = tf.add_n(tower_losses) / len(tower_losses)
  else:
    loss = tower_losses[0]

  ##### Evaluation loop
  tower_mems_np = [
      [np.zeros([FLAGS.mem_len, per_core_bsz, FLAGS.d_model], dtype=np.float32) # pylint: disable=no-member
       for _ in range(FLAGS.n_layer)] # pylint: disable=unused-variable
      for _ in range(FLAGS.num_core_per_host) # pylint: disable=unused-variable
  ]

  saver = tf.train.Saver()

  with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    sess.run(tf.global_variables_initializer())

    if FLAGS.eval_ckpt_path is None:
      eval_ckpt_path = tf.train.latest_checkpoint(FLAGS.model_dir)
    else:
      eval_ckpt_path = FLAGS.eval_ckpt_path
    tf.compat.v1.logging.info("Evaluate {}".format(eval_ckpt_path))
    saver.restore(sess, eval_ckpt_path)

    fetches = [loss, tower_new_mems, tf.size(label_feed)]

    format_str = "  >> processing batch {{:{0}d}}/{{:{0}d}} ..".format(
        len(str(num_batch)))

    total_loss, total_cnt = 0, 0
    for step in range(num_batch):
      if step % (num_batch // 10) == 0:
        tf.compat.v1.logging.info(format_str.format(step, num_batch))

      feed_dict = {}
      for i in range(FLAGS.num_core_per_host):
        for tower_mem_idx, m_np in zip(tower_mems[i], tower_mems_np[i]):
          feed_dict[tower_mem_idx] = m_np

      fetched = sess.run(fetches, feed_dict=feed_dict)

      loss_np, tower_mems_np, cnt_np = fetched[:3]

      total_loss += loss_np * cnt_np
      total_cnt += cnt_np

    avg_loss = total_loss / total_cnt
    tf.compat.v1.logging.info("| loss {:.2f} | pplx {:>7.2f}, bpc {:>7.4f}".format(
        avg_loss, math.exp(avg_loss), avg_loss / math.log(2)))


class Tfxl(object): # pylint: disable=too-many-instance-attributes, too-few-public-methods
  def __init__(self, n_token, eval_ckpt_path, config, bos_idx=-1):
    # configurations
    self.n_token = n_token
    self.mem_len = config["mem_len"]
    self.d_model = config["d_model"]
    self.n_layer = config["n_layer"]
    self.d_embed = config["d_embed"]
    self.n_head = config["n_head"]
    self.d_head = config["d_head"]
    self.d_inner = config["d_inner"]
    self.dropout = config["dropout"]
    self.dropatt = config["dropatt"]
    self.same_length = config["same_length"]
    self.clamp_len = config["clamp_len"]
    self.untie_r = config["untie_r"]
    self.bos_idx = bos_idx

    tf.compat.v1.logging.info("self.n_token {}".format(self.n_token))
    tf.compat.v1.logging.info("self.n_layer {}".format(self.n_layer))
    tf.compat.v1.logging.info("self.d_model {}".format(self.d_model))
    tf.compat.v1.logging.info("self.d_embed {}".format(self.d_embed))
    tf.compat.v1.logging.info("self.n_head {}".format(self.n_head))
    tf.compat.v1.logging.info("self.d_head {}".format(self.d_head))
    tf.compat.v1.logging.info("self.d_inner {}".format(self.d_inner))
    tf.compat.v1.logging.info("self.dropout {}".format(self.dropout))
    tf.compat.v1.logging.info("self.dropatt {}".format(self.dropatt))
    tf.compat.v1.logging.info("self.same_length {}".format(self.same_length))
    tf.compat.v1.logging.info("self.clamp_len {}".format(self.clamp_len))
    tf.compat.v1.logging.info("self.untie_r {}".format(self.untie_r))
    tf.compat.v1.logging.info("self.bos_idx {}".format(self.bos_idx))

    # Building a new graph
    self._graph = tf.Graph()
    with self._graph.as_default():
      self.inputs = tf.compat.v1.placeholder(dtype=tf.int32, shape=(1, None))
      self.labels = tf.compat.v1.placeholder(dtype=tf.int32, shape=(1, None))
      self.tower_new_mems, self.tower_output = [], []
      self.tower_att_prob, self.tower_att_vec = [], []

      with tf.device('/gpu:0'), \
           tf.compat.v1.variable_scope(tf.compat.v1.get_variable_scope(),
                                       reuse=tf.compat.v1.AUTO_REUSE):
        self.mems = [tf.compat.v1.placeholder(tf.float32, [self.mem_len, 1,
                                                           self.d_model])
                     for _ in range(self.n_layer)]

        # batch major to time major?
        inp = tf.transpose(self.inputs, [1, 0])
        tgt = tf.transpose(self.labels, [1, 0])

        _, new_mems_i, output_i, att_prob_i = \
          model.transformer(dec_inp=inp,
                            target=tgt,
                            mems=self.mems,
                            n_token=self.n_token,
                            n_layer=self.n_layer,
                            d_model=self.d_model,  # dimension of hidden
                            d_embed=self.d_embed,  # dimension of embedded
                            n_head=self.n_head,
                            d_head=self.d_head,  # dimension of head
                            d_inner=self.d_inner,  # dim of inner ??
                            dropout=self.dropout,  # dropout rate
                            dropatt=self.dropatt,  # dropout attention ??
                            initializer=None,
                            proj_initializer=None,
                            is_training=False,
                            mem_len=self.mem_len,  # how many segments
                            same_length=self.same_length,  # ??
                            clamp_len=self.clamp_len,  # clamp length..?
                            untie_r=self.untie_r,  # untie ??
                           )

        # number of parameters
        num_params =\
          sum([np.prod(v.shape) for v in tf.compat.v1.trainable_variables()])
        tf.compat.v1.logging.info('#params: {}'.format(num_params))

        self.tower_new_mems.append(new_mems_i)
        self.tower_output.append(output_i)
        self.tower_att_prob.append(att_prob_i)

      # Starting Session
      self.sess =\
        tf.compat.v1.Session(config=\
          tf.compat.v1.ConfigProto(allow_soft_placement=True))
      self.sess.run(tf.compat.v1.global_variables_initializer())

      # Loading a Checkpoints
      saver = tf.compat.v1.train.Saver()
      tf.compat.v1.logging.info("Evaluate {}".format(eval_ckpt_path))
      saver.restore(self.sess, eval_ckpt_path)

  def get_dist(self, sent, softmax=False, temperature=1.0):
    if self.bos_idx >= 0:
      assert isinstance(sent, (np.ndarray, list))
      if isinstance(sent, np.ndarray):
        sent = copy.deepcopy(sent.tolist())
      elif isinstance(sent, list):
        sent = copy.deepcopy(sent)
      sent.insert(0, self.bos_idx)
    else:
      sent = sent

    fetches = [self.tower_new_mems, self.tower_output, self.tower_att_prob]
    sent = [sent]
    feed_dict = {self.inputs: [sent[0]], self.labels: [sent[0]]}

    tower_mems_np = [np.zeros([self.mem_len, 1, self.d_model],
                              dtype=np.float32) # pylint: disable=no-member
                     for _ in range(self.n_layer)]  # pylint: disable=unused-variable
    for tower_mem_idx, m_np in zip(self.mems, tower_mems_np):
      feed_dict[tower_mem_idx] = m_np

    tower_mems_np, output, att_prob = self.sess.run(fetches, feed_dict=feed_dict)

    # prob to heatmap
    q_len = np.shape(np.squeeze(att_prob))[0]
    att_maps = np.squeeze(att_prob)[:, -q_len:, :]
    att_maps = np.swapaxes(att_maps, 0, 2)

    logit = output[0][-1][0]
    logit = logit[:-1] # removing last dim: bos
    if softmax:
      logit = np.exp(logit/temperature) / np.sum(np.exp(logit/temperature))
    return logit, att_maps

def main(unused_argv):
  del unused_argv
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

  if FLAGS.do_decode:
    from tfxl.vocabulary import Vocab
    vocab = Vocab(min_freq=0, max_size=None, lower_case=True, delimiter=None,
                  vocab_file=FLAGS.vocab)
    vocab.build_vocab()

    # Creating Transformer-XL decoding model
    if FLAGS.eval_ckpt_path is None:
      eval_ckpt_path = tf.train.latest_checkpoint(FLAGS.model_dir)
    else:
      eval_ckpt_path = FLAGS.eval_ckpt_path

    config = {"mem_len": FLAGS.mem_len,
              "d_model": FLAGS.d_model,
              "n_layer": FLAGS.n_layer,
              "d_embed": FLAGS.d_embed,
              "n_head": FLAGS.n_head,
              "d_head": FLAGS.d_head,
              "d_inner": FLAGS.d_inner,
              "dropout": FLAGS.dropout,
              "dropatt": FLAGS.dropatt,
              "same_length": FLAGS.same_length,
              "clamp_len": FLAGS.clamp_len,
              "untie_r": FLAGS.untie_r}

    bos_idx = vocab.get_idx("<s>") if vocab.get_idx("<s>") is not \
                                      vocab.get_idx("<unk>") else -1

    tfxl = Tfxl(n_token=len(vocab), eval_ckpt_path=eval_ckpt_path,
                config=config, bos_idx=bos_idx)

    sent = vocab.encode_sentence("a form of asbestos once used to make kent "
                                 "cigarette filters has caused a",
                                 add_eos=False,
                                 add_beos=False)

    output, att_maps = tfxl.get_dist(sent[0], softmax=True, temperature=1.0)
    print(vocab.get_sym(np.argmax(output)))

    import matplotlib.pyplot as plt
    for img_i, vec in enumerate(att_maps):
      plt.imshow(np.transpose(vec), cmap='hot', interpolation='nearest')
      img_path = "%s_%0d.png"%(eval_ckpt_path, img_i + 1)
      plt.savefig(img_path)
      print("heat map for header %d was saved to %s"%(img_i + 1, img_path))
  else:
    with open(FLAGS.corpus_info_path, "r") as fp:
      import json
      corpus_info = json.load(fp)
      n_token = corpus_info["vocab_size"]
      tf.compat.v1.logging.info("n_token {}".format(n_token))

      if FLAGS.do_train:
        train(n_token, "/gpu:0")
      if FLAGS.do_eval:
        evaluate(n_token, "/gpu:0")


if __name__ == "__main__":
  tf.compat.v1.app.run()
