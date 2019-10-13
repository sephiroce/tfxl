#-*- coding:utf-8 -*-
# pylint: disable=import-error, too-many-arguments, too-many-locals
# pylint: disable=too-many-branches, too-many-statements, unused-argument
# pylint: disable=dangerous-default-value

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

import tensorflow as tf # pylint: disable=import-error

def positional_embedding(pos_seq, inv_freq, bsz=None):
  sinusoid_inp = tf.einsum('i,j->ij', pos_seq, inv_freq)
  pos_emb = tf.concat([tf.sin(sinusoid_inp), tf.cos(sinusoid_inp)], -1)
  if bsz is not None:
    return tf.tile(pos_emb[:, None, :], [1, bsz, 1])

  return pos_emb[:, None, :]

def positionwise_FF(inp, d_model, d_inner, dropout, kernel_initializer,
                    scope='ff', is_training=True):
  with tf.compat.v1.variable_scope(scope):
    output = tf.layers.dense(inp, d_inner, activation=tf.nn.relu,
                             kernel_initializer=kernel_initializer,
                             name='layer_1')
    output = tf.layers.dropout(output, dropout, training=is_training,
                               name='drop_1')
    output = tf.layers.dense(output, d_model,
                             kernel_initializer=kernel_initializer,
                             name='layer_2')
    output = tf.layers.dropout(output, dropout, training=is_training,
                               name='drop_2')
    output = tf.contrib.layers.layer_norm(output + inp, begin_norm_axis=-1)
  return output


def rel_shift(arg_x):
  x_size = tf.shape(arg_x)

  arg_x = tf.pad(arg_x, [[0, 0], [1, 0], [0, 0], [0, 0]])
  arg_x = tf.reshape(arg_x, [x_size[1] + 1, x_size[0], x_size[2], x_size[3]])
  arg_x = tf.slice(arg_x, [1, 0, 0, 0], [-1, -1, -1, -1])
  arg_x = tf.reshape(arg_x, x_size)

  return arg_x


def rel_multihead_attn(arg_w, arg_r, r_w_bias, r_r_bias, attn_mask, mems, d_model,
                       n_head, d_head, dropout, dropatt, is_training,
                       kernel_initializer, scope='rel_attn'):
  scale = 1 / (d_head ** 0.5)
  with tf.compat.v1.variable_scope(scope):
    qlen = tf.shape(arg_w)[0]
    rlen = tf.shape(arg_r)[0]
    bsz = tf.shape(arg_w)[1]

    cat = tf.concat([mems, arg_w],
                    0) if mems is not None and mems.shape.ndims > 1 else arg_w
    w_heads = tf.layers.dense(cat, 3 * n_head * d_head, use_bias=False,
                              kernel_initializer=kernel_initializer, name='qkv')
    r_head_k = tf.layers.dense(arg_r, n_head * d_head, use_bias=False,
                               kernel_initializer=kernel_initializer, name='r')

    w_head_q, w_head_k, w_head_v = tf.split(w_heads, 3, -1)
    w_head_q = w_head_q[-qlen:]

    klen = tf.shape(w_head_k)[0]

    w_head_q = tf.reshape(w_head_q, [qlen, bsz, n_head, d_head])
    w_head_k = tf.reshape(w_head_k, [klen, bsz, n_head, d_head])
    w_head_v = tf.reshape(w_head_v, [klen, bsz, n_head, d_head])

    r_head_k = tf.reshape(r_head_k, [rlen, n_head, d_head])

    rw_head_q = w_head_q + r_w_bias
    rr_head_q = w_head_q + r_r_bias

    AC = tf.einsum('ibnd,jbnd->ijbn', rw_head_q, w_head_k) #pylint: disable=invalid-name
    BD = tf.einsum('ibnd,jnd->ijbn', rr_head_q, r_head_k) #pylint: disable=invalid-name
    BD = rel_shift(BD) #pylint: disable=invalid-name

    attn_score = (AC + BD) * scale
    attn_mask_t = attn_mask[:, :, None, None]
    attn_score = attn_score * (1 - attn_mask_t) - 1e30 * attn_mask_t

    attn_prob = tf.nn.softmax(attn_score, 1)
    attn_prob = tf.layers.dropout(attn_prob, dropatt, training=is_training)

    attn_vec = tf.einsum('ijbn,jbnd->ibnd', attn_prob, w_head_v)
    size_t = tf.shape(attn_vec)
    attn_vec = tf.reshape(attn_vec, [size_t[0], size_t[1], n_head * d_head])

    attn_out = tf.layers.dense(attn_vec, d_model, use_bias=False,
                               kernel_initializer=kernel_initializer, name='o')
    attn_out = tf.layers.dropout(attn_out, dropout, training=is_training)

    output = tf.contrib.layers.layer_norm(attn_out + arg_w, begin_norm_axis=-1)
  return output


def embedding_lookup(lookup_table, arg_x):
  return tf.nn.embedding_lookup(lookup_table, arg_x)


def mask_adaptive_embedding_lookup(arg_x, n_token, d_embed, d_proj, cutoffs,
                                   initializer,
                                   proj_initializer,
                                   div_val=1,
                                   proj_same_dim=True,
                                   scope='adaptive_embed'):
  emb_scale = d_proj ** 0.5
  with tf.compat.v1.variable_scope(scope):
    if div_val == 1:
      lookup_table =\
        tf.compat.v1.get_variable('lookup_table', [n_token, d_embed],
                                  initializer=initializer)
      arg_y = embedding_lookup(lookup_table, arg_x)

      if d_proj != d_embed:
        proj_wgt =\
          tf.compat.v1.get_variable('proj_W', [d_embed, d_proj],
                                    initializer=proj_initializer)
        arg_y = tf.einsum('ibe,ed->ibd', arg_y, proj_wgt)
      else:
        proj_wgt = None
      ret_params = [lookup_table, proj_wgt]

    else:
      tables, projs = [], []
      cutoff_ends = [0] + cutoffs + [n_token]
      x_size = tf.shape(arg_x)
      arg_y = tf.zeros([x_size[0], x_size[1], d_proj])
      for i in range(len(cutoff_ends) - 1):
        with tf.compat.v1.variable_scope('cutoff_{}'.format(i)):
          l_idx, r_idx = cutoff_ends[i], cutoff_ends[i + 1]
          mask = (arg_x >= l_idx) & (arg_x < r_idx)
          cur_x = tf.boolean_mask(arg_x, mask) - l_idx
          cur_d_embed = d_embed // (div_val ** i)
          lookup_table = \
            tf.compat.v1.get_variable('lookup_table',
                                      [r_idx - l_idx, cur_d_embed],
                                      initializer=initializer)
          cur_y = embedding_lookup(lookup_table, cur_x)
          if d_proj == cur_d_embed and not proj_same_dim:
            proj_wgt = None
          else:
            proj_wgt =\
              tf.compat.v1.get_variable('proj_W', [cur_d_embed, d_proj],
                                        initializer=proj_initializer)
            cur_y = tf.einsum('id,de->ie', cur_y, proj_wgt)
          mask_idx = tf.to_int64(tf.where(mask))
          arg_y += tf.scatter_nd(mask_idx, cur_y, tf.to_int64(tf.shape(arg_y)))
          tables.append(lookup_table)
          projs.append(proj_wgt)
      ret_params = [tables, projs]

  arg_y *= emb_scale
  return arg_y, ret_params

def mask_adaptive_logsoftmax(hidden, target, n_token, d_embed, d_proj, cutoffs,
                             params, tie_projs,
                             initializer=None, proj_initializer=None,
                             div_val=1, scope='adaptive_softmax',
                             proj_same_dim=True,
                             return_mean=True):
  def _logit(arg_x, arg_wgt, arg_bias, proj):
    arg_y = arg_x
    if proj is not None:
      arg_y = tf.einsum('ibd,ed->ibe', arg_y, proj)
    return tf.einsum('ibd,nd->ibn', arg_y, arg_wgt) + arg_bias

  params_weight, params_projs = params[0], params[1]

  def _gather_logprob(logprob, target):
    lp_size = tf.shape(logprob)
    variable_range = tf.range(lp_size[0])
    idx = tf.stack([variable_range, target], 1)
    return tf.gather_nd(logprob, idx)

  with tf.compat.v1.variable_scope(scope):
    if not cutoffs:
      softmax_b =\
        tf.compat.v1.get_variable('bias', [n_token],
                                  initializer=tf.zeros_initializer())
      output = _logit(hidden, params_weight, softmax_b, params_projs)
      nll = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target,
                                                           logits=output)
    else:
      cutoff_ends = [0] + cutoffs + [n_token]
      nll = tf.zeros_like(target, dtype=tf.float32)
      for i in range(len(cutoff_ends) - 1):
        with tf.compat.v1.variable_scope('cutoff_{}'.format(i)):
          l_idx, r_idx = cutoff_ends[i], cutoff_ends[i + 1]
          mask = (target >= l_idx) & (target < r_idx)
          mask_idx = tf.where(mask)
          cur_target = tf.boolean_mask(target, mask) - l_idx
          cur_d_embed = d_embed // (div_val ** i)

          if div_val == 1:
            cur_weight = params_weight[l_idx: r_idx]
          else:
            cur_weight = params_weight[i]
          cur_b =\
            tf.compat.v1.get_variable('b', [r_idx - l_idx],
                                      initializer=tf.zeros_initializer())
          if tie_projs[i]:
            if div_val == 1:
              cur_proj = params_projs
            else:
              cur_proj = params_projs[i]
          else:
            if (div_val == 1 or not proj_same_dim) and d_proj == cur_d_embed:
              cur_proj = None
            else:
              cur_proj =\
                tf.compat.v1.get_variable('proj', [cur_d_embed, d_proj],
                                          initializer=proj_initializer)
          if i == 0:
            cluster_weight =\
              tf.compat.v1.get_variable('cluster_W', [len(cutoffs), d_embed],
                                        initializer=tf.zeros_initializer())
            cluster_b =\
              tf.compat.v1.get_variable('cluster_b', [len(cutoffs)],
                                        initializer=tf.zeros_initializer())
            cur_weight = tf.concat([cur_weight, cluster_weight], 0)
            cur_b = tf.concat([cur_b, cluster_b], 0)

            head_logit = _logit(hidden, cur_weight, cur_b, cur_proj)
            head_logprob = tf.nn.log_softmax(head_logit)
            cur_head_logprob = tf.boolean_mask(head_logprob, mask)
            cur_logprob = _gather_logprob(cur_head_logprob, cur_target)
          else:
            cur_head_logprob = tf.boolean_mask(head_logprob, mask)
            cur_hidden = tf.boolean_mask(hidden, mask)
            tail_logit = tf.squeeze(_logit(
                cur_hidden[None], cur_weight, cur_b, cur_proj), 0)
            tail_logprob = tf.nn.log_softmax(tail_logit)
            cur_logprob = (cur_head_logprob[:, cutoff_ends[1] + i - 1] +
                           _gather_logprob(tail_logprob, cur_target))
          nll += tf.scatter_nd(mask_idx, -cur_logprob,
                               tf.to_int64(tf.shape(nll)))
  if return_mean:
    nll = tf.reduce_mean(nll)
  return nll


def _create_mask(qlen, mlen, same_length=False):
  attn_mask = tf.ones([qlen, qlen])
  mask_u = tf.linalg.band_part(attn_mask, 0, -1)
  mask_dia = tf.linalg.band_part(attn_mask, 0, 0)
  attn_mask_pad = tf.zeros([qlen, mlen])
  ret = tf.concat([attn_mask_pad, mask_u - mask_dia], 1)
  if same_length:
    mask_l = tf.linalg.band_part(attn_mask, -1, 0)
    ret = tf.concat([ret[:, :qlen] + mask_l - mask_dia, ret[:, qlen:]], 1)
  return ret


def _cache_mem(curr_out, prev_mem, mem_len=None):
  if mem_len is None or prev_mem is None:
    new_mem = curr_out
  elif mem_len == 0:
    return prev_mem
  else:
    new_mem = tf.concat([prev_mem, curr_out], 0)[- mem_len:] #pylint: disable=invalid-unary-operand-type

  return tf.stop_gradient(new_mem)


def transformer(dec_inp, target, mems, n_token, n_layer, d_model, d_embed,
                n_head, d_head, d_inner, dropout, dropatt,
                initializer, is_training, proj_initializer=None,
                mem_len=None, cutoffs=[], div_val=1, tie_projs=[],
                same_length=False, clamp_len=-1, use_tpu=True,
                input_perms=None, target_perms=None, head_target=None,
                untie_r=False, proj_same_dim=True,
                scope='transformer'):
  """
  cutoffs: a list of python int. Cutoffs for adaptive softmax.
  tie_projs: a list of python bools. Whether to tie the projections.
  use_tpu: if True, use one_hot in embedding lookup and bin-based implementation
        of adaptive softmax.
  perms: a list of tensors. Each tensor should of size [len, bsz, bin_size].
        Only used in the adaptive setting.
  """
  new_mems = []
  with tf.compat.v1.variable_scope(scope):
    if untie_r:
      r_w_bias =\
        tf.compat.v1.get_variable('r_w_bias', [n_layer, n_head, d_head],
                                  initializer=initializer)
      r_r_bias =\
        tf.compat.v1.get_variable('r_r_bias', [n_layer, n_head, d_head],
                                  initializer=initializer)
    else:
      r_w_bias =\
        tf.compat.v1.get_variable('r_w_bias', [n_head, d_head],
                                  initializer=initializer)
      r_r_bias =\
        tf.compat.v1.get_variable('r_r_bias', [n_head, d_head],
                                  initializer=initializer)

    qlen = tf.shape(dec_inp)[0]
    mlen = tf.shape(mems[0])[0] if mems is not None else 0
    klen = mlen + qlen

    if proj_initializer is None:
      proj_initializer = initializer

    embeddings, shared_params =\
      mask_adaptive_embedding_lookup(arg_x=dec_inp,
                                     n_token=n_token,
                                     d_embed=d_embed,
                                     d_proj=d_model,
                                     cutoffs=cutoffs,
                                     initializer=initializer,
                                     proj_initializer=proj_initializer,
                                     div_val=div_val,
                                     proj_same_dim=proj_same_dim)

    attn_mask = _create_mask(qlen, mlen, same_length)

    pos_seq = tf.range(klen - 1, -1, -1.0)
    if clamp_len > 0:
      pos_seq = tf.minimum(pos_seq, clamp_len)
    inv_freq = 1 / (10000 ** (tf.range(0, d_model, 2.0) / d_model))
    pos_emb = positional_embedding(pos_seq, inv_freq)

    output = tf.layers.dropout(embeddings, dropout, training=is_training)
    pos_emb = tf.layers.dropout(pos_emb, dropout, training=is_training)

    if mems is None:
      mems = [None] * n_layer

    for i in range(n_layer):
      # cache new mems
      new_mems.append(_cache_mem(output, mems[i], mem_len))

      with tf.compat.v1.variable_scope('layer_{}'.format(i)):
        output = rel_multihead_attn(
            arg_w=output,
            arg_r=pos_emb,
            r_w_bias=r_w_bias if not untie_r else r_w_bias[i],
            r_r_bias=r_r_bias if not untie_r else r_r_bias[i],
            attn_mask=attn_mask,
            mems=mems[i],
            d_model=d_model,
            n_head=n_head,
            d_head=d_head,
            dropout=dropout,
            dropatt=dropatt,
            is_training=is_training,
            kernel_initializer=initializer)
        output = positionwise_FF(
            inp=output,
            d_model=d_model,
            d_inner=d_inner,
            dropout=dropout,
            kernel_initializer=initializer,
            is_training=is_training)

    output = tf.layers.dropout(output, dropout, training=is_training)

    loss = mask_adaptive_logsoftmax(hidden=output,
                         target=target,
                         n_token=n_token,
                         d_embed=d_embed,
                         d_proj=d_model,
                         cutoffs=cutoffs,
                         params=shared_params,
                         tie_projs=tie_projs,
                         initializer=initializer,
                         proj_initializer=proj_initializer,
                         div_val=div_val,
                         proj_same_dim=proj_same_dim)
    return loss, new_mems
