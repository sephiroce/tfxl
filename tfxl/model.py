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

def positional_embedding(pos_seq, inv_freq):
  # outer product between post_seq and inv_freq
  sinusoid_inp = tf.einsum('i,j->ij', pos_seq, inv_freq)
  pos_emb = tf.concat([tf.sin(sinusoid_inp), tf.cos(sinusoid_inp)], -1)

  return pos_emb[:, None, :]


def positionwise_FF(inp, d_model, d_inner, dropout, kernel_initializer,
                    scope='ff', is_training=True):
  with tf.compat.v1.variable_scope(scope):
    # Feed Forward
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

    # Add & Norm
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
    # Preparing head weights for q, k, v
    qlen = tf.shape(arg_w)[0]
    bsz = tf.shape(arg_w)[1]

    cat = tf.concat([mems, arg_w], 0) if mems is not None and \
                                         mems.shape.ndims > 1 else arg_w
    w_heads = tf.layers.dense(cat, 3 * n_head * d_head, use_bias=False,
                              kernel_initializer=kernel_initializer, name='qkv')
    w_head_q, w_head_k, w_head_v = tf.split(w_heads, 3, -1)
    w_head_q = w_head_q[-qlen:]

    klen = tf.shape(w_head_k)[0]

    w_head_q = tf.reshape(w_head_q, [qlen, bsz, n_head, d_head])
    w_head_k = tf.reshape(w_head_k, [klen, bsz, n_head, d_head])
    w_head_v = tf.reshape(w_head_v, [klen, bsz, n_head, d_head])

    # Preparing relative positional head weights
    r_head_k = tf.layers.dense(arg_r, n_head * d_head, use_bias=False,
                               kernel_initializer=kernel_initializer, name='r')

    rlen = tf.shape(arg_r)[0]
    r_head_k = tf.reshape(r_head_k, [rlen, n_head, d_head])
    rw_head_q = w_head_q + r_w_bias
    rr_head_q = w_head_q + r_r_bias

    # Masked Multi-head attention using relative positional encoding
    # 1. MatMul of Q and K
    AC = tf.einsum('ibnd,jbnd->ijbn', rw_head_q, w_head_k) #pylint: disable=invalid-name
    BD = tf.einsum('ibnd,jnd->ijbn', rr_head_q, r_head_k) #pylint: disable=invalid-name
    # because of relative position?????, R_{i-j}
    BD = rel_shift(BD) #pylint: disable=invalid-name

    # 2. Scale: Scaled A_{i,j}^rel
    attn_score = (AC + BD) * scale

    # 3. Mask
    attn_mask_t = attn_mask[:, :, None, None]
    attn_score = attn_score * (1 - attn_mask_t) - 1e30 * attn_mask_t

    # 4. Softmax
    attn_prob = tf.nn.softmax(attn_score, 1)
    attn_prob = tf.layers.dropout(attn_prob, dropatt, training=is_training)

    # 5. MatMul of output and V and reshaping the output vectors of attention
    attn_vec = tf.einsum('ijbn,jbnd->ibnd', attn_prob, w_head_v)
    size_t = tf.shape(attn_vec)
    attn_vec = tf.reshape(attn_vec, [size_t[0], size_t[1], n_head * d_head])
    attn_out = tf.layers.dense(attn_vec, d_model, use_bias=False,
                               kernel_initializer=kernel_initializer, name='o')
    attn_out = tf.layers.dropout(attn_out, dropout, training=is_training)

    # Add & Norm
    output = tf.contrib.layers.layer_norm(attn_out + arg_w, begin_norm_axis=-1)
  return output


def scaled_embedding_lookup(token_idx, n_token, d_embed, d_proj,
                            initializer,
                            proj_initializer,
                            scope='adaptive_embed'):
  """

  :param token_idx: a current token index
  :param n_token: a number of tokens
  :param d_embed: a embedding dimension
  :param d_proj: a projection dimension => if it is different from d_embed
                                          then one more linear projection will
                                          be performed
  :param initializer: init function for hidden weights
  :param proj_initializer: init function for projection weights
  :param scope: a name of variable scope
  :return: scaled_embedding vector, lookup table, projection weight
  """
  with tf.compat.v1.variable_scope(scope):
    lookup_table =\
      tf.compat.v1.get_variable('lookup_table',
                                [n_token, d_embed],
                                initializer=initializer)
    embedded_x = tf.nn.embedding_lookup(lookup_table, token_idx)

    # Linear projection if d_proj and d_embed are different
    if d_proj != d_embed:
      proj_wgt =\
        tf.compat.v1.get_variable('proj_W', [d_embed, d_proj],
                                  initializer=proj_initializer)
      embedded_x = tf.einsum('ibe,ed->ibd', embedded_x, proj_wgt)
    else:
      proj_wgt = None
    ret_params = [lookup_table, proj_wgt]

  # Question: why is sqrt(d_proj) a embedding scale?
  emb_scale = d_proj ** 0.5
  return embedded_x * emb_scale, ret_params


def logsoftmax(hidden, target, n_token, params, scope='adaptive_softmax',
               return_mean=True):
  params_weight, params_projs = params[0], params[1]

  with tf.compat.v1.variable_scope(scope):
    softmax_b =\
      tf.compat.v1.get_variable('bias', [n_token],
                                initializer=tf.zeros_initializer())
    arg_y = hidden
    if params_projs is not None:
      arg_y = tf.einsum('ibd,ed->ibe', arg_y, params_projs)
    output = tf.einsum('ibd,nd->ibn', arg_y, params_weight) + softmax_b

    nll = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target,
                                                         logits=output)
  if return_mean:
    nll = tf.reduce_mean(nll)

  return nll, output


def transformer(dec_inp, target, mems, n_token, n_layer, d_model, d_embed,
                n_head, d_head, d_inner, dropout, dropatt, initializer,
                is_training, proj_initializer=None, mem_len=None,
                same_length=False, clamp_len=-1, untie_r=False,
                scope='transformer'):
  """

  :param dec_inp:
  :param target:
  :param mems:
  :param n_token:
  :param n_layer:
  :param d_model:
  :param d_embed:
  :param n_head:
  :param d_head:
  :param d_inner:
  :param dropout:
  :param dropatt:
  :param initializer:
  :param is_training:
  :param proj_initializer:
  :param mem_len:
  :param same_length:
  :param clamp_len:
  :param untie_r: whether to share r_w_baias and r_r_bias
  :param scope:
  :return:
  """
  new_mems = [] # new cached memories from this segment

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

    # input time length
    qlen = tf.shape(dec_inp)[0]
    # memory length
    mlen = tf.shape(mems[0])[0] if mems is not None else 0
    # hat{h}_tau+1^n-1 = concat(SG(h_tau^n-1), h_tau+1^n-1)
    klen = mlen + qlen # in order to concatenate lower layer of
                       # past segment's h and current segment's h

    if proj_initializer is None:
      proj_initializer = initializer

    # Output embedding
    embeddings, shared_params =\
      scaled_embedding_lookup(token_idx=dec_inp, n_token=n_token, d_embed=d_embed,
                              d_proj=d_model, initializer=initializer,
                              proj_initializer=proj_initializer)
    output = tf.layers.dropout(embeddings, dropout, training=is_training)

    # Positional embedding
    pos_seq = tf.range(klen - 1, -1, -1.0)
    if clamp_len > 0:
      pos_seq = tf.minimum(pos_seq, clamp_len)
    inv_freq = 1 / (10000 ** (tf.range(0, d_model, 2.0) / d_model))
    pos_emb = positional_embedding(pos_seq, inv_freq)
    pos_emb = tf.layers.dropout(pos_emb, dropout, training=is_training)

    # Creating attention mask
    attn_mask = tf.ones([qlen, qlen])
    mask_u = tf.linalg.band_part(attn_mask, 0, -1)
    mask_dia = tf.linalg.band_part(attn_mask, 0, 0)
    attn_mask_pad = tf.zeros([qlen, mlen])
    ret = tf.concat([attn_mask_pad, mask_u - mask_dia], 1)

    # Initializing variables for the Transformer network
    if mems is None:
      mems = [None] * n_layer

    if same_length:
      mask_l = tf.linalg.band_part(attn_mask, -1, 0)
      ret = tf.concat([ret[:, :qlen] + mask_l - mask_dia, ret[:, qlen:]], 1)
    attn_mask = ret

    # Transformer network consists of n layers
    for layer_idx in range(n_layer):
      # cache new mems
      def _cache_mem(curr_out, prev_mem, cache_mem_len=None):
        if cache_mem_len is None or prev_mem is None:
          new_mem = curr_out
        elif cache_mem_len == 0:
          return prev_mem
        else:
          new_mem = tf.concat([prev_mem, curr_out], 0)[-cache_mem_len:]  # pylint: disable=invalid-unary-operand-type

        return tf.stop_gradient(new_mem)

      new_mems.append(_cache_mem(output, mems[layer_idx], mem_len))

      with tf.compat.v1.variable_scope('layer_{}'.format(layer_idx)):
        output =\
          rel_multihead_attn(arg_w=output,
                             arg_r=pos_emb,
                             r_w_bias=\
                               r_w_bias if not untie_r else r_w_bias[layer_idx],
                             r_r_bias=\
                               r_r_bias if not untie_r else r_r_bias[layer_idx],
                             attn_mask=attn_mask,
                             mems=mems[layer_idx],
                             d_model=d_model,
                             n_head=n_head,
                             d_head=d_head,
                             dropout=dropout,
                             dropatt=dropatt,
                             is_training=is_training,
                             kernel_initializer=initializer)

        output =\
          positionwise_FF(inp=output,
                          d_model=d_model,
                          d_inner=d_inner,
                          dropout=dropout,
                          kernel_initializer=initializer,
                          is_training=is_training)

    output = tf.layers.dropout(output, dropout, training=is_training)

    loss, output = logsoftmax(hidden=output, target=target, n_token=n_token,
                              params=shared_params)

    return loss, new_mems, output
