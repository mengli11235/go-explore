"""
The MIT License (MIT) Copyright (c) 2020 Andrej Karpathy

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

"""
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""

import math
import logging
from typing import Any
import joblib
import numpy as np
import tensorflow as tf
import horovod.tensorflow as hvd
import sys
import copy
import cv2
from baselines.common.distributions import make_pdtype

logger = logging.getLogger(__name__)

def torch_gather(x, indices, gather_axis):
    # if pytorch gather indices are
    # [[[0, 10, 20], [0, 10, 20], [0, 10, 20]],
    #  [[0, 10, 20], [0, 10, 20], [0, 10, 20]]]
    # tf nd_gather needs to be
    # [[0,0,0], [0,0,10], [0,0,20], [0,1,0], [0,1,10], [0,1,20], [0,2,0], [0,2,10], [0,2,20],
    #  [1,0,0], [1,0,10], [1,0,20], [1,1,0], [1,1,10], [1,1,20], [1,2,0], [1,2,10], [1,2,20]]

    # create a tensor containing indices of each element
    all_indices = tf.where(tf.fill(indices.shape, True))
    gather_locations = tf.reshape(indices, [indices.shape.num_elements()])

    # splice in our pytorch style index at the correct axis
    gather_indices = []
    for axis in range(len(indices.shape)):
        if axis == gather_axis:
            gather_indices.append(gather_locations)
        else:
            gather_indices.append(all_indices[:, axis])

    gather_indices = tf.stack(gather_indices, axis=-1)
    gathered = tf.gather_nd(x, gather_indices)
    reshaped = tf.reshape(gathered, indices.shape)
    return reshaped

def gelu(x):
  """Gaussian Error Linear Unit.
  This is a smoother version of the RELU.
  Original paper: https://arxiv.org/abs/1606.08415
  Args:
    x: float Tensor to perform activation.
  Returns:
    `x` with the GELU activation applied.
  """
  cdf = 0.5 * (1.0 + tf.math.tanh(
      (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.math.pow(x, 3)))))
  return x * cdf

def to2d(x):
    size = 1
    for shapel in x.shape.as_list()[1:]:
        size *= shapel
    return tf.reshape(x, (-1, size))


def normc_init(std=1.0, axis=0):
    """
    Initialize with normalized columns
    """

    # noinspection PyUnusedLocal
    def _initializer(shape, dtype=None, partition_info=None):  # pylint: disable=W0613
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=axis, keepdims=True))
        return tf.constant(out)
    return _initializer


def ortho_init(scale=1.0):
    # noinspection PyUnusedLocal
    def _ortho_init(shape, dtype, partition_info=None):  # pylint: disable=W0613
        shape = tuple(shape)
        if len(shape) == 2:
            flat_shape = shape
        elif len(shape) == 4:  # assumes NHWC
            flat_shape = (np.prod(shape[:-1]), shape[-1])
        else:
            raise NotImplementedError
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v  # pick the one with the correct shape
        q = q.reshape(shape)
        return (scale * q[:shape[0], :shape[1]]).astype(np.float32)
    return _ortho_init


def fc(x, scope, nout, init_scale=1.0, init_bias=0.0):
    with tf.compat.v1.variable_scope(scope):  # pylint: disable=E1129
        nin = x.shape.as_list()[1]
        w = tf.compat.v1.get_variable("w", [nin, nout], initializer=normc_init(init_scale))
        b = tf.compat.v1.get_variable("b", [nout], initializer=tf.constant_initializer(init_bias))
        return tf.matmul(x, w) + b


def conv(x, scope, noutchannels, filtsize, stride, pad='VALID', init_scale=1.0):
    with tf.compat.v1.variable_scope(scope):
        nin = x.shape.as_list()[3]
        w = tf.compat.v1.get_variable("w", [filtsize, filtsize, nin, noutchannels], initializer=ortho_init(init_scale))
        b = tf.compat.v1.get_variable("b", [noutchannels], initializer=tf.constant_initializer(0.0))
        z = tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding=pad)+b
        return z

def attn(x, n_head, idx, layer_past=None):
    B, T, C = x.shape.as_list()

    with tf.compat.v1.variable_scope(idx):
        mask = tf.Variable(np.tril(np.ones((T+1, T+1))).reshape((1, 1, T+1, T+1)), trainable=False)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = tf.transpose(tf.reshape(fc(tf.reshape(x, (B*T, C)), 'key'+idx, nout=C, init_scale=0.02), (B, T, n_head, C // n_head)), (0, 2, 1, 3)) # (B, nh, T, hs)
        q = tf.transpose(tf.reshape(fc(tf.reshape(x, (B*T, C)), 'value'+idx, nout=C, init_scale=0.02), (B, T, n_head, C // n_head)), (0, 2, 1, 3)) # (B, nh, T, hs)
        v = tf.transpose(tf.reshape(fc(tf.reshape(x, (B*T, C)), 'query'+idx, nout=C, init_scale=0.02), (B, T, n_head, C // n_head)), (0, 2, 1, 3)) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ tf.transpose(k, (0, 1, 3, 2))) * (1.0 / tf.math.sqrt(tf.cast(k.shape.as_list()[-1], dtype=tf.float32)))
        att = tf.where(mask[:,:,:T,:T] == 0, float('-inf'), att)

        att = tf.nn.softmax(att, axis=-1)
        att = tf.nn.dropout(att, 0.1)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = tf.reshape(tf.transpose(y, (0, 2, 1, 3)), (B, T, C)) # re-assemble all head outputs side by side

        # output projection
        y = tf.nn.dropout(fc(tf.reshape(y, (B*T, C)), 'att_output'+idx, nout=C, init_scale=0.02), 0.1)
        return tf.reshape(y, (B, T, C))


def blocks(x, n_head, idx):
    B, T, C = x.shape.as_list()

    with tf.compat.v1.variable_scope(idx+'attn'):
        ln1 = tf.keras.layers.LayerNormalization(center=False,scale=False, epsilon=1e-5)
        ln2 = tf.keras.layers.LayerNormalization(center=False,scale=False, epsilon=1e-5)
        x1 = ln1(x)
        x = x + attn(x1, n_head, idx)

        x2 = ln2(x)
        x2 = gelu(fc(tf.reshape(x2, (B*T, C)), 'bfc1_'+idx, nout=4*C, init_scale=0.02))
        x2 = tf.nn.dropout(fc(x2, 'bfc2_'+idx, nout=C, init_scale=0.02), 0.1)

        return x+tf.reshape(x2, (B, T, C))

class GPT(object):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, sess, ob_space, ac_space, nenv, nsteps, n_embed=768, n_head=12, n_layer=12, test_mode=False, reuse=False, goal_space=None):
        nh, nw, nc = ob_space.shape #12*80*105
        nbatch = nenv*nsteps
        ob_shape = (nbatch, nh, nw, nc)
        logger.info(f'goal_space.shape: {goal_space.shape}')
        goal_shape = tuple([nbatch] + list(goal_space.shape))
        logger.info(f'goal_shape: {goal_shape}')
        nact = ac_space.n
        ac_shape = (nbatch, nact)

        # use variables instead of placeholder to keep data on GPU if we're training
        nn_input = tf.compat.v1.placeholder(tf.int64, ob_shape, 'input')  # obs
        actions = tf.compat.v1.placeholder(tf.int64, (nbatch), 'actions')  # actions
        timesteps = tf.compat.v1.placeholder(tf.int64, (nenv, 1, 1), 'timesteps')  # timesteps
        goal = tf.compat.v1.placeholder(tf.float32, goal_shape, 'goal')  # goal
        mask = tf.compat.v1.placeholder(tf.int64, [nbatch], 'done_mask')  # mask (done t-1)
        entropy = tf.compat.v1.placeholder(tf.float32, [nbatch], 'entropy_factor')
        fake_actions = tf.compat.v1.placeholder(tf.int64, [nbatch], 'fake_actions')
        logger.info(f'fake_actions.shape: {fake_actions.shape}')
        logger.info(f'fake_actions.dtype: {fake_actions.dtype}')

        with tf.compat.v1.variable_scope("model", reuse=reuse):
        # transformer
            # input embedding stem
            #tok_emb = tf.keras.layers.Embedding(vocab_size, n_embd, embeddings_initializer=normc_init(0.02))
            # self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
            # pos_emb = tf.compat.v1.get_variable("pos_embed", [1, nsteps+1, n_embd], initializer=ortho_init(0.02))
            # global_pos_emb = tf.compat.v1.get_variable("global_pos_embed", [1, nsteps+1, n_embd], initializer=ortho_init(0.02))

            pos_emb =  tf.Variable(tf.zeros((1, 3*nsteps, n_embed)))
            global_pos_emb =  tf.Variable(tf.zeros((1, nsteps+1, n_embed)))
            # decoder head
            ln_f = tf.keras.layers.LayerNormalization(center=False,scale=False, epsilon=1e-5)

            action_embed = tf.keras.layers.Embedding(nact, n_embed, embeddings_initializer=normc_init(0.02))

            logger.info(f'input.shape {nn_input.shape}')
            h = tf.nn.relu(conv(tf.cast(tf.reshape(nn_input, (-1, nh, nw, nc)), tf.float32)/255., 'c1', noutchannels=64, filtsize=8, stride=4))
            logger.info(f'h.shape: {h.shape}')
            h2 = tf.nn.relu(conv(h, 'c2', noutchannels=128, filtsize=4, stride=2))
            logger.info(f'h2.shape: {h2.shape}')
            h3 = tf.nn.relu(conv(h2, 'c3', noutchannels=128, filtsize=3, stride=1))
            logger.info(f'h3.shape: {h3.shape}')
            h3 = to2d(h3)
            logger.info(f'h3.shape: {h3.shape}')
            input_embeddings = tf.nn.relu(fc(h3, 'fc1', nout=n_embed, init_scale=0.02))
            input_embeddings = tf.reshape(input_embeddings, (nenv, nsteps, n_embed))

            g1 = tf.cast(goal, tf.float32)
            logger.info(f'g1.shape: {g1.shape}')
            # h3 = tf.concat([h3, g1], axis=1)
            # logger.info(f'h3.shape: {h3.shape}')
            # layer_norma = tf.keras.layers.LayerNormalization(center=False,scale=False)
            #nn.init.normal_(self.action_embeddings[0].weight, mean=0.0, std=0.02)
            goal_embeddings =  tf.math.tanh(fc(goal,'goal_embed1', nout=n_embed, init_scale=0.02))
            goal_embeddings = tf.reshape(goal_embeddings, (nenv, nsteps, n_embed))

            if actions is not None: 
                action_embeddings = tf.math.tanh(action_embed(tf.reshape(actions, (nenv,nsteps))))
                action_embeddings = tf.reshape(action_embeddings, (nenv, nsteps, n_embed))
                # (batch, block_size, n_embd)

                token_embeddings = tf.reshape(tf.stack([goal_embeddings, input_embeddings, action_embeddings[:,-nsteps + int(test_mode):,:]], axis=1), (nenv, nsteps*3 - int(test_mode), n_embed))
                # token_embeddings[:,::3,:] = goal_embeddings
                # token_embeddings[:,1::3,:] = input_embeddings
                # token_embeddings[:,2::3,:] = action_embeddings[:,-nsteps + int(test_mode):,:]
            else: # only happens at very first timestep of evaluation
                # goal_embeddings =  tf.math.tanh(fc(goal,'goal_embed2', nout=n_embd, init_scale=0.02))
                token_embeddings = tf.reshape(tf.stack([goal_embeddings, input_embeddings], axis=-1), (nenv, nsteps*2, n_embed))
                # token_embeddings[:,::2,:] = goal_embeddings # really just [:,0,:]
                # token_embeddings[:,1::2,:] = input_embeddings # really just [:,1,:]
            all_global_pos_emb = tf.repeat(global_pos_emb, nenv, axis=0) # batch_size, traj_length, n_embd

            position_embeddings = torch_gather(all_global_pos_emb, tf.repeat(tf.reshape(timesteps, (nenv, 1, 1)), n_embed, axis=-1), gather_axis=1) + pos_emb[:, :token_embeddings.shape.as_list()[1], :]
            # (batch_size, 1, n_embd) + (1, traj_length, n_embd)

            x = tf.nn.dropout(token_embeddings + position_embeddings, 0.1)
            for i in range(n_layer):
                x = blocks(x, n_head, str(i))
            x = ln_f(x)
            logits = tf.reshape(fc(tf.reshape(x, (nenv*token_embeddings.shape.as_list()[1], n_embed)), 'head', nout=nact, init_scale=0.02), (nenv, token_embeddings.shape.as_list()[1], nact))
            vf_before_squeeze = tf.reshape(fc(tf.reshape(x, (nenv*token_embeddings.shape.as_list()[1], n_embed)), 'v', nout=1, init_scale=0.02), (nenv, token_embeddings.shape.as_list()[1], 1))
            if actions is not None:
                logits = logits[:, 1::3, :] # only keep predictions from input_embeddings
                vf_before_squeeze = vf_before_squeeze[:, 1::3, :]
            else:
                logits = logits[:, 1:, :]
                vf_before_squeeze = vf_before_squeeze[:, 1:, :]

            if test_mode:
                logits *= 2.
            else:
                logits /= tf.reshape(entropy, (nenv, nsteps, 1))
            vf = tf.squeeze(tf.reshape(vf_before_squeeze, (nenv*nsteps, 1)), axis=[1])
            vf1 = tf.reshape(vf, (nenv, nsteps))[:,-1]

        self.pdtype = make_pdtype(ac_space)
        self.pd = self.pdtype.pdfromflat(tf.reshape(logits, (nenv*nsteps, nact)))

        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        a01 = tf.reshape(a0, (nenv, nsteps))[:,-1]
        neglogp01 = tf.reshape(neglogp0, (nenv, nsteps))[:,-1]
        logger.info(f'a0.shape: {a0.shape}')
        logger.info(f'a0.dtype: {a0.dtype}')

        def step(local_ob, local_goal, local_actions, local_mask, local_timesteps, local_increase_ent):
            return sess.run([a01, vf1, neglogp01],
                            {nn_input: local_ob, mask: local_mask, timesteps: local_timesteps, entropy: local_increase_ent,
                             goal: local_goal, actions:local_actions})

        def step_fake_action(local_ob, local_goal, local_actions, local_mask, local_timesteps, local_increase_ent, local_fake_action):
            return sess.run([a01, vf1, neglogp01, neg_log_fake_a],
                            {nn_input: local_ob,
                             mask: local_mask,
                             timesteps: local_timesteps,
                             entropy: local_increase_ent,
                             goal: local_goal,
                             actions:local_actions,
                             fake_actions: local_fake_action})

        def value(local_ob, local_goal, local_actions, local_mask, local_timesteps, local_increase_ent):
            return sess.run(vf1, {nn_input: local_ob, mask: local_mask, timesteps: local_timesteps, entropy: local_increase_ent,
                             goal: local_goal, actions:local_actions})

        self.X = nn_input
        self.goal = goal
        self.M = mask
        self.A = actions
        self.T = timesteps
        self.E = entropy
        self.logits = logits
        self.vf = vf
        self.step = step
        self.step_fake_action = step_fake_action
        self.value = value
        self.block_size = nsteps

    def get_block_size(self):
        return self.block_size

class Model(object):
    def __init__(self):
        self.A = None
        self.VALID = None
        self.R = None
        self.OLDNEGLOGPAC = None
        self.OLDVPRED = None
        self.LR = None
        self.vpred = None
        self.entropy = None
        self.vf_loss = None
        self.pg_loss = None
        self.approxkl = None
        self.clipfrac = None
        self.dt_loss = None
        self.params = None
        self.l2_loss = None
        self.train_op = None
        self.loss_requested_dict = None
        self.loss_requested = None
        self.loss_names = None
        self.loss_enabled = None
        self.step = None
        self.step_fake_action = None
        self.value = None
        self.act_model = None
        self.train_model = None
        self.disable_hvd = None
        self.logits = None
        
    def init(self, model, ob_space, ac_space, nenv, nsteps, adam_epsilon=1e-6, load_path=None, test_mode=False, goal_space=None, disable_hvd=False):
        self.sess = tf.compat.v1.get_default_session()
        self.init_models(model, ob_space, ac_space, nenv, nsteps, test_mode, goal_space)
        self.init_loss(nenv, nsteps, disable_hvd)
        #self.loss = self.dt_loss
        self.loss = self.pg_loss - self.entropy * 1e-4 + self.vf_loss * 0.5 + 1e-7 * self.l2_loss #+  self.dt_loss * 1e-4

        self.finalize(load_path, adam_epsilon)

    def init_loss(self, nenv, nsteps, disable_hvd):

        # These objects are used to store the experience of all our rollouts.
        self.A = self.train_model.pdtype.sample_placeholder([nenv * nsteps], name='action')
        self.ADV = tf.compat.v1.placeholder(tf.float32, [nenv * nsteps], name='advantage')

        self.logits = self.train_model.logits

        # Valid allows you to ignore specific time-steps for the purpose of updating the policy
        self.VALID = tf.compat.v1.placeholder(tf.float32, [nenv * nsteps], name='valid')
        self.R = tf.compat.v1.placeholder(tf.float32, [nenv * nsteps], name='return')

        # The old negative log probability of each action (i.e. -log(pi_old(a_t|s_t)) )
        self.OLDNEGLOGPAC = tf.compat.v1.placeholder(tf.float32, [nenv * nsteps], name='neglogprob')

        # The old value prediction for each state in our rollout (i.e. V_old(s_t))
        self.OLDVPRED = tf.compat.v1.placeholder(tf.float32, [nenv * nsteps], name='valuepred')

        # This is just the learning rate
        self.LR = tf.compat.v1.placeholder(tf.float32, [], name='lr')

        # We ask the model for its value prediction
        self.vpred = self.train_model.vf

        # We ask the model for the negative log probability for each action, but given which state?
        neglogpac = self.train_model.pd.neglogp(self.A)

        # We ask the model for its entropy
        self.entropy = tf.math.reduce_mean(self.train_model.pd.entropy())

        self.dt_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(tf.one_hot(tf.stop_gradient(self.A), self.logits.shape.as_list()[-1]), tf.reshape(self.logits, (-1, self.logits.shape.as_list()[-1]))))
        #0.5*self.entropy 

        # The clipped value prediction, which is just vpred if the difference between vpred and OLDVPRED is smaller
        # than the clip range.
        vpredclipped = self.OLDVPRED + tf.clip_by_value(self.vpred - self.OLDVPRED, -0.1, 0.1)
        vf_losses1 = tf.math.square(self.vpred - self.R)
        vf_losses2 = tf.math.square(vpredclipped - self.R)
        self.vf_loss = .5 * tf.math.reduce_mean(self.VALID * tf.math.maximum(vf_losses1, vf_losses2))

        # This is pi_current(a_t|s_t) / pi_old(a_t|s_t)
        ratio = tf.math.exp(self.OLDNEGLOGPAC - neglogpac)
        pg_losses = -self.ADV * ratio
        pg_losses2 = -self.ADV * tf.clip_by_value(ratio, 1.0 - 0.1, 1.0 + 0.1)
        self.pg_loss = tf.math.reduce_mean(self.VALID * tf.math.maximum(pg_losses, pg_losses2))
        mv = tf.math.reduce_mean(self.VALID)

        # This is the KL divergence (approximated) between the old and the new policy
        # (i.e. KL(pi_current(a_t|s_t), pi_old(a_t|s_t))
        self.approxkl = .5 * tf.math.reduce_mean(self.VALID * tf.math.square(neglogpac - self.OLDNEGLOGPAC)) / mv
        self.clipfrac = tf.math.reduce_mean(self.VALID * tf.compat.v1.to_float(tf.math.greater(tf.math.abs(ratio - 1.0), 0.1))) / mv
        self.params = tf.compat.v1.trainable_variables()
        self.l2_loss = .5 * sum([tf.math.reduce_sum(tf.math.square(p)) for p in self.params])
        self.disable_hvd = disable_hvd

    def init_models(self, model, ob_space, ac_space, nenv, nsteps, test_mode, goal_space):
        # At test time, we only need the most recent action in order to take a step.
        self.act_model = model(self.sess, ob_space, ac_space, nenv, nsteps, test_mode=test_mode, reuse=False, goal_space=goal_space)
        # At training time, we need to keep track of the last T (nsteps) of actions that we took.
        self.train_model = model(self.sess, ob_space, ac_space, nenv, nsteps, test_mode=test_mode, reuse=True, goal_space=goal_space)


    def finalize(self, load_path, adam_epsilon):
        opt = tf.compat.v1.train.AdamOptimizer(self.LR, epsilon=adam_epsilon)
        if not self.disable_hvd:
            opt = hvd.DistributedOptimizer(opt)
        self.train_op = opt.minimize(self.loss)
        self.step = self.act_model.step
        self.step_fake_action = self.act_model.step_fake_action
        self.value = self.act_model.value

        self.sess.run(tf.compat.v1.global_variables_initializer())
        if load_path and hvd.rank() == 0:
            self.load(load_path)
        if not self.disable_hvd:
            self.sess.run(hvd.broadcast_global_variables(0))
        tf.compat.v1.get_default_graph().finalize()

        self.loss_requested_dict = {self.dt_loss: 'transformer_loss', self.pg_loss: 'policy_loss',
                                    self.vf_loss: 'value_loss',
                                    self.l2_loss: 'l2_loss',
                                    self.entropy: 'policy_entropy',
                                    self.approxkl: 'approxkl',
                                    self.clipfrac: 'clipfrac',
                                    self.train_op: ''}
        self.init_requested_loss()

    def init_requested_loss(self):
        self.loss_requested = []
        self.loss_names = []
        self.loss_enabled = []
        for key, value in self.loss_requested_dict.items():
            self.loss_requested.append(key)
            if value != '':
                self.loss_names.append(value)
                self.loss_enabled.append(True)
            else:
                self.loss_enabled.append(False)

    def filter_requested_losses(self, losses):
        result = []
        for loss, enabled in zip(losses, self.loss_enabled):
            if enabled:
                result.append(loss)
        return result

    def save(self, save_path):
        ps = self.sess.run(self.params)
        joblib.dump(ps, save_path)

    def load(self, load_path):
        loaded_params = joblib.load(load_path)
        restores = []
        for p, loaded_p in zip(self.params, loaded_params):
            restores.append(p.assign(loaded_p))
        self.sess.run(restores)

    def train_from_runner(self, lr: float, runner: Any):
        obs = runner.ar_mb_obs_2.reshape(self.train_model.X.shape)
        return self.train(lr,
                          obs,
                          runner.ar_mb_goals,
                          runner.ar_mb_timesteps,
                          runner.ar_mb_rets,
                          runner.ar_mb_advs,
                          runner.ar_mb_dones,
                          runner.ar_mb_actions,
                          runner.ar_mb_values,
                          runner.ar_mb_neglogpacs,
                          runner.ar_mb_valids,
                          runner.ar_mb_ent,)

    def train(self, lr, obs, goals, timesteps, returns, advs, masks, actions, values, neglogpacs, valids, increase_ent):
        td_map = {self.LR: lr, self.train_model.X: obs,  self.train_model.goal: goals, self.train_model.T: timesteps, self.A: actions, self.train_model.A: actions, self.ADV: advs, self.VALID: valids, self.R: returns, self.OLDNEGLOGPAC: neglogpacs, self.OLDVPRED: values, self.train_model.E: increase_ent}
        return self.sess.run(self.loss_requested, feed_dict=td_map)[:-1]

# x = tf.zeros((8,4,2))
# y = tf.ones((8,4,2))
# z = tf.reshape(tf.stack([x, y], axis=1), (8,8,2))
# print(z)