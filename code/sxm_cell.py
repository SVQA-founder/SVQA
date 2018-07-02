from tensorflow.python.ops.rnn_cell_impl import RNNCell, _linear
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
import tensorflow as tf
import collections
import numpy as np
tf.set_random_seed(0)

SXMState = collections.namedtuple('SXMState', ['h', 'i'])

class SXMCell(RNNCell):
  def __init__(self,
               num_units,
               batch_size,
               time_step,
               att_hidden=512,
               activation=None,
               reuse=None,
               kernel_initializer=None,
               bias_initializer=None):
    super(SXMCell, self).__init__(_reuse=reuse)
    self._num_units = num_units
    self._batch_size = batch_size
    self._time_step = time_step
    self._att_hidden = att_hidden
    self._activation = activation or math_ops.tanh
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer
    self._history = []

  @property
  def state_size(self):
    return (self._num_units, 1)
  
  @property
  def ouput_size(self):
    return self._num_units
  
  def zero_state(self):
    h = tf.zeros([self._batch_size, self._num_units])
    i = 1
    return SXMState(h=h,i=i)
  
  def call(self, inputs, state):
    inputs, encoded_question = inputs
    i = state.i
    state = state.h
    with tf.variable_scope("gates"):  # Reset gate and update gate.
      # We start with bias of 1.0 to not reset and not update.
      bias_ones = self._bias_initializer
      if self._bias_initializer is None:
        dtype = [a.dtype for a in [inputs, state]][0]
        bias_ones = init_ops.constant_initializer(1.0, dtype=dtype)
      value = math_ops.sigmoid(
          _linear([inputs, state], 2 * self._num_units, True, bias_ones,
                  self._kernel_initializer))
      r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)
    with tf.variable_scope("candidate"):
      c = self._activation(
          _linear([inputs, r * state], self._num_units, True, 
                  self._bias_initializer, self._kernel_initializer))
    new_h = u * state + (1 - u) * c
    self._history.append(new_h)
    
    with tf.variable_scope('attention'):
      question_dim = encoded_question.shape.as_list()
      hq = tf.tile(tf.expand_dims(encoded_question, 1), [1, self._time_step, 1])
      padding = tf.constant(0.0, shape=(self._batch_size, self._time_step - len(self._history), self._num_units))
      gru_h = tf.transpose(tf.convert_to_tensor(self._history), [1,0,2])
      gru_h = tf.concat([gru_h, padding], axis=1)
      hq = tf.reshape(hq, [-1, question_dim[-1]])
      gru_h = tf.reshape(gru_h, [-1, self._num_units])
      with tf.variable_scope('inner'):
        att = tf.tanh(_linear([gru_h, hq], self._att_hidden, True,
                      self._bias_initializer, self._kernel_initializer))
      with tf.variable_scope('outer'):
        att = _linear([att], 1, False,
                      self._bias_initializer, self._kernel_initializer)
        att = tf.reshape(att, [self._batch_size, self._time_step])
        att_mask = np.zeros([self._batch_size, self._time_step], dtype=np.float32)
        att_mask[:,i:] = 10000.0
        att_mask = tf.convert_to_tensor(att_mask)
        att = tf.reshape(tf.nn.softmax(att - att_mask), [-1, 1])
      final_h = tf.reduce_sum(tf.reshape(tf.multiply(gru_h, att), [self._batch_size, self._time_step, self._num_units]), axis=1)
      self._history[-1] = final_h
      
    return final_h, SXMState(h=final_h,i=i+1)


if __name__ == '__main__':
  cell = SXMCell(num_units=512, batch_size=32, time_step=40)
  state = cell.zero_state()
  input = tf.constant(np.random.normal(size=(32,5)), dtype=tf.float32, shape=(32,5))
  hq = tf.constant(np.random.normal(32, 512), dtype=tf.float32, shape=(32, 512))
  
  out_list = []
  for i in range(40):
    output, state = cell([input, hq], state)
    out_list.append(output)

  # p_list = []
  # gru_state = tf.zeros((32, 512))
  # gru = tf.contrib.rnn.GRUCell(num_units = 512)
  # for i in range(40):
  #   output, gru_state = gru(input, gru_state)
  #   p_list.append(output)
 
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for d in out_list:
      print(sess.run(d))
