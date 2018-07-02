import tensorflow as tf
from sxm_cell import SXMCell

class Model:
  def __init__(self, num_units, num_ques_word, num_ans):
    self.num_units = num_units
    self.num_ques_word = num_ques_word
    self.num_ans = num_ans
    self.att_embed = 512
    self.batch_size = 32
  
  def lstmq(self, tf_question, tf_sen_len):
    with tf.variable_scope('embed'):
      embedding_mat = tf.Variable(tf.random_normal([self.num_ques_word, 300]))
      embedding_qus = tf.nn.embedding_lookup(embedding_mat, tf_question)
    with tf.variable_scope('lstmq'):
      cell = tf.contrib.rnn.GRUCell(num_units=self.num_units)
      outputs, state = tf.nn.dynamic_rnn(cell, embedding_qus, tf_sen_len, dtype=tf.float32)
    return state
  
  def lstmv(self, tf_feature):
    with tf.variable_scope('lstmv'):
      cell = tf.contrib.rnn.GRUCell(num_units=self.num_units)
      outputs, state = tf.nn.dynamic_rnn(cell, tf_feature, dtype=tf.float32)
      
    return state
  
  def att_c3d(self, tf_c3d, tf_qn):
    '''
    tf_c3d : b x t x 7 x 7 x 1024
    tf_qn  : b x 1024
    '''
    Wv  = tf.get_variable("att_c3d/Wv", shape=[1024,  1024], initializer=tf.contrib.layers.xavier_initializer())
    Wq  = tf.get_variable("att_c3d/Wq", shape=[1024, 1024], initializer=tf.contrib.layers.xavier_initializer())
    b   = tf.get_variable("att_c3d/b",  shape=[1024]     ,  initializer=tf.constant_initializer())
    Wo  = tf.get_variable("att_c3d/Wo", shape=[1024, 1],    initializer=tf.contrib.layers.xavier_initializer())
    qn  = tf.tile(tf_qn, [1, 49])
    qn  = tf.reshape(qn, [-1, 1024])
    Rq  = tf.matmul(qn, Wq)
    ret = []                         
    for i in range(40):
      cur_c3d        = tf_c3d[:,i,:,:,:]
      cur_c3d        = tf.reshape(cur_c3d, [-1, 1024])
      Rv             = tf.matmul(cur_c3d, Wv)
      hidden         = Rv + Rq + b                      
      act_h          = tf.tanh(hidden)             
      Ro             = tf.matmul(act_h, Wo)
      logits         = tf.reshape(Ro, [-1, 49])
      weight         = tf.nn.softmax(logits)
      weight         = tf.expand_dims(weight, 2)
      frame          = tf.reshape(cur_c3d, [-1, 49, 1024])
      weighted_frame = tf.multiply(frame, weight)
      avg            = tf.reduce_sum(weighted_frame, 1)
      ret.append(avg)

    ret = tf.convert_to_tensor(ret)                         
    ret = tf.transpose(ret, [1, 0, 2])

    return ret


  def attv(self, tf_feature, tf_question):
    cell = SXMCell(num_units=1024, batch_size=32, time_step=40)
    state = cell.zero_state()
    for i in range(40):
      output, state = cell([tf_feature[:,i,:], tf_question], state)

    return output

  def run_one_batch(self, tf_feature, tf_conv5b, tf_question, tf_sen_len, tf_labels):
    h = self.lstmq(tf_question, tf_sen_len)
    v = self.lstmv(tf_feature)
    av = self.attv(tf_feature, h)
    att_conv5b = self.att_c3d(tf_conv5b, h)
    with tf.variable_scope('3d'):
      av3 = self.attv(att_conv5b, h)
    info = tf.concat([h, v, av, av3], axis=1)

    with tf.variable_scope('mlp'):
      W1 = tf.Variable(tf.random_normal([self.num_units * 4, self.num_ans]), name='W1')
      b1 = tf.Variable(tf.zeros([self.num_ans]), name='b1')
      logits = tf.add(tf.matmul(info, W1), b1, name='fc1')


    labels = tf.one_hot(tf_labels, self.num_ans)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    correct_prediction = tf.equal(tf.argmax(logits,1),tf.argmax(labels,1))
    # accuracy = tf.reduce_sum(tf.cast(correct_prediction,tf.float32))
    return loss, correct_prediction