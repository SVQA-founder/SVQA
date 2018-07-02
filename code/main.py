import h5py
import numpy as np
import tensorflow as tf
import argparse
import os
import tqdm
import json
from model import Model
from reader import Reader


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--pattern', type=str, default='train', choices=['train', 'test'])
  parser.add_argument('--dir', type=str)
  parser.add_argument('--gpu', type=str, default='0')
  parser.add_argument('--epoch', type=int, default=51)
  parser.add_argument('--base_learning_rate', type=float, default=0.002)
  parser.add_argument('--bsz', type=int, default=32)
  parser.add_argument('--checkpoint', type=str)
  args = parser.parse_args()
  os.environ['CUDA_VISIBLE_DEVICES']=args.gpu


QA_DATA ='./data/question/' + args.dir + '/train.h5' if args.pattern == 'train' else './data/question/' + args.dir + '/test.h5'

reader = Reader(QA_DATA,
            './data/pool5.h5',
            './data/fc6.h5',
            './data/conv5b.h5',
            './data/question/' + args.dir + '/vocabulary.json',
            use_video_feature=True,
            bsz=args.bsz)

model = Model(1024, reader.num_ques_word, reader.num_ans)

tf_pool5 = tf.placeholder(shape=[None,40,2048], dtype=tf.float32)
tf_fc6 = tf.placeholder(shape=[None,40,4096], dtype=tf.float32)
tf_conv5b = tf.placeholder(shape=[None,40,7,7,1024], dtype=tf.float32)
tf_question = tf.placeholder(shape=[None, reader.max_sen_len], dtype=tf.int32)
tf_sen_len = tf.placeholder(shape=[None], dtype=tf.int32)
tf_labels = tf.placeholder(shape=[None], dtype=tf.int32)

l2_tf_fc6 = tf.nn.l2_normalize(x=tf_fc6, dim=[2])
l2_tf_pool5 = tf.nn.l2_normalize(x=tf_pool5, dim=[2])
l2_tf_conv5b = tf.nn.l2_normalize(x=tf_conv5b, dim=[4])
tf_video = tf.concat([l2_tf_fc6, l2_tf_pool5], axis=2)

global_step = tf.Variable(0, trainable=False, name="global_step")
lr = tf.train.exponential_decay(
    args.base_learning_rate,
    global_step,
    reader.num_batch,
    0.95,
    staircase=True)

tf_loss, tf_acc = model.run_one_batch(tf_video, l2_tf_conv5b, tf_question, tf_sen_len, tf_labels)
train_op = tf.train.AdamOptimizer(lr).minimize(tf_loss, global_step=global_step)
saver = tf.train.Saver(max_to_keep=100)

def train():
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(args.epoch):
      total_loss = 0.0
      for i in tqdm.tqdm(range(reader.num_batch)):
        fc6, pool5, conv5b, question, sen_len, ans, id = reader.next_batch()
        _, loss = sess.run([train_op, tf_loss], feed_dict={
                                        tf_fc6:fc6,
                                        tf_pool5:pool5,
                                        tf_conv5b:conv5b,
                                        tf_question:question,
                                        tf_sen_len:sen_len,
                                        tf_labels:ans})
        total_loss += loss
      print('epoch: {0}, loss: {1}'.format(epoch, total_loss / reader.num_batch))
      checkpoint_path = os.path.join('./model/ori3d/lstmAtt_' + args.dir, 'model_{0}.ckpt'.format(epoch))
      if not os.path.exists(os.path.dirname(checkpoint_path)):
        os.makedirs(os.path.dirname(checkpoint_path))
      saver.save(sess, checkpoint_path)

def test():
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  with tf.Session(config=config) as sess:
    if args.checkpoint:
      saver.restore(sess, args.checkpoint)
    else:
      ckpt = tf.train.get_checkpoint_state('./model/ori3d/lstmAtt_' + args.dir)
      print('Resotring params from {0} ...'.format(ckpt.model_checkpoint_path))
      saver.restore(sess, ckpt.model_checkpoint_path)

    
    f = open('./model/ori3d/lstmAtt_' + args.dir + '.txt', 'w')
    for i in tqdm.tqdm(range(reader.num_batch)):
      fc6, pool5, conv5b, question, sen_len, ans, id = reader.next_batch()
      acc = sess.run(tf_acc, feed_dict={
                          tf_fc6:fc6,
                          tf_pool5:pool5,
                          tf_conv5b:conv5b,
                          tf_question:question,
                          tf_sen_len:sen_len,
                          tf_labels:ans})
      for x in acc:
        if x == True:
          f.write('1\n')
        else:
          f.write('0\n')
    f.close()
    pd = []
    with open('./model/ori3d/lstmAtt_' + args.dir + '.txt') as f:
      for line in f:
        pd.append(int(line.strip()))

    gt = []
    with h5py.File(QA_DATA, 'r') as f:
      ans = f['ans']
      for x in ans:
        gt.append(x)

    cnt = {}
    for i in range(len(pd)):
      x = gt[i]
      if x not in cnt:
        cnt[x] = {0:0, 1:0}
      cnt[x][pd[i]] += 1

    r = 0
    w = 0
    for x in cnt:
      print(x, cnt[x])
      r += cnt[x][1]
      w += cnt[x][0]

    print('acc =', r * 1.0 / (r + w))


if __name__ == '__main__':
  if args.pattern == 'train':
    train()
  else:
    test()
