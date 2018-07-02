import h5py
import numpy as np
import json

class Reader:
  def __init__(self, data_path, pool5_path, fc6_path, conv5b_path, voc_path, use_video_feature=True, bsz=32):
    self.f = h5py.File(data_path, 'r')
    self.pool5 = h5py.File(pool5_path, 'r')
    self.fc6 = h5py.File(fc6_path, 'r')
    self.conv5b = h5py.File(conv5b_path, 'r')
    self.question = self.f['question']
    self.sen_len = self.f['sen_len']
    self.ans = self.f['ans']
    self.vid = self.f['id']
    self.bsz = bsz
    self.current_batch = 0
    self.num_batch = len(self.ans) // self.bsz
    self.batch_id = np.random.choice(range(len(self.ans)), len(self.ans), replace=False)
    self.use_video_feature = use_video_feature
      
    with open(voc_path, 'r') as f:
      data = json.load(f)
    self.max_sen_len = data['max_sen_len']
    self.word_question = data['word_question']
    self.word_ans = data['word_ans']
    self.num_ques_word = len(self.word_question)
    self.num_ans = len(self.word_ans)

  def next_batch(self):
    pool5 = []
    fc6 = []
    conv5b = []
    question = []
    sen_len = []
    ans = []
    vid = []
    idx = self.batch_id[self.current_batch * self.bsz:(self.current_batch + 1) * self.bsz]
    for i in idx:
      question.append(self.question[i])
      sen_len.append(self.sen_len[i])
      ans.append(self.ans[i])
      vid.append(self.vid[i])
      if self.use_video_feature:
        pool5.append(self.pool5[str(vid[-1])][:])
        fc6.append(self.fc6[str(vid[-1])][:])
        conv5b.append(self.conv5b[str(vid[-1])][:])
    
    self.current_batch += 1
    if self.current_batch == self.num_batch:
      self.current_batch = 0
      self.batch_id = np.random.choice(range(len(self.ans)), len(self.ans), replace=False)
    
    return fc6, pool5, conv5b, question, sen_len, ans, vid