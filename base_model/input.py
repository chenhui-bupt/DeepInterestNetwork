import numpy as np

class DataInput:
  def __init__(self, data, batch_size):

    self.batch_size = batch_size
    self.data = data
    self.epoch_size = len(self.data) // self.batch_size
    if self.epoch_size * self.batch_size < len(self.data):
      self.epoch_size += 1
    self.i = 0

  def __iter__(self):
    return self

  def __next__(self):

    if self.i == self.epoch_size:
      raise StopIteration

    ts = self.data[self.i * self.batch_size : min((self.i+1) * self.batch_size,
                                                  len(self.data))]  # batch
    self.i += 1

    u, i, y, sl = [], [], [], []
    for t in ts:  # ts: batch; t: reviewerID, hist, pos_list[i], 1 或者 reviewerID, hist, neg_list[i], 0
      u.append(t[0])  # user
      i.append(t[2])  # item
      y.append(t[3])  # label
      sl.append(len(t[1]))  # history length
    max_sl = max(sl)

    hist_i = np.zeros([len(ts), max_sl], np.int64)  # 小批量一块做padding，减少存储节约计算

    k = 0
    for t in ts:  # safety copy hist list in ts to hist_i
      for l in range(len(t[1])):
        hist_i[k][l] = t[1][l]
      k += 1

    return self.i, (u, i, y, hist_i, sl)

class DataInputTest:
  def __init__(self, data, batch_size):

    self.batch_size = batch_size
    self.data = data
    self.epoch_size = len(self.data) // self.batch_size
    if self.epoch_size * self.batch_size < len(self.data):
      self.epoch_size += 1
    self.i = 0

  def __iter__(self):
    return self

  def __next__(self):

    if self.i == self.epoch_size:
      raise StopIteration

    ts = self.data[self.i * self.batch_size : min((self.i+1) * self.batch_size,
                                                  len(self.data))]
    self.i += 1

    u, i, j, sl = [], [], [], []
    for t in ts:  #
      u.append(t[0])  # user
      i.append(t[2][0])  # pos item
      j.append(t[2][1])  # neg item
      sl.append(len(t[1]))  # hist length
    max_sl = max(sl)  # max hist length

    hist_i = np.zeros([len(ts), max_sl], np.int64)

    k = 0
    for t in ts:  # safety copy hist list in ts to hist_i
      for l in range(len(t[1])):
        hist_i[k][l] = t[1][l]
      k += 1

    return self.i, (u, i, j, hist_i, sl)
