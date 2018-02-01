"""
P36 2.1.1
坐标轴中的四个点分别对应label，判断给出的任意点的label
"""
from matplotlib import pyplot as plt
import numpy as np
import operator


def classify0(in_x, data_set, labels, k):
  data_set_size = data_set.shape[0]
  diff_mat = np.tile(in_x, (data_set_size, 1)) - data_set  # np.tile 重复inX dataSetSize行，1次
  sq_diff_mat = diff_mat ** 2
  sq_distances = sq_diff_mat.sum(axis=1)  # axis=1 每行相加
  distances = sq_distances ** 0.5
  sorted_dist_indicies = distances.argsort()  # 返回对应编号 从小到大依次是0 1 2 ...
  class_count = {}
  for i in range(k):
    vote_ilabel = labels[sorted_dist_indicies[i]]
    class_count[vote_ilabel] = class_count.get(vote_ilabel, 0) + 1
  sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
  return sorted_class_count[0][0]


def create_data_set():
  # 创建四个点，以及每个点的label
  group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
  labels = ['A', 'A', 'B', 'B']
  return group, labels


if __name__ == '__main__':
  group, lables = create_data_set()
  # 画出点的分布
  plt.plot(group[:, 0], group[:, 1], 'ro', label="point")
  plt.ylim(-0.2, 1.2)
  plt.xlim(-0.2, 1.2)
  plt.show()

  # 测试[4,5]所属类别
  print(classify0([4, 5], group, lables, 3))
