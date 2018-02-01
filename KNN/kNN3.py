"""
P45 2.2.4
90%数据当作训练集，10%数据当作测试集
"""
import numpy as np
import operator


def classify0(in_x, data_set, labels, k):
  data_set_size = data_set.shape[0]
  diff_mat = np.tile(in_x, (data_set_size, 1)) - data_set
  sq_diff_mat = diff_mat ** 2
  sq_distances = sq_diff_mat.sum(axis=1)
  distances = sq_distances ** 0.5
  sorted_dist_indicies = distances.argsort()
  class_count = {}
  for i in range(k):
    vote_ilabel = labels[sorted_dist_indicies[i]]
    class_count[vote_ilabel] = class_count.get(vote_ilabel, 0) + 1
  sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
  return sorted_class_count[0][0]


def file2matrix(filename):
  fr = open(filename)
  number_of_lines = len(fr.readlines())  # get the number of lines in the file
  return_mat = np.zeros((number_of_lines, 3))  # prepare matrix to return
  class_label_vector = []  # prepare labels return
  fr = open(filename)
  index = 0
  for line in fr.readlines():
    line = line.strip()
    list_from_line = line.split('\t')
    return_mat[index, :] = list_from_line[0:3]
    class_label_vector.append(int(list_from_line[-1]))
    index += 1
  return return_mat, class_label_vector


def auto_norm(data_set):
  min_vals = data_set.min(0)  # 取每列最小值 组成一行
  max_vals = data_set.max(0)
  ranges = max_vals - min_vals
  m = data_set.shape[0]
  norm_data_set = data_set - np.tile(min_vals, (m, 1))
  norm_data_set = norm_data_set / np.tile(ranges, (m, 1))  # element wise divide
  return norm_data_set, ranges, min_vals


def dating_class_test():
  ho_ratio = 0.10  # hold out 10%
  dating_data_mat, dating_labels = file2matrix('datingTestSet2.txt')  # load data setfrom file
  norm_mat, ranges, min_vals = auto_norm(dating_data_mat)
  m = norm_mat.shape[0]
  num_test_vecs = int(m * ho_ratio)  # 测试样本中的10%数据 100个 其余900个为训练数据
  error_count = 0.0
  for i in range(num_test_vecs):
    classifier_result = classify0(norm_mat[i, :], norm_mat[num_test_vecs:m, :], dating_labels[num_test_vecs:m], 3)
    print("the classifier came back with: %d, the real answer is: %d" % (classifier_result, dating_labels[i]))
    if classifier_result != dating_labels[i]:
      error_count += 1.0
  print("the total error rate is: %f" % (error_count / float(num_test_vecs)))
  print(error_count)


if __name__ == '__main__':
  dating_class_test()
