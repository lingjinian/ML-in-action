"""
P47 2.3.1
手写数字识别
"""
import numpy as np
import operator
from os import listdir


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


def img2vector(filename):  # 32*32转成1*1024
  return_vect = np.zeros((1, 1024))
  fr = open(filename)
  for i in range(32):
    line_str = fr.readline()
    for j in range(32):
      return_vect[0, 32 * i + j] = int(line_str[j])
  return return_vect


def handwriting_class_test():
  hw_labels = []
  training_file_list = listdir('trainingDigits')  # load the training set
  m = len(training_file_list)  # 1934
  training_mat = np.zeros((m, 1024))  # (1934, 1024)
  for i in range(m):
    file_name_str = training_file_list[i]
    file_str = file_name_str.split('.')[0]  # take off .txt
    class_num_str = int(file_str.split('_')[0])
    hw_labels.append(class_num_str)  # 从文件名提取label
    training_mat[i, :] = img2vector('trainingDigits/%s' % file_name_str)
  test_file_list = listdir('testDigits')  # iterate through the test set
  error_count = 0.0
  m_test = len(test_file_list)
  for i in range(m_test):
    file_name_str = test_file_list[i]
    file_str = file_name_str.split('.')[0]  # take off .txt
    class_num_str = int(file_str.split('_')[0])
    vector_under_test = img2vector('testDigits/%s' % file_name_str)
    classifier_result = classify0(vector_under_test, training_mat, hw_labels, 3)
    print("the classifier came back with: %d, the real answer is: %d" % (classifier_result, class_num_str))
    if classifier_result != class_num_str:
      error_count += 1.0
  print("\nthe total number of errors is: %d" % error_count)
  print("\nthe total error rate is: %f" % (error_count / float(m_test)))


if __name__ == '__main__':
  handwriting_class_test()
