"""
P42 2.2.2
分别对特征值两两组合，以label区分不同的类别
实际上可以把三个特征值全部放到三维坐标，kNN3中就是以三维坐标比较临近点
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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


if __name__ == '__main__':
  dating_data_mat, dating_labels = file2matrix('datingTestSet2.txt')
  
  fig = plt.figure(figsize=(16, 12))
  ax = fig.add_subplot(221)
  # add_subplot()返回一个axes对象，里面的参数abc表示在一个figure窗口中，有a行b列个小窗口，然后本次plot在第c个窗口中。
  # x轴、y轴、点大小、点颜色
  ax.scatter(dating_data_mat[:, 0], dating_data_mat[:, 1], 15.0 * np.array(dating_labels), 15.0 * np.array(dating_labels))
  plt.xlabel("Frequent Flyier Miles Earned Per Year")
  plt.ylabel("Percentage of Time Spent Playing Video Games")

  ax = fig.add_subplot(222)
  ax.scatter(dating_data_mat[:, 1], dating_data_mat[:, 2], 15.0 * np.array(dating_labels), 15.0 * np.array(dating_labels))
  plt.xlabel("Percentage of Time Spent Playing Video Games")
  plt.ylabel("Liters of Ice Cream Consumed Per Week")

  ax = fig.add_subplot(223)
  ax.scatter(dating_data_mat[:, 0], dating_data_mat[:, 2], 15.0 * np.array(dating_labels), 15.0 * np.array(dating_labels))
  plt.xlabel("Frequent Flyier Miles Earned Per Year")
  plt.ylabel("Liters of Ice Cream Consumed Per Week")
  plt.show()

  fig = plt.figure()
  ax = Axes3D(fig)
  #  将数据点分成三部分画，在颜色上有区分度
  ax.scatter(dating_data_mat[:, 0], dating_data_mat[:, 1], dating_data_mat[:, 2],
             1.0 * np.array(dating_labels), 15 * np.array(dating_labels), 15.0 * np.array(dating_labels))  # 绘制数据点

  ax.set_zlabel('Liters of Ice Cream Consumed Per Week')  # 坐标轴
  ax.set_ylabel('Percentage of Time Spent Playing Video Games')
  ax.set_xlabel('Frequent Flyier Miles Earned Per Year')
  plt.show()
