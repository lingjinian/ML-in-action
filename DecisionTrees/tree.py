"""
P54 3.1.1
data_set:不浮出水面是否可以生存、是否有脚蹼、属于鱼类
熵越高，则混合的数据也越多，我们可以在数据集中添加更多的分类，观察熵是如何变化的。
"""
from math import log
import operator


def create_data_set():
  data_set = [[1, 1, 'yes'],
              [1, 1, 'yes'],
              [1, 0, 'no'],
              [0, 1, 'no'],
              [0, 1, 'no']]
  labels = ['no surfacing', 'flippers']
  # change to discrete values
  return data_set, labels


def calc_shannon_ent(data_set):
  num_entries = len(data_set)  # 5
  label_counts = {}
  for feat_vec in data_set:  # the the number of unique elements and their occurance
    current_label = feat_vec[-1]
    if current_label not in label_counts.keys():
      label_counts[current_label] = 0
    label_counts[current_label] += 1
  shannon_ent = 0.0
  for key in label_counts:  # {'yes': 2, 'no': 3}
    prob = float(label_counts[key]) / num_entries
    shannon_ent -= prob * log(prob, 2)  # log base 2
  return shannon_ent  # -2/5*log2(2/5)-3/5*log2(3/5)


def split_data_set(data_set, axis, value):  # 待划分的数据集、划分数据集的特征、特征的返回值
  ret_data_set = []
  for feat_vec in data_set:
    if feat_vec[axis] == value:  # 比较data_set每一行第axis+1个元素是否等于value
      reduced_feat_vec = feat_vec[:axis]   # 删除划分数据特征值的列
      reduced_feat_vec.extend(feat_vec[axis+1:])
      ret_data_set.append(reduced_feat_vec)
  return ret_data_set


def choose_best_feature_to_split(data_set):
  num_features = len(data_set[0]) - 1    # 最后一列为labels 特征值2列
  base_entropy = calc_shannon_ent(data_set)  # 0.9709505944546686
  best_info_gain = 0.0
  best_feature = -1
  for i in range(num_features):    # 循环所有特征列
    feat_list = [example[i] for example in data_set]  # 获取每一列所有特征值，由于不是数组，不能直接选取列
    unique_vals = set(feat_list)
    new_entropy = 0.0
    for value in unique_vals:
      sub_data_set = split_data_set(data_set, i, value)
      prob = len(sub_data_set)/float(len(data_set))
      new_entropy += prob * calc_shannon_ent(sub_data_set)
    info_gain = base_entropy - new_entropy   # calculate the info gain; ie reduction in entropy
    if info_gain > best_info_gain:     # compare this to the best gain so far
      best_info_gain = info_gain     # if better than current best, set to best
      best_feature = i
  return best_feature            # returns an integer


def majority_cnt(class_list):
  class_count = {}
  for vote in class_list:
    if vote not in class_count.keys():
      class_count[vote] = 0
    class_count[vote] += 1
  sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
  return sorted_class_count[0][0]


def create_tree(data_set, labels):
  class_list = [example[-1] for example in data_set]  # label list
  if class_list.count(class_list[0]) == len(class_list):  # label完全相同时返回label
    return class_list[0]
  if len(data_set[0]) == 1:  # stop splitting when there are no more features in data_set
    # 遍历所有特征返回出现次数最多的
    # 第二个停止条件：使用完了所有特征
    return majority_cnt(class_list)
  best_feat = choose_best_feature_to_split(data_set)  # 0 data_set列号
  best_feat_label = labels[best_feat]  # no surfacing
  my_tree = {best_feat_label: {}}  # {'no surfacing':{}}
  del(labels[best_feat])  # ['flippers']
  feat_values = [example[best_feat] for example in data_set]  # feature list 列号为best_feat
  unique_vals = set(feat_values)
  for value in unique_vals:
    sub_labels = labels[:]     # copy all of labels, so trees don't mess up existing labels
    my_tree[best_feat_label][value] = create_tree(split_data_set(data_set, best_feat, value), sub_labels)
  return my_tree


def classify(input_tree, feat_labels, test_vec):
  first_str = list(input_tree.keys())[0]
  second_dict = input_tree[first_str]
  feat_index = feat_labels.index(first_str)
  key = test_vec[feat_index]
  value_of_feat = second_dict[key]
  if isinstance(value_of_feat, dict):
    class_label = classify(value_of_feat, feat_labels, test_vec)
  else:
    class_label = value_of_feat
  return class_label


def store_tree(input_tree, filename):
  import pickle
  fw = open(filename, 'w')
  pickle.dump(input_tree, fw)
  fw.close()


def grab_tree(filename):
  import pickle
  fr = open(filename)
  return pickle.load(fr)


if __name__ == '__main__':
  my_dat, labels = create_data_set()
  my_label = labels.copy()
  my_tree = create_tree(my_dat, labels)
  print(my_tree)
  print(classify(my_tree, my_label, [1, 0]))
