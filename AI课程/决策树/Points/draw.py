import numpy as np
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None


class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=10):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None

    def fit(self, x, y):
        x = np.array(x)
        y = np.array(y)
        self.root = self._create(x, y, 0)

    def _gini(self, y):
        cnt = Counter(y)
        tot = len(y)
        return 1 - sum((count / tot) ** 2 for count in cnt.values())

    def _find_threshold(self, x, y):
        gini = float("inf")
        feature = None
        threshold = None
        m, n = x.shape
        for each_feature in range(n):
            thresholds = np.unique(x[:, each_feature])
            for each_threshold in thresholds:
                left_mask = x[:, each_feature] <= each_threshold
                right_mask = ~left_mask
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                gini_left = self._gini(y[left_mask])
                gini_right = self._gini(y[right_mask])
                w_gini = (np.sum(left_mask) * gini_left + np.sum(right_mask) * gini_right) / m
                if w_gini < gini:
                    gini = w_gini
                    feature = each_feature
                    threshold = each_threshold
        return feature, threshold

    def _create(self, x, y, depth):
        data_num = x.shape[0]
        class_num = len(np.unique(y))
        if data_num < self.min_samples_split or depth >= self.max_depth or class_num == 1:
            res = Counter(y).most_common(1)[0][0]
            return Node(value=res)
        feature, threshold = self._find_threshold(x, y)
        if feature is None:
            res = Counter(y).most_common(1)[0][0]
            return Node(value=res)
        left_mask = x[:, feature] <= threshold
        right_mask = ~left_mask
        left = self._create(x[left_mask], y[left_mask], depth + 1)
        right = self._create(x[right_mask], y[right_mask], depth + 1)
        return Node(feature=feature, threshold=threshold, left=left, right=right)

    def predict(self, x):
        return [self._search(each, self.root) for each in x]

    def _search(self, x, tree: Node):
        if tree.is_leaf():
            return tree.value
        if x[tree.feature] > tree.threshold:
            return self._search(x, tree.right)
        else:
            return self._search(x, tree.left)

    # 新增：收集所有分割线
    def get_splits(self, node=None, splits=None):
        if splits is None:
            splits = []
        if node is None:
            node = self.root
        if node.is_leaf():
            return splits
        # 记录当前分割
        splits.append((node.feature, node.threshold))
        # 递归左右子树
        self.get_splits(node.left, splits)
        self.get_splits(node.right, splits)
        return splits


# === 主程序 ===
df = pd.read_csv("points.csv")
data = df.to_numpy()
tot = data.shape[0]

indices = np.random.permutation(tot)
r = 0.2
test_len = int(r * tot)
test_x = data[indices[:test_len], :-1]
train_x = data[indices[test_len:], :-1]
test_y = data[indices[:test_len], -1]
train_y = data[indices[test_len:], -1]

# 训练树
tree = DecisionTree(min_samples_split=2, max_depth=5)  # 建议降低 max_depth 便于可视化
tree.fit(train_x, train_y)

# 预测与准确率
predictions = tree.predict(test_x)
acc = sum(p == t for p, t in zip(predictions, test_y)) / test_len * 100
print("Accuracy: %.2f%%" % acc)

# === 绘图 ===
plt.figure(figsize=(10, 10))

# 分别绘制两类点，以便图例显示 Class 0 和 Class 1
plt.scatter(data[data[:, 2] == 0, 0], data[data[:, 2] == 0, 1],
            c='blue', s=30, alpha=0.6, label='Class 0')
plt.scatter(data[data[:, 2] == 1, 0], data[data[:, 2] == 1, 1],
            c='red', s=30, alpha=0.6, label='Class 1')

# 获取所有分割线
splits = tree.get_splits()

# 画分割线（不加 label，所以不会进图例）
for feature, thresh in splits:
    if feature == 0:  # 基于 x 分割 → 垂直线 x = thresh
        plt.axvline(x=thresh, color='green', linestyle='--', linewidth=3)
    elif feature == 1:  # 基于 y 分割 → 水平线 y = thresh
        plt.axhline(y=thresh, color='green', linestyle='--', linewidth=3)

# 设置图形
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.gca().set_aspect('equal', adjustable='box')
plt.xlim(-21, 21)
plt.ylim(-21, 21)
plt.title(f'Decision Tree Visualization\nAccuracy:%.2f%%'%acc)
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(False)
plt.legend()  # 现在会显示 Class 0 和 Class 1
plt.show()