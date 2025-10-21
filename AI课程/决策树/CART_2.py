import numpy as np
from collections import Counter
import pandas as pd


class Node:

    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None, major=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.major = major

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

    def _eval(self, tree, vx, vy):
        return sum([1 if self._search(vx[i], tree) == vy[i] else 0 for i in range(len(vy))]) / len(vy)

    def post_prune(self, tree: Node, vx, vy):
        if tree.is_leaf():
            return
        if tree.left:
            self.post_prune(tree.left, vx, vy)
        if tree.right:
            self.post_prune(tree.right, vx, vy)
        noraml_acc = self._eval(tree, vx, vy)
        tree.value = tree.major
        leaf_acc = self._eval(tree, vx, vy)
        if noraml_acc <= leaf_acc:
            tree.left = None
            tree.right = None
            tree.feature = None
            tree.threshold = None
        else:
            tree.value = None

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
        res = Counter(y).most_common(1)[0][0]
        if data_num < self.min_samples_split or depth >= self.max_depth or class_num == 1:
            return Node(value=res, major=res)
        feature, threshold = self._find_threshold(x, y)
        if feature is None:
            return Node(value=res, major=res)
        left_mask = x[:, feature] <= threshold
        right_mask = ~left_mask
        left = self._create(x[left_mask], y[left_mask], depth + 1)
        right = self._create(x[right_mask], y[right_mask], depth + 1)
        return Node(feature=feature, threshold=threshold, left=left, right=right, major=res)

    def predict(self, x):
        return [self._search(each, self.root) for each in x]

    def _search(self, x, tree: Node):
        if tree.is_leaf():
            return tree.value
        if x[tree.feature] > tree.threshold:
            return self._search(x, tree.right)
        else:
            return self._search(x, tree.left)


df = pd.read_csv("heartdisease.csv")
data = df.to_numpy()
tot = data.shape[0]
for EPOCH in range(20):
    indices = np.random.permutation(tot)
    r = [0.2, 0.1]

    test_len = int(r[0] * tot)
    eval_len = int(r[1] * tot)

    test_x = data[indices[:test_len], :-1]
    eval_x = data[indices[test_len : test_len + eval_len], :-1]
    train_x = data[indices[test_len + eval_len :], :-1]
    test_y = data[indices[:test_len], -1]
    eval_y = data[indices[test_len : test_len + eval_len], -1]
    train_y = data[indices[test_len + eval_len :], -1]

    tree = DecisionTree(min_samples_split=2, max_depth=10)
    tree.fit(train_x, train_y)
    tree.post_prune(tree.root, eval_x, eval_y)
    predictions = tree.predict(test_x)
    res = sum([1 if predictions[i] == test_y[i] else 0 for i in range(test_len)]) / test_len * 100
    print("EPOCH %d: %.2f%%" % (EPOCH + 1, res))
