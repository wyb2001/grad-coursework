import numpy as np
from collections import Counter
import pandas as pd


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

    def fit(self, x, y, eval_x, eval_y):
        x = np.array(x)
        y = np.array(y)
        self.root = self._create(x, y, eval_x, eval_y, 0)

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

    def _create(self, x, y, vx, vy, depth):
        data_num = x.shape[0]
        class_num = len(np.unique(y))
        if data_num < self.min_samples_split or depth >= self.max_depth or class_num == 1:
            res = Counter(y).most_common(1)[0][0]
            return Node(value=res)
        feature, threshold = self._find_threshold(x, y)

        if feature is None:
            res = Counter(y).most_common(1)[0][0]
            return Node(value=res)
        now_res = Counter(y).most_common(1)[0][0]
        acc_now = np.mean(vy == now_res)
        left_mask = x[:, feature] <= threshold
        right_mask = ~left_mask
        left_res = Counter(y[left_mask]).most_common(1)[0][0]
        right_res = Counter(y[right_mask]).most_common(1)[0][0]
        left_v_mask = vx[:, feature] <= threshold
        right_v_mask = ~left_v_mask
        left_vy = vy[left_v_mask]
        right_vy = vy[right_v_mask]
        acc_split = (np.sum(left_vy == left_res) + np.sum(right_vy == right_res)) / len(vy)
        if acc_split > acc_now:
            left = self._create(x[left_mask], y[left_mask], vx[left_v_mask], left_vy, depth + 1)
            right = self._create(x[right_mask], y[right_mask], vx[right_v_mask], right_vy, depth + 1)
            return Node(feature=feature, threshold=threshold, left=left, right=right)
        else:
            return Node(value=now_res)

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

    tree = DecisionTree(
        min_samples_split=2,
        max_depth=10,
    )
    tree.fit(train_x, train_y, eval_x, eval_y)

    predictions = tree.predict(test_x)
    res = sum([1 if predictions[i] == test_y[i] else 0 for i in range(test_len)]) / test_len * 100
    print("EPOCH %d: %.2f%%" % (EPOCH + 1, res))
