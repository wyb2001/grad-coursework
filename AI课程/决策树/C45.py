import numpy as np
from collections import Counter
import pandas as pd
import math

class Node:
    def __init__(self, feature=None, threshold=None, children=None, left=None, right=None, value=None, is_discrete=False):
        self.feature = feature
        self.threshold = threshold
        self.children = children or {}
        self.left = left
        self.right = right
        self.value = value
        self.is_discrete = is_discrete

    def is_leaf(self):
        return self.value is not None


class C45:
    def __init__(self, min_samples_split=2, max_depth=10):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None
        self.feature_types = None

    def _entropy(self, y):
        if len(y) == 0:
            return 0
        probs = [count / len(y) for count in Counter(y).values()]
        return -sum(p * math.log2(p) for p in probs if p > 0)

    def _ent(self, subsets):
        total = sum(len(s) for s in subsets)
        if total == 0:
            return 1
        entropy= 0
        for subset in subsets:
            if len(subset) == 0:
                continue
            p = len(subset) / total
            entropy -= p * math.log2(p)
        return entropy if entropy > 0 else 1

    def _gain_ratio_continuous(self, y, y_left, y_right):
        before = self._entropy(y)
        after = (len(y_left) / len(y)) * self._entropy(y_left) + (len(y_right) / len(y)) * self._entropy(y_right)
        info_gain = before - after
        split = self._ent([y_left, y_right])
        return info_gain / split if split > 0 else 0

    def _gain_ratio_discrete(self, y, subsets):
        before = self._entropy(y)
        after = sum((len(s) / len(y)) * self._entropy(s) for s in subsets)
        info_gain = before - after
        split = self._ent(subsets)
        return info_gain / split if split > 0 else 0

    def _find_best_split(self, x, y):
        best_gr = -1
        best_feature = None
        best_threshold = None
        best_is_discrete = False
        best_children_vals = None

        for feat_idx in range(len(self.feature_types)):
            if self.feature_types[feat_idx] == "continuous":
                col = x[:, feat_idx]
                sorted_idx = np.argsort(col)
                x_sorted = col[sorted_idx]
                y_sorted = y[sorted_idx]

                gr = -1
                th = None
                for i in range(1, len(x_sorted)):
                    if y_sorted[i] != y_sorted[i - 1]: 
                        threshold = (x_sorted[i] + x_sorted[i - 1]) / 2
                        left_mask = col <= threshold
                        y_left = y[left_mask]
                        y_right = y[~left_mask]
                        if len(y_left) == 0 or len(y_right) == 0:
                            continue
                        gain_ratio = self._gain_ratio_continuous(y, y_left, y_right)
                        if gain_ratio > gr:
                            gr = gain_ratio
                            th = threshold
                if gr > best_gr:
                    best_gr = gr
                    best_feature = feat_idx
                    best_threshold = th
                    best_is_discrete = False
            else: 
                col = x[:, feat_idx]
                unique_vals = np.unique(col)
                subsets = [y[col == val] for val in unique_vals]
                if any(len(s) == 0 for s in subsets):
                    continue
                gain_ratio = self._gain_ratio_discrete(y, subsets)
                if gain_ratio > best_gr:
                    best_gr = gain_ratio
                    best_feature = feat_idx
                    best_is_discrete = True
                    best_children_vals = unique_vals

        return best_feature, best_threshold, best_is_discrete, best_children_vals

    def _create(self, x, y, depth):
        data_num = x.shape[0]
        class_num = len(np.unique(y))
        if data_num < self.min_samples_split or depth >= self.max_depth or class_num == 1:
            res = Counter(y).most_common(1)[0][0]
            return Node(value=res)

        feature, threshold, is_discrete, children_vals = self._find_best_split(x, y)
        if feature is None:
            res = Counter(y).most_common(1)[0][0]
            return Node(value=res)

        if is_discrete:
            col = x[:, feature]
            children = {}
            for val in children_vals:
                mask = col == val
                x_child = x[mask]
                y_child = y[mask]
                if len(y_child) == 0:
                    child = Node(value=Counter(y).most_common(1)[0][0])
                else:
                    child = self._create(x_child, y_child, depth + 1)
                children[val] = child
            return Node(feature=feature, children=children, is_discrete=True)
        else:
            left_mask = x[:, feature] <= threshold
            right_mask = ~left_mask
            left = self._create(x[left_mask], y[left_mask], depth + 1)
            right = self._create(x[right_mask], y[right_mask], depth + 1)
            return Node(feature=feature, threshold=threshold, left=left, right=right)

    def fit(self, x, y, feature_types):
        x = np.array(x)
        y = np.array(y)
        self.feature_types = feature_types
        self.root = self._create(x, y, 0)

    def _search(self, x, node):
        if node.is_leaf():
            return node.value
        if node.is_discrete:
            val = x[node.feature]
            if val in node.children:
                return self._search(x, node.children[val])
            else:
                preds = [self._search(x, child) for child in node.children.values()]
                return Counter(preds).most_common(1)[0][0]
        else:
            if x[node.feature] <= node.threshold:
                return self._search(x, node.left)
            else:
                return self._search(x, node.right)

    def predict(self, x):
        return [self._search(each, self.root) for each in x]


df = pd.read_csv("heartdisease.csv")
data = df.to_numpy()
tot = data.shape[0]
feature_types = [
    "continuous",  # age
    "discrete",  # sex
    "discrete",  # cp
    "continuous",  # trestbps
    "continuous",  # chol
    "discrete",  # fbs
    "discrete",  # restecg
    "continuous",  # thalach
    "discrete",  # exang
    "continuous",  # oldpeak
    "discrete",  # slope
    "discrete",  # ca
    "discrete",  # thal
]


for EPOCH in range(20):
    indices = np.random.permutation(tot)
    r = 0.2
    test_len = int(r * tot)
    test_x = data[indices[:test_len], :-1]
    train_x = data[indices[test_len:], :-1]
    test_y = data[indices[:test_len], -1]
    train_y = data[indices[test_len:], -1]

    tree = C45(min_samples_split=2, max_depth=10)
    tree.fit(train_x, train_y, feature_types) 
    predictions = tree.predict(test_x)
    res = sum(p == t for p, t in zip(predictions, test_y)) / test_len * 100
    print("EPOCH %d: %.2f%%" % (EPOCH + 1, res))
