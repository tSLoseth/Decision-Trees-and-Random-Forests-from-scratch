import numpy as np
from typing import Self

def count(y: np.ndarray) -> np.ndarray:
    """
    Count unique values in y and return the proportions of each class sorted by label in ascending order.
    Example:
        count(np.array([3, 0, 0, 1, 1, 1, 2, 2, 2, 2])) -> np.array([0.2, 0.3, 0.4, 0.1])
    """
    if len(y) == 0:
        return np.array([])
    return np.bincount(y) / len(y)


def gini_index(y: np.ndarray) -> float:
    """
    Return the Gini Index of a given NumPy array y.
    The forumla for the Gini Index is 1 - sum(probs^2), where probs are the proportions of each class in y.
    Example:
        gini_index(np.array([1, 1, 2, 2, 3, 3, 4, 4])) -> 0.75
    """
    probs = count(y)
    return 1 - np.sum(probs**2)


def entropy(y: np.ndarray) -> float:
    """
    Return the entropy of a given NumPy array y.
    """
    probs = count(y)
    if probs.size == 0:
        return 0.0
    probs = probs[probs > 0]  # log2(0) undefined
    return float(-np.sum(probs * np.log2(probs)))


def split(x: np.ndarray, value: float) -> np.ndarray:
    """
    Return a boolean mask for the elements of x satisfying x <= value.
    Example:
        split(np.array([1, 2, 3, 4, 5, 2]), 3) -> np.array([True, True, True, False, False, True])
    """
    x = np.asarray(x, dtype=float)
    return x <= value


def most_common(y: np.ndarray) -> int:
    """
    Return the most common element in y.
    Example:
        most_common(np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])) -> 4
    """
    if len(y) == 0:
        raise ValueError("Empty array recieved")
    y = np.asarray(y)
    count_vals = np.bincount(y)
    return int(np.argmax(count_vals))


class Node:
    """
    A class to represent a node in a decision tree.
    If value != None, then it is a leaf node and predicts that value, otherwise it is an internal node (or root).
    """

    def __init__(
        self,
        feature_idx: int = 0,
        threshold: float = 0.0,
        left: Self | None = None,
        right: Self | None = None,
        value: int | None = None,
    ) -> None:
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self) -> bool:
        return self.value is not None


class DecisionTree:
    def __init__(
        self,
        max_depth: int | None = None,
        criterion: str = "entropy",
        max_features:str = None,
        feature_names=None,
    ) -> None:
        self.root = None
        self.criterion = criterion
        self.max_depth = max_depth
        self.max_features = max_features
        self.feature_names = feature_names

        if criterion not in ["entropy", "gini"]: raise ValueError("Not correct criterion",str(criterion))
        
        if max_features not in [None,"sqrt","log2"]: raise ValueError("not correct max feature") 

    def fit(self, X: np.ndarray, y: np.ndarray):
        #np.random.seed(42)
        self.root = self.build_tree(X, y)

    def build_tree(self, X, y, depth: int = 0):
        node = Node()

        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(int)

        if (#stop creterias. If one of the criterais we return the node as a leaf node
            len(y) == 1 #only one datapoint, we return a leaf node
            or np.all(y == y[0]) #Node is pure.
            or (self.max_depth is not None and depth >= self.max_depth) #Max depth is reached
            or (len(np.unique(X, axis=0)) == 1) #all feturerows are identical.
        ):
            node.value = most_common(y)
            return node
        else:
            node.feature_idx, node.threshold = self.find_best_split_feature(X, y)
            mask = X[:, node.feature_idx] <= node.threshold 
            if (
                self.calculate_information_gain(y, y[mask], y[~mask]) <= 0
                or min(len(y[mask]), len(y[~mask])) == 0
            ):
                node.value = most_common(y)
                return node

        node.left = self.build_tree(X[mask], y[mask], depth + 1)
        node.right = self.build_tree(X[~mask], y[~mask], depth + 1)

        return node



    def feature_subset(self,n_features):                        
        if self.max_features is None:
            k=n_features
        elif self.max_features == "sqrt":
            k = max(1,int(np.floor(np.sqrt(n_features))))
        else: #log2
            k = max(1,int(np.floor(np.log2(n_features))))
        return np.random.choice(n_features,size=k,replace=False)

    def find_best_split_feature(self, X, y):
        best_info_gain = -np.inf
        threshold = 0.0
        best_feature_idx = None
        

        candidates = self.feature_subset(X.shape[1])
        
        for feature_idx in candidates: 
            median = np.median(X[:, feature_idx])
            mask = split(X[:, feature_idx], median)
            if mask.sum() == 0 or (~mask).sum() == 0:       
                continue
            info_gain = self.calculate_information_gain(y, y[mask], y[~mask])
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                threshold = median
                best_feature_idx = feature_idx
        
        if best_feature_idx is None:
            return 0, float(np.median(X[:, 0]))  

        return best_feature_idx, threshold
    


    def calculate_information_gain(self, y, left_child, right_child):
        n_left = len(left_child)
        n_right = len(right_child)
        n = n_left + n_right

        impurity_parent = self.calculate_impurity(y)
        weighted_child_imp = (
            n_left / n * self.calculate_impurity(left_child)
            + n_right / n * self.calculate_impurity(right_child)
        )

        return impurity_parent - weighted_child_imp

    def calculate_impurity(self, y):
        if self.criterion == "gini":
            return gini_index(y)
        else:
            return entropy(y)

    def print_tree(self, node: Node | None = None) -> None:
        if node is None:
            node = self.root
        if node is None:
            print("(tomt tre)")
            return

        def rec(n: Node | None, depth: int) -> None:
            indent = "  " * depth
            if n is None:
                print(indent + "∅")
                return
            if n.is_leaf():
                print(indent + f"Leaf: {n.value}")
                return
            print(indent + f"X{n.feature_idx} <= {n.threshold:.6g}")
            rec(n.left, depth + 1)
            rec(n.right, depth + 1)

        rec(node, 0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        #This method loops through every row in x, and traverses the decision tree until it reaches a leaf node
        #(By evaluating the feature and threshold in the decision nodes)
        #When it reaches a leaf node, it stores Node.value as the predicted label, and returns an array of all predictions of X
        if self.root is None:
            raise ValueError("Tree not built")
        predictions = []
        for row in X:
            node = self.root
            while node.value is None:
                if row[node.feature_idx] <= node.threshold:
                    node = node.left
                else:
                    node = node.right
            predictions.append(node.value)
        return np.array(predictions)


if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    seed = 42
    np.random.seed(seed)

    X, y = make_classification(
        n_samples=100, n_features=10, random_state=seed, n_classes=2
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=seed, shuffle=True
    )

    rf = DecisionTree(
        max_depth=None,
        criterion="entropy",
        max_features=None
    )
    rf.fit(X_train, y_train)

    print(f"Training accuracy: {accuracy_score(y_train, rf.predict(X_train))}")
    print(f"Validation accuracy: {accuracy_score(y_val, rf.predict(X_val))}")
    rf.print_tree()

