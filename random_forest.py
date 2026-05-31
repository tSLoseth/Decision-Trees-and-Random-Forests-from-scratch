import numpy as np
import decision_tree


class RandomForest:
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 5,
        criterion: str = "entropy",
        max_features = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.criterion = criterion
        self.max_features = max_features
        self.trees = []


    def fit(self, X: np.ndarray, y: np.ndarray):
        #This method creates n_estimators decition trees, and fits each tree to the training data

        X = np.asarray(X,dtype=float)
        y = np.asarray(y).astype(int)
        n = X.shape[0]

        self.trees = []        
        
        for i in range(self.n_estimators):
            
            idx = np.random.choice(n,size=n,replace=True)
            Xi,yi = X[idx],y[idx]
            tree = decision_tree.DecisionTree(max_depth=self.max_depth,criterion=self.criterion,max_features=self.max_features,feature_names=None)
            tree.fit(Xi,yi)
            self.trees.append(tree)
        return self



    def predict(self, X: np.ndarray) -> np.ndarray:
        #This method loops through all decition trees stored in self.trees, and for every tree, it predicts the labels of X. 
        #The method returns the most commmonly predicted label for each row in X, by majority vote
        if not self.trees:
            raise ValueError("Forest not fitted")
        elif X.shape[0] == 0:
            raise ValueError("X is empty")

    # Collect per-tree predictions: shape -> (n_trees, n_samples)
        
        all_preds = np.vstack([tree.predict(X) for tree in self.trees])

        # Majority vote per sample

        max_label = int(all_preds.max())
        votes = np.apply_along_axis(lambda col: np.bincount(col,minlength=max_label+1).argmax(),axis=0,arr=all_preds)

        return votes.astype(int)
    

    def permutation_importance(self,X,y,metric,n_repeats,seed):
        #
        np.random.seed(seed)
        pred = self.predict(X)
        baseline = metric(y_pred=pred,y_true=y)
        results = []
        for feature in range(X.shape[1]):
            scores = []
            for i in range(n_repeats):
                Xi = X.copy()
                Xi[:, feature] = np.random.permutation(Xi[:, feature])
                scores.append(baseline-metric(y_pred=self.predict(Xi),y_true=y))
            
            results.append(np.average(np.array(scores)))
        return results
        
        

        
if __name__ == "__main__":
    # Test the RandomForest class on a synthetic dataset
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

    rf = RandomForest(
        n_estimators=25, max_depth=None, criterion="entropy", max_features="sqrt"
    )
    rf.fit(X_train, y_train)

    print(f"Training accuracy: {accuracy_score(y_train, rf.predict(X_train))}")
    print(f"Validation accuracy: {accuracy_score(y_val, rf.predict(X_val))}")
