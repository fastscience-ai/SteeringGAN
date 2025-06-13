from sklearn.decomposition import PCA

class Reduction:
    def __init__(self, n_components: int):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)

    def fit(self, X):
        self.pca.fit(X)

    def transform(self, X):
        return self.pca.transform(X)