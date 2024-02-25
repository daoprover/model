from numpy import ndarray
from sklearn.neighbors import NearestNeighbors


class KnnValidation:
    def __init__(self, count = 2):
        self.knn = NearestNeighbors(n_neighbors = count, algorithm = 'ball_tree')

    def fit(self, data: ndarray):
        self.knn.fit(data)

    def validate(self, data: ndarray, labels: ndarray) -> tuple[int, int]:
        tp = 0
        fn = 0

        knn_clusters = self.knn.kneighbors(data, return_distance = False)

        for item in knn_clusters:  # item contain array of transactions
            item1 = labels[item[0]]
            item2 = labels[item[1]]

            if item1 == item2:
                tp += 1
            else:
                fn += 1

        return tp, fn


if __name__ == "__main__":
    X_train = ndarray((50, 2))

    knn = KnnValidation(count = 3)
    knn.fit(X_train)
