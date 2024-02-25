import sklearn
from matplotlib import pyplot as plt
from metrics import metrics
from numpy import ndarray
import os
import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


class KmeansLabelModel(object):
    def __init__(self, pretrained_modal_path='', class_count=2):
        self.pretrained_modal_path = pretrained_modal_path
        if os.path.isfile(self.pretrained_modal_path):
            self.modal = joblib.load(self.pretrained_modal_path)
        else:
            self.modal = sklearn.cluster.KMeans(n_clusters=class_count, init='k-means++', max_iter=300, n_init=10, random_state=0)

    def fit(self, data: ndarray):
        self.modal.fit(data)

    def predict(self, data: ndarray) -> str:
        return self.modal.predict(data)

    def mark_accuracy(self, labels: ndarray):
        labels_count = len(labels)

        print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, self.modal.labels_[:labels_count]))
        print("Completeness: %0.3f" % metrics.completeness_score(labels, self.modal.labels_[:labels_count]))
        print("V-measure: %0.3f" % metrics.v_measure_score(labels, self.modal.labels_[:labels_count]))

    def show_graphic(self, data: ndarray):
        y_kmeans = self.modal.predict(data)
        plt.scatter(data[:, 0], data[:, 1], c=y_kmeans, s=50, cmap='viridis')

        centers = self.modal.cluster_centers_
        plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
        plt.show()