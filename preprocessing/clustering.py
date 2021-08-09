import pandas as pd
from numpy import unique, where
from datacleaner import dataCleaner
from sklearn.cluster import AffinityPropagation, AgglomerativeClustering
import matplotlib.pyplot as plt

churn = pd.read_csv("assets/BankChurners.csv")
X, y = dataCleaner(churn)

def affinityPropagator(X, y):
    # define the model
    model = AffinityPropagation(damping=0.9, random_state = None)
    # fit the model
    model.fit(X, y)
    # assign a cluster to each example
    yhat = model.predict(X)
    # retrieve unique clusters
    clusters = unique(yhat)
    # create scatter plot for samples from each cluster
    for cluster in clusters:
        # get row indexes for samples with this cluster
        row_ix = where(yhat == cluster)
        # create scatter of these samples
        plt.scatter(X[row_ix, 0], X[row_ix, 1])
    # show the plot
    plt.legend()
    plt.show()

# def agglomerativeClusterer(X, y):
#     # define the model
#     model = AgglomerativeClustering(n_clusters=2)
#     # fit model and predict clusters
#     yhat = model.fit_predict(X, y)
#     # retrieve unique clusters
#     clusters = unique(yhat)
#     # create scatter plot for samples from each cluster
#     for cluster in clusters:
#         # get row indexes for samples with this cluster
#         row_ix = where(yhat == cluster)
#         # create scatter of these samples
#         plt.scatter(X[row_ix, 0], X[row_ix, 1])
#     # show the plot
#     plt.show()