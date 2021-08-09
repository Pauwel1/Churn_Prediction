import pandas as pd
from numpy import unique, where
from datacleaner import dataCleaner
from sklearn.cluster import AffinityPropagation, AgglomerativeClustering
import matplotlib.pyplot as plt

churn = pd.read_csv("assets/BankChurners.csv")
X, y = dataCleaner(churn)

def affinityPropagator(X):
    # define and fit the model
    model = AffinityPropagation(max_iter = 250, damping = 0.9, random_state = None).fit(X)
    cluster_centers_indices = model.cluster_centers_indices_
    n_clusters_ = len(cluster_centers_indices)
    # assign a cluster to each example
    yhat = model.predict(X)
    # retrieve unique clusters
    clusters = unique(yhat)

    # create scatter plot for samples from each cluster
    for cluster in clusters:
        # get row indexes for samples with this cluster
        row_ix = where(yhat == cluster)
        # create scatter of these samples
        plt.scatter(X.loc[row_ix, 0], X.loc[row_ix, 1])

    # plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c = yhat)

    # show the plot
    plt.title(f'Estimated number of clusters = {n_clusters_}')
    plt.legend()
    plt.show()

aff = affinityPropagator(X)

# def agglomerativeClusterer(X):
#     # define the model
#     model = AgglomerativeClustering(n_clusters = 2)
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