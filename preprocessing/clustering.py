import pandas as pd
import numpy as np
from datacleaner import dataCleaner
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

churn = pd.read_csv("assets/BankChurners.csv")
X, y = dataCleaner(churn)
print(X.columns)

def kmClusterer(X):
    # Check out how many clusters are desirable
    # WCSS = "Within Cluster Sum of Squares"
    # k = number of clusters
    wcss = []
    # Choose a range between 1 and 10 clusters
    for k in range(1,11):
        kmeans = KMeans(n_clusters = k, init = "k-means++").fit(X)
        wcss.append(kmeans.inertia_)
    # plot the figure: where the "bend in the elbow" is, 
    # that number of clusters is desirable
    plt.figure(figsize=(12,6))    
    plt.grid()
    plt.plot(range(1,11), wcss, linewidth = 2, color = "red", marker = "8")
    plt.xlabel("K Value")
    plt.xticks(np.arange(1,11,1))
    plt.ylabel("WCSS")
    plt.savefig("assets/elbow_method.png")
    plt.clf()

    km = KMeans(n_clusters = 10).fit(X)
    score = km.score(X)
    print("score: ", score)

kmClusterer(X)