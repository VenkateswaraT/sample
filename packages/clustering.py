from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans;
from sklearn.cluster import KMeans;
import matplotlib.ticker as mtick
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import math
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import collections
import warnings
from sklearn.cluster import MiniBatchKMeans

# pd.options.display.max_rows = None
# pd.options.display.max_columns = None

warnings.filterwarnings('ignore')
from IPython.core.display import display, HTML
from sklearn.cluster import KMeans
display(HTML("<style>.container { width:100% !important; }</style>"))


def pca_for_vectors(sum_vectors, top_pc_roundNumber):
    pca = PCA()
    principalComponents = pca.fit_transform(sum_vectors)
    pca_df = pd.DataFrame(principalComponents)

    # Plot Cumulative Variance
    features = range(pca.n_components_)
    cumulative_variance = np.round(np.cumsum(pca.explained_variance_ratio_) * 100, decimals=4)
    plt.figure(figsize=(175 / 20, 100 / 20))
    plt.plot(cumulative_variance)
    # Find the exact percentile
    cv_list = pd.DataFrame(cumulative_variance.astype(int))
    top_cv = np.where(cv_list > 90)[0][0]
    top_pc_round = int(math.ceil(top_cv / float(top_pc_roundNumber))) * top_pc_roundNumber
    final_sum_vocab_docs_pc = pca_df.iloc[:, :top_pc_round]
    return final_sum_vocab_docs_pc

def clustering_on_shipevents_minbatch(ship_events, num_clusters, batch_size):
    mbk = MiniBatchKMeans(init='k-means++', n_clusters=num_clusters, batch_size=batch_size, max_no_improvement=10,
                          verbose=0)
    idx = mbk.fit_predict(ship_events)
    return mbk.cluster_centers_, idx, mbk.inertia_, mbk.labels_

def clustering_on_shipevents(ship_events, num_clusters):
    # Initalize a k-means object and use it to extract centroids
    kmeans_clustering = KMeans(n_clusters=num_clusters, init='k-means++')
    idx = kmeans_clustering.fit_predict(ship_events)

    return kmeans_clustering.cluster_centers_, idx, kmeans_clustering.inertia_


# ----------------------------------------------------------------------------------------------------------------------


def find_optimal_clusters_kmeans(data, max_k, algo, batch_size):
    iters = range(2, max_k + 1, 2)
    sse = []
    for k in iters:
        if algo == 'KMeans':
            centers, clusters, inertia = clustering_on_shipevents(data, k)
        elif algo == 'MinBatchKMeans':
            centers, clusters, inertia, lables = clustering_on_shipevents_minbatch(data, batch_size, k)
        sse.append(inertia)
    plt.figure(figsize=(15, 15))
    f, ax = plt.subplots(1, 1)
    ax.plot(iters, sse, marker='o')
    ax.set_xlabel('Cluster Centers')
    ax.set_xticks(list(range(0, 300, 10)))
    ax.set_ylabel('SSE')
    ax.grid(which='major', alpha=1)
    ax.set_title('SSE by Cluster Center Plot')
    f.set_size_inches(50, 15)
    plt.show()

def webChart(df, colNameOfCluster, clusterCloumnsRange, colorsList, autosize=True, width=1700, height=1000,
             margin=dict(l=30, r=30, b=30, t=30, pad=1),
             hoverlabel=dict(bgcolor="white", font_size=16, font_family="Rockwell")):
    fig = go.Figure()
    colorCount = 0
    fig.layout.polar.angularaxis.type = "category"
    for CID in df[colNameOfCluster]:
        tempdf = df[df[colNameOfCluster] == CID][list(range(clusterCloumnsRange[0], clusterCloumnsRange[1]))]
        fig.add_trace(go.Scatterpolar(marker_color=colorsList[colorCount],
                                      r=tempdf.iloc[0].values,
                                      theta=list(range(clusterCloumnsRange[0], clusterCloumnsRange[1])),
                                      name='Cluster - ' + str(CID) + ''
                                      ))
        colorCount = colorCount + 1
    fig.update_layout(autosize=autosize,
                      width=width,
                      height=height,
                      margin=margin,
                      hoverlabel=hoverlabel)
    return fig