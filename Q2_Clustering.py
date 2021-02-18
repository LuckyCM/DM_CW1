# 2.Clustering
import pandas as pd
import numpy as np
import sklearn
from matplotlib.font_manager import FontProperties
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Define the resource of data
DATA_DIR  = r'C:\Users\Administrator\PycharmProjects\DM_CW1\data'
DATA_FILE = r'\wholesale_customers.csv'

# Load data and pretreatment
rawdata0 = pd.read_csv(DATA_DIR + DATA_FILE)
rawdata = rawdata0.drop(['Channel'], axis=1)
rawdata = rawdata.drop(['Region'], axis=1)

######################################################
# 2.1 Means value, min value and max value
######################################################
mean_data = rawdata.mean()
min_data = rawdata.min()
max_data = rawdata.max()
print(mean_data, min_data, max_data)
######################################################
# 2.2 Scatter plot use K-means algorithm
######################################################
k_means = KMeans(n_clusters=3, random_state=0)  # n_cluster=3 means that 3 centroid
k_means.fit(rawdata)
result_hat = k_means.predict(rawdata)

data_size = len(np.transpose(rawdata))-1
fig, axis = plt.subplots(data_size, data_size, figsize=(25, 25))
for x in range(data_size):
    for y in range(x + 1, data_size+1):    # make sure the x,y axis are different
        # reference to https://blog.csdn.net/weixin_39940253/article/details/110865498
        axis[x, y-1].scatter(rawdata.iloc[:, x], rawdata.iloc[:, y], c=result_hat)  # c=result_hat, because kmeans automately give the 0-2 figure to result_hat
        axis[x, y-1].set_xlabel(rawdata.columns.values[x])
        axis[x, y-1].set_ylabel(rawdata.columns.values[y])
fig.show()

# # one by one
# k_means = KMeans(n_clusters=3, random_state=0)  # n_cluster=3 means that 3 centroid
# k_means.fit(rawdata)
# result_hat = k_means.predict(rawdata)
# # Plot the data
# for i in range(5):
#     for j in range(i+1, 6):
#         plt.scatter(rawdata.values[:, i], rawdata.values[:, j], c=result_hat, s=5)
#         plt.xlabel(rawdata.columns[i])
#         plt.ylabel(rawdata.columns[j])
#         plt.show()

######################################################
# 2.3 Compute BC, WC and BC/WC in K-means which k={3,5,10}
######################################################
# print(sklearn.metrics.calinski_harabasz_score(rawdata,result_hat))
K = [3, 5, 10]
WC = []
BC = []
BC_WC = 0
for k in K:
    kmeans = KMeans( n_clusters=k )
    kmeans.fit( rawdata )
    WC.append(
        sum(
            np.min(
                cdist( rawdata, kmeans.cluster_centers_, metric='euclidean' ), axis=1 )
            **2 )
    )
    BC.append(
        sum(
            sum(
                cdist( kmeans.cluster_centers_, kmeans.cluster_centers_, metric='euclidean' )
        ** 2 )
        )
    )
    # BC_WC = BC[k-1] / WC[k-1]
BC_WC = np.divide(BC, WC)
print("WC: %", WC)
print("BC: %", BC )
print("BC/WC:", BC_WC)


# ######################################################
# # * elbow to look for the best K value (not include in the coursework requirement)
# # Reference from https://my.oschina.net/u/4418231/blog/3506187
# ######################################################
# K = range( 1, 20 )
# mean_distortions = []
# for k in K:
#     kmeans = KMeans( n_clusters=k )
#     kmeans.fit( rawdata )
#     mean_distortions.append(
#         sum(
#             np.min(
#                 cdist( rawdata, kmeans.cluster_centers_, metric='euclidean' ), axis=1 ) )
#         / rawdata.shape[0] )
# plt.plot( K, mean_distortions, 'bx-' )
# plt.xlabel( 'k' )
# font = FontProperties( fname=r'c:\windows\fonts\msyh.ttc', size=15 )
# plt.ylabel('WC_mean', fontproperties=font )
# plt.title('K-value', fontproperties=font )
# plt.show()