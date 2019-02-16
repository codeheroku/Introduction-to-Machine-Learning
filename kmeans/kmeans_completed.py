from copy import deepcopy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
plt.rcParams['figure.figsize'] = (16, 9)
plt.ion()

#####Helper Functions#####
def show_plot(C,X):
    colors = ['r', 'g', 'b', 'y', 'c', 'r']
    for i in range(k):

            points = np.array([X[j] for j in range(len(X)) if clusters[j] == i])
            plt.scatter(points[:, 0], points[:, 1], s=20, c=colors[i], alpha=0.2)
            plt.scatter(C[i,0], C[i,1], marker='D', s=200, c='black',edgecolor='w')

    plt.draw()
    plt.pause(5)
    plt.clf()

def get_random_centeroids(X,k):
    C_x = np.random.randint(0, np.max(X)-20, size=k) 
    C_y = np.random.randint(0, np.max(X)-20, size=k)
    C = np.array(list(zip(C_x, C_y)))
    return C,C_x,C_y    

# Euclidean Distance Caculator
def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)    

#######Begin Here###############################
## Step1: Read Dataset 
data = pd.read_csv('dataset.csv')


## Step2: Get values f1,f2 and plotting it
f1 = data['V1'].values
f2 = data['V2'].values
X = list(zip(f1, f2))

## Step3: Plot data and centroids

##Init K Random Centroids
k = 4
C,C_x,C_y = get_random_centeroids(X,k)

# plt.scatter(f1, f2, c='black', s=7)
# plt.scatter(C_x, C_y, marker='*', s=200, c='r')
# plt.show()

# To store the value of centroids when it updates
C_old = np.zeros(C.shape)
# Cluster Lables(0, 1, 2)
clusters = np.zeros(len(X))

change_in_c = dist(C, C_old, None)

erros = []

while change_in_c != 0:    
    ##Part1: Cluster Assignment. Assign points to closest cluster
    for i in range(len(X)):
        distances_x_centroids = dist(X[i], C)  
        clusters[i] = np.argmin(distances_x_centroids)
    # Storing the old centroid values
    C_old = deepcopy(C)
    
    #Find the new centroids by taking the average value
    #Part2: Cluster Movement
    for i in range(k):
        points = [X[j] for j in range(len(X)) if clusters[j] == i]
        C[i] = np.mean(points, axis=0)

    
    show_plot(C,X)
          
    change_in_c = dist(C, C_old, None)

plt.show()    


