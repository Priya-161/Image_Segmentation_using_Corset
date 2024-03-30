"""
***********
IIIT Delhi License
Copyright (c) 2023 Supratim Shit
***********
"""

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import wkpp as wkpp 
import numpy as np
import cv2
import matplotlib.pyplot as plt


# Load image using OpenCV
image = cv2.imread('example_image.jpg')
data = image.reshape(-1, image.shape[-1])                            # Fetch kddcup99 
            # Preprocess
data = data.astype(float)                                # Preprocess
data = StandardScaler().fit_transform(data)                # Preprocess

n = np.size(data,0)                                        # Number of points in the dataset
d = np.size(data,1)                                        # Number of dimension/features in the dataset.
k = 17                                  # Number of clusters (say k = 17)
Sample_size = 100

# D2-Sampling function
def D2(data, k):
    # Initialize the set of cluster centers
    B = []

    # Sample the first cluster center uniformly at random from X
    first_center_idx = np.random.choice(len(data))
    B.append(data[first_center_idx])

    # Calculate the squared distances to the first center
    distances_to_first_center = np.linalg.norm(data - data[first_center_idx], axis=1) ** 2

    for i in range(1, k):
        # Sample the next center with probability proportional to squared distance
        probabilities = distances_to_first_center / np.sum(distances_to_first_center)
        next_center_idx = np.random.choice(len(data), p=probabilities)
        B.append(data[next_center_idx])

        # Update the distances to the new center
        new_distances = np.linalg.norm(data - data[next_center_idx], axis=1) ** 2
        distances_to_first_center = np.minimum(distances_to_first_center, new_distances)

    # Calculate the center using the generated set of cluster centers
    center = np.mean(B, axis=0)
    
    return center

# Coreset construction function
def Sampling(data, k, centers, Sample_size):
    # Initialize coreset and weights
    coreset = np.empty((Sample_size, data.shape[1]))
    weights = np.empty((Sample_size, 1))

    N = data.shape[0]

    # compute proposal distribution
    q = np.linalg.norm(data - centers, axis=1)**2
    sum_q = np.sum(q)
    q = 0.5 * (q/sum_q + 1.0/N)

    # get sample and fill coreset
    samples = np.random.choice(N, Sample_size, p=q)
    coreset = data[samples]
    weights = 1.0 / (q[samples] * Sample_size)

    return coreset, weights

# Call D2-Sampling (D2())
centers = D2(data, k)

# Call coreset construction algorithm (Sampling())
coreset, weight = Sampling(data, k, centers, Sample_size)

#---Running KMean Clustering---#
fkmeans = KMeans(n_clusters=k,init='k-means++')
fkmeans.fit_predict(data)

#----Practical Coresets performance----#     
Coreset_centers, _ = wkpp.kmeans_plusplus_w(coreset, k, w=weight, n_local_trials=100)                        # Run weighted kMeans++ on coreset points
wt_kmeansclus = KMeans(n_clusters=k, init=Coreset_centers, max_iter=10).fit(coreset,sample_weight = weight)    # Run weighted KMeans on the coreset, using the inital centers from the above line.
Coreset_centers = wt_kmeansclus.cluster_centers_                                                            # Compute cluster centers
coreset_cost = np.sum(np.min(cdist(data,Coreset_centers)**2,axis=1))                                        # Compute clustering cost from the above centers
relative_error_practicalCoreset = abs(coreset_cost - fkmeans.inertia_)/fkmeans.inertia_                        # Computing relative error from practical coreset, here fkmeans.inertia_ is the optimal cost on the complete data.

#-----Uniform Sampling based Coreset-----#
tmp = np.random.choice(range(n),size=Sample_size,replace=False)        
sample = data[tmp][:]                                                                                        # Uniform sampling
sweight = n*np.ones(Sample_size)/Sample_size                                                                 # Maintain appropriate weight
sweight = sweight/np.sum(sweight)                                                                            # Normalize weight to define a distribution

#-----Uniform Sampling based Coreset performance-----#     
wt_kmeansclus = KMeans(n_clusters=k, init='k-means++', max_iter=10).fit(sample,sample_weight = sweight)        # Run KMeans on the random coreset
Uniform_centers = wt_kmeansclus.cluster_centers_                                                            # Compute cluster centers
uniform_cost = np.sum(np.min(cdist(data,Uniform_centers)**2,axis=1))                                        # Compute clustering cost from the above centers
relative_error_uniformCoreset = abs(uniform_cost - fkmeans.inertia_)/fkmeans.inertia_                        # Computing relative error from random coreset, here fkmeans.inertia_ is the optimal cost on the full data.
    
Coreset_centers, _ = wkpp.kmeans_plusplus_w(coreset, k, w=weight, n_local_trials=100)  # Run weighted kMeans++ on coreset points



# Use the cluster centers obtained from the coreset to assign each pixel in the original image to the nearest cluster center
labels = np.argmin(cdist(data, Coreset_centers), axis=1)

# Reshape the labels array to match the shape of the original image
labels_reshaped = labels.reshape(image.shape[0], image.shape[1])

# Assign original colors from the image to different clusters
cluster_colors = []
for i in range(k):
    cluster_pixels = image[labels_reshaped == i]
    average_color = np.mean(cluster_pixels, axis=0)
    cluster_colors.append(average_color)

cluster_colors = np.array(cluster_colors).astype(int)

# Create a colored segmented image
colored_segmented_image = np.zeros_like(image)
for i in range(k):
    colored_segmented_image[labels_reshaped == i] = cluster_colors[i]

# Display the colored segmented image
'''plt.imshow(cv2.cvtColor(colored_segmented_image.astype('uint8'), cv2.COLOR_BGR2RGB))
plt.title('Colored Segmented Image (Coreset)')
plt.axis('off')
plt.show()'''



print("Relative error from Practical Coreset is", relative_error_practicalCoreset)
print("Relative error from Uniformly random Coreset is", relative_error_uniformCoreset)