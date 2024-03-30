import cv2
import numpy as np
from sklearn.cluster import KMeans
import skeleton as sk
import matplotlib.pyplot as plt

# Load the image

image = cv2.imread('example_image.jpg')
image1 = cv2.imread('example_image.jpg')

# Reshape the image into a 2D array of pixels
pixels = image.reshape((-1, 3))  # -1 means automatically calculate the number of rows

# Convert the pixel values to float32
pixels = np.float32(pixels)

# Define the number of clusters (k)
k = 17  # You can adjust this value based on your requirement

# Apply KMeans clustering
kmeans = KMeans(n_clusters=k)
kmeans.fit(pixels)

# Get the cluster centers and labels
centers = np.uint8(kmeans.cluster_centers_)  # Convert back to uint8
labels = kmeans.labels_

# Assign each pixel the color of its corresponding cluster center
segmented_image = centers[labels]

# Reshape the segmented image to its original dimensions
segmented_image = segmented_image.reshape(image.shape)

# Display the original and segmented images
plt.imshow(cv2.cvtColor(segmented_image.astype('uint8'), cv2.COLOR_BGR2RGB))
plt.title('Colored Segmented Image (Coreset)')
plt.axis('off')
plt.show()
