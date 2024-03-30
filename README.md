# Image_Segmentation_using_Corset
The provided Python files serve as a framework for implementing coreset construction algorithms and evaluating their performance: pip install numpy opencv-python scikit-learn scipy matplotlib streamlit

Skeleton.py:
Provides a structure for implementing D2-sampling and coreset construction algorithms. Includes data preprocessing, parameter initialization, and performance evaluation steps.

Uniform_Sampling.py:
Offers an alternative approach for constructing a coreset using uniform sampling. Randomly selects points from the dataset, assigns uniform weights, and applies KMeans clustering for evaluation. 

wkpp.py:
Contains the implementation of the kmeans_plusplus_w function, enabling weighted k-means++ initialization for coreset construction. 

Next Steps:
1.Implement algorithms in Skeleton.py.
2.Test implemented algorithms with sample datasets. 
3.Evaluate algorithm performance against KMeans clustering results.

Advanced Task (Bonus): Explore image segmentation using adapted coreset construction algorithms. Modify coreset construction for image data. Implement image segmentation using constructed coresets and compare results with original image segmentation
