# flixstock
Flixstock Visual Search Assignment

2 files have been submitted.

1 is final.py which can be used to directly search for 10 matching outputs from a given input. Please keep the visual similarity folder in the root folder as it will seach for all files within this directory. 

The path of the file needs to be given as input and a matplot of 10 similar images along with the path of these images will be printed in output.

Steps to solve the problem.

Step 1 : Reshape the data for featur extraction.
Step 2 : Extract features and classify using VGG16 model.
Step 3 : Use Principal component analysis to reduce computational power required.
Step 4 : Use AffinityPropagation cluster with euclidian distances to get the cluster data for each file.(Kmeans not used as data is not labelled)
Step 5 : For input, output the images according to cluster.

For the final.py file, the post PCA array has been saved and used. The kagglecodefinalnotebook.ipynb file can be used to see the entire process from start to finish. 



