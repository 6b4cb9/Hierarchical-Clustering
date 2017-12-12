=======================
Hierarchical-Clustering
=======================


It is a implementation of aglomerative hierarchical clustering.
We use two metrics and three linkage methods .


Description
===========

The metrics that we use are euclides=l2 and manhattan=l1.
The linkage methods that we use are ward, max and average.
Ward mathod base on special funcion and can be comuped recursively,
for more information https://en.wikipedia.org/wiki/Ward%27s_method
Max linkage uses the maximum distances between all observations of the two sets.
Average linkage uses the average of the distances of each observation of the two sets.
Main class is HierarchicalClustering and this class has fit method which fill with data
points and performs clustering with number of clusters equal to n_clusters as a target.
The method of HierarchicalClustering class that is responsible for finding minimum of distance and then
merge two clusters is a _step method.
As a input data points you can give a n-dimensional numpy list and also n-dimensional normal list.
Class Cluster is interface for ClusterWard,ClusterAverage and ClusterMax.





Note
====

This project has been set up using PyScaffold 2.5.8. For details and usage
information on PyScaffold see http://pyscaffold.readthedocs.org/.
