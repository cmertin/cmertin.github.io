
# Efficient Neural Networks

This is a project that I'm working on with Hari Sundar. It is currently under development. 

The point of this project is to allow neural networks to essentially learn their own structure such that as it learns about the data, it also learns about how the structure is progressing. This will bring about things such as faster convergence and a faster learning rate, with hopefully optimized storage costs as well. It is essentially a mix between a dense NN and a sparse NN.

As this is my ongoing project, this site will only be updated after I finish significant parts of the work. This is to keep other people from claiming this work as their own before I’m able to finish implementing everything.

The final version will be uploaded to github as well so it can be easily forked and modified.


This project is implemented with a TensorFlow backend. The goal of this project is to reduce the training and prediction time with minimal loss to accuracy.

The project consists of using Hierarchial Matrices as the "hidden layers" which are usually the largest and most computationally demanding in the network. With the use of Hierarchial Matrices, instead of the matrix-matrix multiplcation being $$\mathcal{O}(n^{3})$$, it is reduced down to $$\mathcal{O}(n^)$$ by refomulating "low rank" portions of the matrix.

The use of Low Rank approximations of the weight matrices have been used to **test** the performance with decent results. On the current data set being tested, it was able to reduce the training time from 10 minutes on an AWS instance to just under 2 minutes with only a 1% loss in accuracy.

Hierarchial Matrices should be able to improve this performance such that they will keep the same training time as the low rank implementation (or even improve it), while making the accuracy of the system perform better than the Low Rank implementation.

The beauty of this implementation is that instead of "solving" for the low rank or the Hierarchial implementations of the weight matrices, we have the Neural Network "learn" about its own structure such that an SVD computation isn't required. This is how the computational costs of the SVD are avoided and make the implementation beneficial for large scale neural networks.
