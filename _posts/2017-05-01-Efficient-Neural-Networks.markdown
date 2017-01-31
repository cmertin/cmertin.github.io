
# Efficient Neural Networks

This is a project that I'm working on with Hari Sundar. It is currently under development. 

The point of this project is to allow neural networks to essentially *learn* their own structure such that as it *learns* about the data, it also *learns* about how the structure is progressing. This will bring about things such as faster convergence and a faster learning rate, with hopefully optimized storage costs as well. It is essentially a mix between a *dense NN* and a *sparse NN.*

As this is my ongoing project, this site will only be updated after I finish significant parts of the work. This is to keep other people from claiming this work as their own before I'm able to finish implementing everything.


```python
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
tf.set_random_seed(0)
```

We need to download the MNIST data


```python
mnist = read_data_sets("data", one_hot=True, reshape=False, validation_size=0)
```

    Extracting data/train-images-idx3-ubyte.gz
    Extracting data/train-labels-idx1-ubyte.gz
    Extracting data/t10k-images-idx3-ubyte.gz
    Extracting data/t10k-labels-idx1-ubyte.gz


These are the place holders used for the first layer


```python
# X: 28x28 grayscale images
# First dimension (None) will index the images in the mini-batch
X = tf.placeholder(tf.float32, [None, 28, 28, 1])
# Holds the correct answers
Y_ = tf.placeholder(tf.float32, [None, 10])
# Learning rate
lr = tf.placeholder(tf.float32)
```


```python

```
