
# Efficient Neural Networks

This is a project that I'm working on with Hari Sundar. It is currently under development. 


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
