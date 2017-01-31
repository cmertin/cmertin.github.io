
# Efficient Neural Networks

This is a project that I'm working on with Hari Sundar. It is currently under development. 

The point of this project is to allow neural networks to essentially learn their own structure such that as it learns about the data, it also learns about how the structure is progressing. This will bring about things such as faster convergence and a faster learning rate, with hopefully optimized storage costs as well. It is essentially a mix between a dense NN and a sparse NN.

As this is my ongoing project, this site will only be updated after I finish significant parts of the work. This is to keep other people from claiming this work as their own before I’m able to finish implementing everything.

The final version will be uploaded to github as well so it can be easily forked and modified.


```python
from __future__ import print_function, division
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import math
import matplotlib.pyplot as plt
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
# Dropout keep percentage
pkeep = tf.placeholder(tf.float32)
```

We can define the number of neurons per layer as


```python
L = 200
M = 100
N = 60
O = 30
n_ = 28*28
```

Next we can initialize the weights with small random values between `-0.2` and `+0.2`


```python
W1 = tf.Variable(tf.truncated_normal([n_, L], stddev=0.1))
B1 = tf.Variable(tf.ones([L])/10)
W2 = tf.Variable(tf.truncated_normal([L, M], stddev=0.1))
B2 = tf.Variable(tf.ones([M])/10)
W3 = tf.Variable(tf.truncated_normal([M, N], stddev=0.1))
B3 = tf.Variable(tf.ones([N])/10)
W4 = tf.Variable(tf.truncated_normal([N, O], stddev=0.1))
B4 = tf.Variable(tf.ones([O])/10)
W5 = tf.Variable(tf.truncated_normal([O, 10], stddev=0.1))
B5 = tf.Variable(tf.zeros([10]))
```

Finally, we need to build our model. Each $Y_{i}$ is a layer of the model, with `Y` being the final predictor. `XX` is the "flattened" MNIST data. The `dropout` is also added to prevent overfitting of the training data.


```python
XX = tf.reshape(X, [-1, n_])
Y1 = tf.nn.relu(tf.matmul(XX, W1) + B1)
Y1d = tf.nn.dropout(Y1, pkeep)
Y2 = tf.nn.relu(tf.matmul(Y1d, W2) + B2)
Y2d = tf.nn.dropout(Y2, pkeep)
Y3 = tf.nn.relu(tf.matmul(Y2d, W3) + B3)
Y3d = tf.nn.dropout(Y3, pkeep)
Y4 = tf.nn.relu(tf.matmul(Y3d, W4) + B4)
Y4d = tf.nn.dropout(Y4, pkeep)
Ylogits = tf.matmul(Y4d, W5) + B5
Y = tf.nn.softmax(Ylogits)
```

Finally, to verify our prediction rate, we can implement the `cross_entropy` function which will be the *log-loss function* which is defined as $-\sum_{i}\tilde{Y_{i}} \log(Y_{i})$, where $\tilde{Y_{i}}$ is the *true value* and $Y_{i}$ is the predicted value of that result.


```python
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
cross_entropy = tf.reduce_mean(cross_entropy)*100
```

We can also get the accuracy of the trained model, where `0` is worst and `1` is best


```python
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
```

Finally, we can define the training step and the learning rate


```python
train_set = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
```

Now we can initialize our model


```python
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
```

We can define our training function such that it will train over a certain number of batches


```python
def TrainingStep(i, update_test_data=False, update_train_data=False, p_keep=0.75, num_batches=100, max_lr=0.003, min_lr=0.0001):
    # Pulls out the next number of batches
    batch_X, batch_Y = mnist.train.next_batch(num_batches)
    train_vals = []
    test_vals = []
    
    decay_speed = 2000.0
    learning_rate = min_lr + (max_lr - min_lr) * math.exp(-i/decay_speed)
    
    # Compute training values
    if update_train_data:
        a, c = sess.run([accuracy, cross_entropy], feed_dict={X: batch_X, Y_: batch_Y, pkeep: 1.0})
        train_vals.extend([i, a, c])
        if i % 1000 == 0:
            print("i = " + str(i) + "\t Training Accuracy: " + str(a) + "\t Loss: " + str(c))
        
    # Compute test values
    if update_test_data:
        at, ct = sess.run([accuracy, cross_entropy], feed_dict={X: mnist.test.images, Y_: mnist.test.labels, pkeep: 1.0})
        test_vals.extend([i, at, ct])
        if i % 1000 == 0:
            print("i = " + str(i) + "\t Epoch " + str(i * num_batches/mnist.train.images.shape[0] + 1) + "\t test accuracy: " + str(at) + "\t test loss: " + str(ct))
        
    # Backpropogation training step
    sess.run(train_set, {X: batch_X, Y_: batch_Y, lr: learning_rate, pkeep: p_keep})
    
    return train_vals, test_vals
        
```

Now that we have our training set function, we just need to implement the number of iterations to call this function. This can be done with a simple for loop as shown below


```python
def TrainData(itr=10000, train_freq=10, test_freq=10, p_keep=0.75):
    all_train = []
    all_test = []
    for i in range(itr+1):
        train_bool = (i % train_freq) == 0
        test_bool = (i % test_freq) == 0
        if train_bool is False and test_bool is False:
            TrainingStep(i, update_test_data=test_bool, update_train_data=train_bool)
        else:
            train_vals, test_vals = TrainingStep(i, update_test_data=test_bool, update_train_data=train_bool)
            if len(train_vals) > 0:
                all_train.append(train_vals)
            if len(test_vals) > 0:
                all_test.append(test_vals)
                
    return all_train, all_test
```


```python
train_vals, test_vals = TrainData()
```

    i = 0	 Training Accuracy: 0.06	 Loss: 232.134
    i = 0	 Epoch 1.0	 test accuracy: 0.1009	 test loss: 230.289
    i = 1000	 Training Accuracy: 0.97	 Loss: 18.8419
    i = 1000	 Epoch 2.66666666667	 test accuracy: 0.9628	 test loss: 13.4265
    i = 2000	 Training Accuracy: 0.99	 Loss: 2.1172
    i = 2000	 Epoch 4.33333333333	 test accuracy: 0.9706	 test loss: 10.5921
    i = 3000	 Training Accuracy: 0.98	 Loss: 7.36559
    i = 3000	 Epoch 6.0	 test accuracy: 0.9768	 test loss: 8.70295
    i = 4000	 Training Accuracy: 0.99	 Loss: 2.06494
    i = 4000	 Epoch 7.66666666667	 test accuracy: 0.9779	 test loss: 8.30613
    i = 5000	 Training Accuracy: 0.99	 Loss: 2.33446
    i = 5000	 Epoch 9.33333333333	 test accuracy: 0.9779	 test loss: 8.02562
    i = 6000	 Training Accuracy: 0.97	 Loss: 8.98592
    i = 6000	 Epoch 11.0	 test accuracy: 0.9793	 test loss: 8.31421
    i = 7000	 Training Accuracy: 0.98	 Loss: 4.9268
    i = 7000	 Epoch 12.6666666667	 test accuracy: 0.9795	 test loss: 8.19503
    i = 8000	 Training Accuracy: 1.0	 Loss: 0.0980976
    i = 8000	 Epoch 14.3333333333	 test accuracy: 0.98	 test loss: 8.09575
    i = 9000	 Training Accuracy: 1.0	 Loss: 0.721754
    i = 9000	 Epoch 16.0	 test accuracy: 0.9805	 test loss: 8.15281
    i = 10000	 Training Accuracy: 1.0	 Loss: 0.04327
    i = 10000	 Epoch 17.6666666667	 test accuracy: 0.9801	 test loss: 8.42842



```python
loss_test = []
loss_train = []
i_test = []
i_train = []
acc_test = []
acc_train = []

for item in train_vals:
    i_train.append(item[0])
    acc_train.append(item[1])
    loss_train.append(item[2])
    
for item in test_vals:
    i_test.append(item[0])
    acc_test.append(item[1])
    loss_test.append(item[2])
```


```python
plt.clf()
plt.plot(i_train, acc_train)
plt.title("Training Accuracy Per Batch")
plt.xlabel("Batch")
plt.ylabel("Accuracy")
plt.show()

plt.clf()
plt.plot(i_test, acc_test)
plt.title("Test Accuracy Per Batch")
plt.xlabel("Batch")
plt.ylabel("Accuracy")
plt.show()

plt.clf()
plt.plot(i_train, loss_train, 'r')
plt.title("Train Loss Per Batch")
plt.xlabel("Batch")
plt.ylabel("Cross Entropy")
plt.show()

plt.clf()
plt.plot(i_test, loss_test, 'r')
plt.title("Test Loss Per Batch")
plt.xlabel("Batch")
plt.ylabel("Cross Entropy")
plt.show()
```


![png](images/2017/05/05_1-NN/output_26_0.png)



![png](images/2017/05/05_1-NN/output_26_1.png)



![png](images/2017/05/05_1-NN/output_26_2.png)



![png](images/2017/05/05_1-NN/output_26_3.png)


