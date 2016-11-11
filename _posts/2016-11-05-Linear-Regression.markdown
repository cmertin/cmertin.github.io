Linear Regression is one of the most used algorithms when it comes to regression
in machine learning practices. _Linear Regression_ treats each feature of the
input feature vector as having a linear dependence on the regression model.

For example, imagine that you have some data such as height and age of various
people. Something that looks like

x: Age (years) | y: Height (meters)
---------------|-------------------
2.0659         | 0.7919
2.3684         | 0.9160
2.5400         | 0.9054
...            | ...

And so on, producing a plot of

![Data Points](images/11_5-Lin_Reg/linreg_1.png)

Now let's say that given a certain age, you want to predict the height in meters
from this data. This data would be considered your "training set," and would be
used to train your linear regression classifier so that you can predict results.

What it would essentially do is give you a linear fit based on some given parameter,
which in this instance would be the age. In other words, it _maps_ values from $$x$$
to $$y$$.

# Univariate Linear Regression

For everything in the 1-Dimensional case, the data and corresponding code that I
write can be found [here](https://github.com/cmertin/Machine_Learning/tree/master/Linear_Regression),
which contains the *entire* dataset from the plot above which will be used in this
exploration of linear regression.

As stated before, we want our linear regression model to essentially be able to
predict a value based on a given input parameter $$x$$. In other words, since it
is a linear model, we want some hypothesis function

$$h_{\theta}(x) = \theta_{0} + \theta_{1}x$$

where $$\theta_{0}$$ is the "offset" of our function. As you can see, the above
equation is linear in $$x$$. Essentially what we're going to get is a straight line
that fits the data above. We can do this with a _cost function_.

### Cost function

We want to choose $$\theta_{0}$$ and $$\theta_{1}$$ such that $$h_{\theta}(x)$$
is close to $$y$$ for our training examples $$(x,y)$$. More formally, we want
to solve

$$ \min_{\theta_{0},\theta{1}} \frac{1}{2N}\sum_{i=1}^{N}\left[ h_{\theta}\left(x^{(i)}\right) - y^{(i)} \right]^{2}$$

Where we're going to be minimizing the *average* difference, hence the fractional
value out front.

Using this above equation, we can define it as a function to determine the best
values, namely

$$J(\theta_{0}, \theta_{1}) =  \frac{1}{2N}\sum_{i=1}^{N}\left[ h_{\theta}\left(x^{(i)}\right) - y^{(i)} \right]^{2}$$

where $$J(\theta_{0}, \theta_{1})$$ is our **cost function**. We want to use our
cost function to solve

$$\min_{\theta_{0},\theta{1}} J(\theta_{0}, \theta_{1})$$

which is a minimization problem. This can be done with the
_Method of Gradient Descent_.

### Gradient Descent
