# Logistic Regression

Logistic Regression is a classification algorithm for classifying discrete classes.
In other words, we want to classify data such that we get $$y = 0$$ or $$y = 1$$
for the class. This can be transformed into a _Multi-class Classifier_, such that
$$y = \{0, 1, 2, \ldots\}$$, but for now we will just take the approach of it being
a single classifier.

Logistic Regression has a hypothesis function $$h_{\theta}(x)$$ that produces
values in the range $$0 \leq h_{\theta}(x) \leq 1$$. This is due to the fact that
the equation for the hypothesis is

$$h_{\theta}(x) = \frac{1}{1 + e^{-\vec{\theta}^{T}\vec{x}}}$$

Which is also known as the "Sigmoid Function." The Sigmoid Function has the shape

![Sigmoid Function](images/2016/11/11_17-Loan_Approvals/Sigmoid_Function.png)

which asymptotes at $$1$$ and $$0$$. Just like with Linear Regression, we need to
fit the parameters $$\theta$$ to the data to produce the results. We can interpret
the output of $$h_{\theta}(x)$$ as being the "probability" that $$y = 1$$ on
a given $$x$$.

In order to classify the data, we need a decision boundary, which is the point
at which we will decide if we classify the data as $$y = 1$$ or $$y = 0$$. What
we can do, is we can predict that $$y = 0$$ if $$h_{\theta}(x) < 0.5$$ and
$$y = 1$$ if $$h_{\theta}(x) \geq 0.5$$.

From the picture of the Sigmoid Function, we can see that $$h_{\theta}(x) \geq 0.5$$
for values in the positive domain, and $$h_{\theta}(x) < 0.5$$ in the negative domain.

Therefore, we can say that $$h_{\theta}(x) \geq 0.5$$ whenever $$\vec{\theta}^{T}\vec{x} \geq 0$$. And conversly, we can say that it will be less than $$0.5$$ whenever
$$\vec{\theta}^{T}\vec{x} < 0$$.

## Cost Function

We can define our cost function for Logistic Regression as being

$$cost(h_{\theta}(x), y) = \left\{ \begin{array}-\log(h_{\theta}(x)) & \text{if }y = 1\\
-\log(1 - h_{\theta}(x)) & \text{if }y = 0 \end{array} \right.$$

Which makes sense as a cost function to use. Because if $$y = 1$$ and $$h_{\theta}(x) = 1$$, then the cost is $$0$$. However, as $$h_{\theta}(x) \rightarrow 0$$, we get
$$cost \rightarrow \infty$$. This is good because we want our classifier to pay
a large cost if it gets it wrong.

We can simplify this cost function into a single equation, instead of a piecewise equation, as being

$$cost(h_{\theta}(x), y) = -y\log(h_{\theta}(x)) - (1-y)\log(1 - h_{\theta}(x))$$

Which we are able to do since $$y = \{ 0, 1\}$$. So it cancels out the appropriate
opposing factor. This gives our cost function as being

$$J(\theta) = -\frac{1}{N}\left[\sum_{i=1}^{M}y^{(i)}\log(h_{\theta}(x^{(i)})) + (1 - y^{(i)})\log(1 - h_{\theta}(x^{(i)})) \right]$$

And, again, to get the values of $$\theta$$ we want to solve

$$\min_{\theta}J(\theta)$$

For which we will use Gradient Descent. In simplifying the $$\frac{\partial}{\partial \theta_{j}}J(\theta)$$ like before, we get

$$\theta_{j} = \theta_{j} - \alpha \sum_{i=1}^{N}\left( h_{\theta}(x^{(i)}) - y^{(i)} \right)x_{j}^{(i)}\quad \forall j = \{ 0, 1, \ldots, M\}$$

which is the same as we had before. Therefore, we can use the same function as
we did for Gradient Descent to get the minimization of $$\theta$$.

## Regularized Logistic Regression

Regularizing helps with overfitting. In order to implement regularization, we
need to change the cost function. For example, we now have

$$J(\theta) = \frac{1}{2N}\left[ \sum_{i=1}^{N}(h_{\theta}(x^{(i)}) - y^{(i)})^2 + \lambda \sum_{i=1}^{M}\theta_{j}^{2} \right]$$

With the second term being the **Regularization Term**, and $$\lambda$$ is the
_regularization parameter_, which controls the parameters of fitting the training
set well, and secondly keeping the parameters small.

If $$\lambda$$ is very large $$(\sim 10^{10})$$, then we will start penalizing
all of the parameters and we will have all of the parameters tend towards zero.
And if $$\lambda$$ is very small $$(\sim 10^{-10})$$, then it will have very little
effect on regularizing the data and is back to prone to overfitting again. Therefore,
we need a "good choice" in choosing $$\lambda$$.

We also need to rederive $$\frac{\partial}{\partial \theta_{j}}J(\theta)$$ since
it has now changed. However, not much has changed so we can simply write it as

$$\theta_{j} = \theta_{j} - \alpha \frac{1}{N} \sum_{i=1}^{N}(h_{\theta}(x^{(i)}) - y^{(i)})x_{j}^{(i)} + \frac{\lambda}{N}\theta_{j}$$

Which ignores the first "offset" term since there is no need to regularize it,
and it stays the same as before. However, we can write all $$\theta_{j}$$'s as a
single equation by rearranging the two to give

$$\theta_{j} = \theta_{j}\left(1 - \alpha \frac{\lambda}{N}\right) - \alpha\frac{1}{N}\sum_{i=1}^{N}(h_{\theta}(x^{(i)}) - y^{(i)})x_{j}^{(i)}$$

This update can be used for Linear Regression as well.

# Support Vector Machines

Sometimes called a "Large Margin Classifier." This means that it attempts to
_maximize_ the margin spacing between the two classes, by maximizing the distance
between the closest $$y = 0$$ and $$y = 1$$, while still separating the two
classes.

It does this with the cost function

$$J(\theta) = C\sum_{i=1}^{N}\left[ y^{(i)}cost_{1}(\theta^{T}x^{(i)}) + (1-y^{(i)})cost_{0}(\theta^{T}x^{(i)}) \right] + \frac{1}{2}\sum_{j=1}^{M}\theta_{j}^{2}$$

where we want $$\theta^{T}x \geq 1$$ for $$y = 1$$ and $$\theta^{T}x \leq -1$$ for $$y = 0$$.

Linear SVM is implemented in Scikit-Learn by using `sklearn.svm.LinearSVC`,
with the documentation can be found here [here](http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC).

# Naive Bayes classification

Naive Bayes use Bayes' Theorem with the "naive" assumption of independence between
every pair of features. Given a class variable $$y$$ and a dependent feature vector
$$x_{1}$$ through $$x_{n}$$, Bayes' theorem states the following relationship:

$$P(y| x_{1},\ldots ,x_{n}) = \frac{P(y)P(x_{1},\ldots, x_{n}| y)}{P(x_{1},\ldots ,x_{n})}$$

Using the naive independence assumption that

$$P(x_{i} | y, x_{1}, \ldots, x_{i-1}, x_{i+1}, \ldots, x_{n}) = P(x_{i} | y)$$

for all $$i$$'s. This relationship can be simplified into

$$P(y | x_{1}, \ldots, x_{n}) = \frac{P(y)\prod_{i=1}^{n}P(x_{i} | y)}{P(x_{1}, \ldots, x_{n})}$$

Since $$P(x_{1}, \ldots, x_{n})$$ is constant for any of the inputs, we can simplify
it further by stating

$$P(y | x_{1}, \ldots, x_{n}) \propto P(y) \prod_{i=1}^{n} P(x_{i} | y)$$

Which we can then write down the classifier as

$$\hat{y} = \arg\!\max_{y} P(y)\prod_{i=1}^{n}P(x_{i}|y)$$

# Upsampling

Up Sampling is a technique that is used for when data is uneven in equal representation
of the two classes. What this does is it continually resamples from the underrepresented
class until both data sets are relatively equal. This is done for the *training set*.

It is important that the data remains relatively the same unevenness as the original
data set, as you want to see how your model works for predicting on the test set.

# Credit Card Fraud Detection

Both of these two techniques can be used for classifying credit card fraud. The
dataset was download from [here](https://www.kaggle.com/dalpozz/creditcardfraud).




###Linear SVM

&nbsp;       |**precision**   | **recall**  |**f1-score**   |**support**
-------------|------------|---------|-----------|-------
        0.0  |     0.97   |   1.00  |    0.98   |  55179
        1.0  |     0.83   |   0.05  |    0.09   |   1785
**avg / total** |       0.97|      0.97|      0.96|     56964


###Logistic Regression

&nbsp;       |**precision**   | **recall**  |**f1-score**   |**support**
-------------|------------|---------|-----------|-------
0.0  |     0.99   |   1.00  |    0.99   |  56294
1.0  |     0.88   |   0.13  |    0.23   |   670
**avg / total** |       0.99|      0.99|      0.99|     56964


###Gaussian Naive Bayes

&nbsp;       |**precision**   | **recall**  |**f1-score**   |**support**
-------------|------------|---------|-----------|-------
0.0  |     0.99   |   1.00  |    1.00   |  56432
1.0  |     0.68   |   0.13  |    0.23   |   532
**avg / total** |       0.99|      0.99|      0.99|     56964
