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

![Sigmoid Function](https://upload.wikimedia.org/wikipedia/commons/thumb/8/88/Logistic-curve.svg/600px-Logistic-curve.svg.png)

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
-\log(1 - h_{\theta}(x)) & \text{if }y == 0 \end{array} \right.$$

Which makes sense as a cost function to use. Because if $$y = 1$$ and $$h_{\theta}(x) = 1$$, then the cost is $$0$$. However, as $$h_{\theta}(x) \rightarrow 0$$, we get
$$cost \rightarrow \infty$$. This is good because we want our classifier to pay
a large cost if it gets it wrong.

We can simplify this cost function into a single equation, instead of a piecewise equation, as being

$$cost(h_{\theta}(x), y) = -y\log(h_{\theta}(x)) - (1-y)\log(1 - h_{\theta}(x))$$

Which we are able to do since $$y = \{ 0, 1\}$$. So it cancels out the appropriate
opposing factor. This gives our cost function as being

$$J(\theta) = -\frac{1}{N}\left[\sum_{i=1}^{M}y^{(i)}\log(h_{\theta}(x^{(i)})) + (1 - y^(i))\log(1 - h_{\theta}(x^{(i)})) \right]$$

And, again, to get the values of $$\theta$$ we want

$$\min_{\theta}J(\theta)$$


# Support Vector Machines

# Classifying Loan Approvals
