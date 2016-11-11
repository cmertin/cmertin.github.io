For this project, I will be using two regression techniques, [KNN Regression](https://cmertin.github.io/KNN-Regression.html) and Multivariate
[Linear Regression](https://cmertin.github.io/Linear-Regression.html). The difference
is that I'm going to be using Multivariate Linear Regression, instead of Univariate
Linear Regression like I discussed in my previous post.

### Multivariate Linear Regression

Multivariate linear regression is very similar to Univariate, the only difference
is that we have more variables/features and more derivatives to take. Our hypothesis function can be represented as

$$h_{\theta}(x) = \theta_{0} + \sum_{i=1}^{M}\theta_{i}x_{i}$$

where $$x_{0} = 1$$ and $$\vec{\theta} \in \mathbb{R}^{M+1}$$ and $$\vec{x} \in \mathbb{R}^{M+1}$$. This means that we can rewrite our function as

$$h_{\theta}(\vec{x}) = \vec{\theta}^{T}\vec{x}$$

where it is just a resulting vector inner-product. From here, there are two ways we
can solve this. One is by _Gradient Descent_ but also by the _Normal Equation_.

#### Multivariate Gradient Descent

Our **cost function** is similar to that in univariate gradient descent, and is

$$J(\vec{\theta}) =  \frac{1}{2N}\sum_{i=1}^{N}\left[ h_{\theta}\left(x^{(i)}\right) - y^{(i)} \right]^{2}$$

where $J$ is a function of the parameter vector, $$\vec{\theta}$$ rather than only
two $$\theta$$ parameters as the last time.

How we're going to implement Gradient Descent is

$$\theta_{j} = \theta_{j} - \alpha \frac{\partial}{\partial \theta_{j}}J(\vec{\theta})$$

where $$j \in \{ 0, 1, \ldots, m\}$$ since there is one for each feature. We also
want to update each $$\theta$$ simultaneously, just like before since we don't want
to update $$\theta_{j}$$ until the next iteration. The partial derivative terms follow
the form

$$ \frac{\partial}{\partial \theta_{j}} = \frac{1}{N} \sum_{i=1}^{N}\left[h_{\theta}\left( x^{(i)}\right) - y^{(i)} \right]x_{j}^{(i)}$$

which we can plug back into our equation above, giving

$$\theta_{j} = \theta_{j} - \alpha \frac{1}{N}\sum_{i=1}^{N}\left[h_{\theta}\left( x^{(i)}\right) - y^{(i)} \right]x_{j}^{(i)}$$

##### Feature Scaling



#### Normal Equation

### Polynomial Regression

# Air Quality Prediction

* Use KNN for missing data
* Use KNN and Multivariate Linear Regression to predict particulates
