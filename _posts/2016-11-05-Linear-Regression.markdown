Linear Regression is one of the most used algorithms when it comes to regression
in machine learning practices. _Linear Regression_ treats each feature of the
input feature vector as having a linear dependence on the regression model. It
can be represented as

$$y(\mathbf{x}) = \mathbf{\theta}^{T}\mathbf{x}$$

which for the 1-Dimensional case translates to

$$y(\mathbf{x}) = \theta_{0} + x\theta_{1}$$

where $$\theta_{0}$$ is the "offset" and comes about by setting $$x_{0} = 1$$,
thus leaving it as a single dimensional problem.

This can be extrapolated into a multi-dimensional problem quite easily, giving

$$y(\mathbf{x}) = \theta_{0} + \sum_{i=1}^{N}\theta_{i}$$

where again we have $$x_{0} = 1$$. We can reformulate this to simply being

$$y(\mathbf{x}) = \mathbf{\theta}^{T}\mathbf{x}$$

which will give us the same result since it is just a simple vector multiplication.
