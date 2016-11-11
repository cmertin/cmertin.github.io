Linear Regression is one of the most used algorithms when it comes to regression
in machine learning practices. _Linear Regression_ treats each feature of the
input feature vector as having a linear dependence on the regression model.

For example, imagine that you have some data such as height and age of various
people. Something that looks like

x: Age (years) | y: Height (meters)
---------------|-------------------
2.0659         | 0.7919
2.3684         | 0.9160
2.5400         | 9.0538
...            | ...

And so on, producing a plot of

![Data Points](images/11_5-Lin_Reg/linreg_1.png)

Now let's say that given a certain age, you want to predict the height in meters
from this data. This data would be considered your "training set," and would be
used to train your linear regression classifier so that you can predict results.

What it would essentially do is give you a linear fit based on some given parameter,
which in this instance would be the age. In other words, it _maps_ values from $$x$$
to $$y$$.

# 1 Dimensional Linear Regression
