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

Up Sampling is a technique that is used for when one class of the data is drastically unevenly represented in the data. To help with this, upsampling continually resamples from the underrepresented
class until both data sets are represented relatively equal. This is done for the *training set*.

It is important that this is only done for the training set since you want the test data to remain relatively the same unevenness as the original data when predicting.

For example, in the case of credit card fraud, it only happens in a fraction of a percent of credit card transactions. Therefore, for the best "total accuracy" if we just let it train on the normal data, it would essentially ignore the fradulent cases since they're not as represented. We changed this by giving it a roughly 50/50 distribution between the two classes, resampling from the data.

However, when running our trained model on the test set, we want to see how it will perform on the "real world data," so the test set was not upsampled at all. 

# Credit Card Fraud Detection

All three of these techniques can be used for classifying credit card fraud. The
dataset was download from [here](https://www.kaggle.com/dalpozz/creditcardfraud). It's important to note that the data was anonymized so that there was no identifiable information in it. Therefore, it was impossible to tell what the "dominant property" was in identifying a transaction as credit card fraud. 

The code that was used for this classification problem can be found [here](https://github.com/cmertin/Machine_Learning/tree/master/Credit_Card_Fraud). 

In order to perform the classification on this data, the following steps were
taken:

1. Separate the data into two classes
2. Take 20% of the data from both classes as the "test set"
3. Build the training set with Upsampling
4. Independently shuffle both data sets


But first, we have to look at and explore the data


```python
%matplotlib inline
from __future__ import print_function, division
import datetime
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set(color_codes=True)
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegressionCV
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from scipy import stats, integrate
# Gets the path/library with my Machine Learning Programs and adds it to the
# current PATH so it can be imported
LIB_PATH = os.path.dirname(os.getcwd()) # Goes one parent directory up
LIB_PATH = LIB_PATH + "/Library/" # Appends the Library folder to the path
sys.path.append(LIB_PATH)
from ML_Alg import LogisticRegression, UpSample
from f_io import ReadCSV
```

And we can read in the data

```python
data_file = "creditcard.csv"
DIR = os.getcwd() + "/data/"
FILE = DIR + data_file

x, y = ReadCSV(FILE)
card_data = pd.read_csv(FILE)
```

Check for missing data in the dataset

```python
card_data.isnull().sum()
```




    Time      0
    V1        0
    V2        0
    V3        0
    V4        0
    V5        0
    V6        0
    V7        0
    V8        0
    V9        0
    V10       0
    V11       0
    V12       0
    V13       0
    V14       0
    V15       0
    V16       0
    V17       0
    V18       0
    V19       0
    V20       0
    V21       0
    V22       0
    V23       0
    V24       0
    V25       0
    V26       0
    V27       0
    V28       0
    Amount    0
    Class     0
    dtype: int64

Learn what the columns mean


```python
card_data.describe()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Time</th>
      <th>V1</th>
      <th>V2</th>
      <th>V3</th>
      <th>V4</th>
      <th>V5</th>
      <th>V6</th>
      <th>V7</th>
      <th>V8</th>
      <th>V9</th>
      <th>...</th>
      <th>V21</th>
      <th>V22</th>
      <th>V23</th>
      <th>V24</th>
      <th>V25</th>
      <th>V26</th>
      <th>V27</th>
      <th>V28</th>
      <th>Amount</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>284807.000000</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>...</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>284807.000000</td>
      <td>284807.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>94813.859575</td>
      <td>3.919560e-15</td>
      <td>5.688174e-16</td>
      <td>-8.769071e-15</td>
      <td>2.782312e-15</td>
      <td>-1.552563e-15</td>
      <td>2.010663e-15</td>
      <td>-1.694249e-15</td>
      <td>-1.927028e-16</td>
      <td>-3.137024e-15</td>
      <td>...</td>
      <td>1.537294e-16</td>
      <td>7.959909e-16</td>
      <td>5.367590e-16</td>
      <td>4.458112e-15</td>
      <td>1.453003e-15</td>
      <td>1.699104e-15</td>
      <td>-3.660161e-16</td>
      <td>-1.206049e-16</td>
      <td>88.349619</td>
      <td>0.001727</td>
    </tr>
    <tr>
      <th>std</th>
      <td>47488.145955</td>
      <td>1.958696e+00</td>
      <td>1.651309e+00</td>
      <td>1.516255e+00</td>
      <td>1.415869e+00</td>
      <td>1.380247e+00</td>
      <td>1.332271e+00</td>
      <td>1.237094e+00</td>
      <td>1.194353e+00</td>
      <td>1.098632e+00</td>
      <td>...</td>
      <td>7.345240e-01</td>
      <td>7.257016e-01</td>
      <td>6.244603e-01</td>
      <td>6.056471e-01</td>
      <td>5.212781e-01</td>
      <td>4.822270e-01</td>
      <td>4.036325e-01</td>
      <td>3.300833e-01</td>
      <td>250.120109</td>
      <td>0.041527</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>-5.640751e+01</td>
      <td>-7.271573e+01</td>
      <td>-4.832559e+01</td>
      <td>-5.683171e+00</td>
      <td>-1.137433e+02</td>
      <td>-2.616051e+01</td>
      <td>-4.355724e+01</td>
      <td>-7.321672e+01</td>
      <td>-1.343407e+01</td>
      <td>...</td>
      <td>-3.483038e+01</td>
      <td>-1.093314e+01</td>
      <td>-4.480774e+01</td>
      <td>-2.836627e+00</td>
      <td>-1.029540e+01</td>
      <td>-2.604551e+00</td>
      <td>-2.256568e+01</td>
      <td>-1.543008e+01</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>54201.500000</td>
      <td>-9.203734e-01</td>
      <td>-5.985499e-01</td>
      <td>-8.903648e-01</td>
      <td>-8.486401e-01</td>
      <td>-6.915971e-01</td>
      <td>-7.682956e-01</td>
      <td>-5.540759e-01</td>
      <td>-2.086297e-01</td>
      <td>-6.430976e-01</td>
      <td>...</td>
      <td>-2.283949e-01</td>
      <td>-5.423504e-01</td>
      <td>-1.618463e-01</td>
      <td>-3.545861e-01</td>
      <td>-3.171451e-01</td>
      <td>-3.269839e-01</td>
      <td>-7.083953e-02</td>
      <td>-5.295979e-02</td>
      <td>5.600000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>84692.000000</td>
      <td>1.810880e-02</td>
      <td>6.548556e-02</td>
      <td>1.798463e-01</td>
      <td>-1.984653e-02</td>
      <td>-5.433583e-02</td>
      <td>-2.741871e-01</td>
      <td>4.010308e-02</td>
      <td>2.235804e-02</td>
      <td>-5.142873e-02</td>
      <td>...</td>
      <td>-2.945017e-02</td>
      <td>6.781943e-03</td>
      <td>-1.119293e-02</td>
      <td>4.097606e-02</td>
      <td>1.659350e-02</td>
      <td>-5.213911e-02</td>
      <td>1.342146e-03</td>
      <td>1.124383e-02</td>
      <td>22.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>139320.500000</td>
      <td>1.315642e+00</td>
      <td>8.037239e-01</td>
      <td>1.027196e+00</td>
      <td>7.433413e-01</td>
      <td>6.119264e-01</td>
      <td>3.985649e-01</td>
      <td>5.704361e-01</td>
      <td>3.273459e-01</td>
      <td>5.971390e-01</td>
      <td>...</td>
      <td>1.863772e-01</td>
      <td>5.285536e-01</td>
      <td>1.476421e-01</td>
      <td>4.395266e-01</td>
      <td>3.507156e-01</td>
      <td>2.409522e-01</td>
      <td>9.104512e-02</td>
      <td>7.827995e-02</td>
      <td>77.165000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>172792.000000</td>
      <td>2.454930e+00</td>
      <td>2.205773e+01</td>
      <td>9.382558e+00</td>
      <td>1.687534e+01</td>
      <td>3.480167e+01</td>
      <td>7.330163e+01</td>
      <td>1.205895e+02</td>
      <td>2.000721e+01</td>
      <td>1.559499e+01</td>
      <td>...</td>
      <td>2.720284e+01</td>
      <td>1.050309e+01</td>
      <td>2.252841e+01</td>
      <td>4.584549e+00</td>
      <td>7.519589e+00</td>
      <td>3.517346e+00</td>
      <td>3.161220e+01</td>
      <td>3.384781e+01</td>
      <td>25691.160000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 31 columns</p>
</div>

Look at the class frequency

```python
class_freq = card_data["Class"].value_counts()
print(class_freq)
```

    0    284315
    1       492
    Name: Class, dtype: int64

```python
sns.countplot(x="Class", data=card_data)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fafc6083860>




![png](images/2016/11/11_17-Loan_Approvals/output_11_1.png)

Since they are unequal, we can use upsampling


```python
data, test_data = UpSample(x, y)
```

Now we can look for correlations in the data


```python
X_data = card_data.iloc[:,1:29]

# Correlation matrix for margin features

corr = X_data.corr()

# Set up the matplotlib figure
plt.clf()
plt.subplots(figsize=(10,10))

# Draw the heat map
sns.heatmap(corr, vmax=0.3, square=True, xticklabels=5, yticklabels=5, linewidths=0.5, cbar_kws={"shrink": 0.5})

plt.title("Correlation Between Different Features")
```




    <matplotlib.text.Text at 0x7fafbd1f6668>




    <matplotlib.figure.Figure at 0x7fafc6007b00>


![png](images/2016/11/11_17-Loan_Approvals/output_15_2.png)


Finally, we can see the correlation of the number of fraud cases per hour


```python
fraud = card_data.loc[card_data["Class"] == 1]
```


```python
per_bins = 3600
bin_range = np.arange(0, 172801, per_bins)

out, bins = pd.cut(fraud["Time"], bins=bin_range, include_lowest=True, right=False, retbins=True)

out.cat.categories = ((bins[:-1]/3600)+1).astype(int)
out.value_counts(sort=False).plot(kind="bar", title="Fraud Cases Per Hour")
```

    <matplotlib.axes._subplots.AxesSubplot at 0x7fafc5faf320>




![png](images/2016/11/11_17-Loan_Approvals/output_18_1.png)

Now we can run the classifiers to predict the classes

### Support Vector Machines


```python
svm = LinearSVC()
svm.fit(data[0], data[1])
y_pred = svm.predict(test_data[0])
print(classification_report(y_pred, test_data[1]))
```

                 precision    recall  f1-score   support
    
            0.0       0.98      1.00      0.99     55690
            1.0       0.82      0.06      0.12      1274
    
    avg / total       0.98      0.98      0.97     56964
    


### Logistic Regression


```python
log_reg = LogisticRegressionCV()
log_reg.fit(data[0], data[1])
y_pred = log_reg.predict(test_data[0])
print(classification_report(y_pred, test_data[1]))
```

                 precision    recall  f1-score   support
    
            0.0       0.99      1.00      0.99     56315
            1.0       0.86      0.13      0.23       649
    
    avg / total       0.99      0.99      0.99     56964
    


### Gaussian Naive Bayes


```python
gnb = GaussianNB()
gnb.fit(data[0], data[1])
y_pred = gnb.predict(test_data[0])
print(classification_report(y_pred, test_data[1]))
```

                 precision    recall  f1-score   support
    
            0.0       0.99      1.00      1.00     56432
            1.0       0.68      0.13      0.22       532
    
    avg / total       0.99      0.99      0.99     56964

In these tests, Logistic Regression outperformed both Linear SVM and Naive Bayes.
This is most likely due to the fact that Logistic Regression performs well with
uneven data sets, and also that there is probably some correlation between the
features, reducing the effectiveness of Gaussian Naive Bayes.

Also, in looking at the results for each of these, the low values for `recall` and `f1-score` are not really relevant for the fraud causes. The reason is as follows:

Recall is defined by the following function

$$\text{\bf Recall} = \frac{t_{p}}{t_{p} + f_{n}}$$

where $$t_{n}$$ stands for "true positive" and $$f_{n}$$ stands for "false negative." While the accuracy for the cases that wasn't fraud detection was on the order of 99% accurate, this leaves approximately 1% being mislabeled. Due to the above definition of recall, this will greatly inflate the denominator since the number of mislabelled non-fraud transactions would be quite large compared to the total number of actual fradulent transactions.

The same goes for the f1-score for the fraud transacations since the F1 score is defined as

$$F_{1} = \frac{2t_{p}}{2t_{p} + f_{p} + f_{n}}$$

Since `recall` is a multiplied factor in the numerator, it skews the results here as well.