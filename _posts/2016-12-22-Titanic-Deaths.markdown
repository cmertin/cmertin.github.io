This dataset came from [kaggle](https://www.kaggle.com/c/titanic/) which stated to use [random forests](https://en.wikipedia.org/wiki/Random_forest) for prediction.

# Random Forests

Random forests are not much different from decision trees, with the difference being that they are an agglomeration of $$n$$ decision trees, or in other words a "forest."

In using this method, it allows for it to reduce the variance of the estimate by avaeraging together many estimates. For example, to train $$n$$ trees, we take $n$ subsets of the data, chosen randomly with replacement, and then compute the ensemble

$$f(\mathbf{x}) = \sum_{i=1}^{n}\frac{1}{n}f_{i}(\mathbf{x})$$

where $$f_{i}$$ is the $$i^{th}$$ tree. This technique is known as "bagging" or "bootstrap aggregating."

In implementing this algorithm, too high of a value of $$n$$ will be prone to over-fitting while too small a value will be prone to underfitting. 

# Titanic Deaths

The code can be found [here](https://github.com/cmertin/Machine_Learning/tree/master/Titanic).

|Label    |Description|
|---------|---------|
|survival | Survival (0 = No; 1 = Yes)|
|pclass   |       Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)|
|name     |       Name|
|sex      |       Sex|
|age      |       Age|
|sibsp    |       Number of Siblings/Spouses Aboard|
|parch    |       Number of Parents/Children Aboard|
|ticket   |       Ticket Number|
|fare     |       Passenger Fare|
|cabin    |       Cabin|
|embarked |       Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)|

![age_dist](images/2016/12/12_22-Titanic/age_titanic.png)
![age_prob](images/2016/12/12_22-Titanic/ages_titanic.png)
![fare_prob](images/2016/12/12_22-Titanic/fares_titanic.png)
![class_prob](images/2016/12/12_22-Titanic/class_titanic.png)
![features](images/2016/12/12_22-Titanic/feature_importance.png)


## Results

|Feature | Ranking |
|--------|---------|
|Fare    | 0.297525|
|Sex     | 0.267874|
|Age     | 0.259869|
|Class   | 0.090779|
|SibSp   | 0.049662|
|ParCh   | 0.034291|

|Prediction | precision | recall | f1-score | support |
|-----------|-----------|--------|----------|---------|
|0          | 0.86      | 0.89   | 0.88     | 267     |
|1          | 0.79      | 0.74   | 0.77     | 151     |
| avg/total | 0.84      | 0.84   | 0.84     | 418     |

"Know that a score of 0.79 - 0.81 is doing well on this challenge, and 0.81-0.82 is really going beyond the basic models!" [kaggle](https://www.kaggle.com/c/titanic/details/getting-started-with-random-forests)

Data downloaded from [kaggle](https://www.kaggle.com/c/titanic/)