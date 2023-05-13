# Decision Tree Classifier
### ```Task#6 || The Sparks Foundation```

```python
print("Task#6: Decision Tree Classifier")
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
data = pd.read_csv("data/Iris.csv",delimiter=",")


xTrain, xTest, yTrain, yTest = train_test_split(data.iloc[:, :-1], data.iloc[:, -1], test_size=0.25)

tree = DecisionTreeClassifier(max_depth=3,random_state=42)
tree.fit(xTrain, yTrain)

acrcy = tree.score(xTest, yTest)

print(acrcy)

predict = tree.predict(xTest)

print(predict)
```




This code demonstrates how to train a decision tree classifier on the Iris dataset using scikit-learn.

The code uses the pandas library to read in the data from a CSV file, and then uses the train_test_split function from sklearn.model_selection to split the data into training and testing sets. The decision tree classifier is then trained on the training set using DecisionTreeClassifier from sklearn.tree, with a maximum depth of 3 and a random state of 42.

The accuracy of the model is then calculated using the score method, and the predicted class labels for the testing set are printed to the console using the predict method. Finally, the accuracy score is printed to the console as well.

Note that the accuracy score is a measure of how well the model is able to predict the class labels of the testing set. A score of 1.0 indicates that the model is able to predict all of the class labels perfectly, while lower scores indicate that the model may be less accurate. It is important to carefully evaluate the performance of the model using multiple metrics and techniques, and to tune the hyperparameters of the model as needed to achieve the best possible performance.