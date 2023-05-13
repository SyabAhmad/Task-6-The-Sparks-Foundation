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