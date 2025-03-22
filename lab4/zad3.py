from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

df = pd.read_csv("c:\\Users\\natal\\Studies\\inteligencja-obliczeniowa\\lab4\\diabetes.csv")

(train_set, test_set) = train_test_split(df, train_size=0.7, random_state=29)

#ReLU - rectified linear unit - funckja aktywacji, wprowadza nieliniowosc do sieci neuronowej
#ReLU - f(x) = max(0,x)

train_inputs = train_set.iloc[:, 0:8].values 
train_classes = train_set.iloc[:, 8].values 
test_inputs = test_set.iloc[:, 0:8].values
test_classes = test_set.iloc[:, 8].values


mlp = MLPClassifier(hidden_layer_sizes=(6,3), max_iter=500, random_state=19, activation="relu") 

mlp.fit(train_inputs, train_classes)

train_predictions = mlp.predict(train_inputs)
print(f"sprawdzenie skutecznosci sieci na treningowych: {accuracy_score(train_classes,train_predictions)}")

test_predictions = mlp.predict(test_inputs)
print(f"sprawdzenie skutecznosci sieci na testowych: {accuracy_score(test_classes, test_predictions)}")

matrix = ConfusionMatrixDisplay.from_estimator(mlp, test_inputs, test_classes)
matrix.plot()

plt.savefig("c:\\Users\\natal\\Studies\\inteligencja-obliczeniowa\\lab4\\confusion_matrix.png")

#fp - false positive, fn - false negative
# wiecej jest false negative
# zalezy co jest gorsze bo fn moze porowadzic do nieleczenia choroby a fp do dodatowych badan