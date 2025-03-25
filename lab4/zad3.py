from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn import tree

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


#prawdzenie czy drzewo decyzyjne poradzi sobie lepiej z danymi o cukrzycy

clf = tree.DecisionTreeClassifier()

clf = clf.fit(train_inputs, train_classes)

count = 0
length = test_set.shape[0]
for i in range(length): 
    if clf.predict([test_inputs[i]]) == test_classes[i]: 
        count = count + 1 

print(f"sprawdzenie skutecznosci drzewa na testowych: {count / length * 100} %")

matrix = ConfusionMatrixDisplay.from_estimator(clf, test_inputs, test_classes)
matrix.plot()
plt.savefig("c:\\Users\\natal\\Studies\\inteligencja-obliczeniowa\\lab4\\confusion_matrix2.png")

#drzewko ma bardzo podobne accuracy


#testowanie na 2 innych sieciach neuronowych

mlp2 = MLPClassifier(hidden_layer_sizes=(6,3), max_iter=500, random_state=19, activation="tanh")
mlp2.fit(train_inputs, train_classes)
train_predictions = mlp2.predict(train_inputs)
print(f"skutecznosc sieci 2 warstwowej po 6 i 3 z aktywacja logistics na treningowych: {accuracy_score(train_classes,train_predictions)}")

test_predictions = mlp2.predict(test_inputs)
print(f"skutecznosc sieci 2 warstwowej po 6 i 3 z aktywacja logistics na testowych: {accuracy_score(test_classes, test_predictions)}")

mlp3 = MLPClassifier(hidden_layer_sizes=(6,6,6), max_iter=500, random_state=19, activation="relu")
mlp3.fit(train_inputs, train_classes)
train_predictions = mlp3.predict(train_inputs)
print(f"skutecznosc sieci 3 warstwowej po 6 z aktywacja ReLU na treningowych: {accuracy_score(train_classes,train_predictions)}")

test_predictions = mlp3.predict(test_inputs)
print(f"skutecznosc sieci 3 warstwowej po 6 z aktywacja ReLU na testowych: {accuracy_score(test_classes, test_predictions)}")


mlp4 = MLPClassifier(hidden_layer_sizes=(6,6,6), max_iter=500, random_state=19, activation="tanh")
mlp4.fit(train_inputs, train_classes)
train_predictions = mlp4.predict(train_inputs)
print(f"skutecznosc sieci 3 warstwowej po 6 z aktywacja logistics na treningowych: {accuracy_score(train_classes,train_predictions)}")

test_predictions = mlp4.predict(test_inputs)
print(f"skutecznosc sieci 3 warstwowej po 6 z aktywacja logistics na testowych: {accuracy_score(test_classes, test_predictions)}")

#najlepsza siec to ta z aktywacja relu i 3 warstwami po 6 neuronow aczkolwiek wszystkie wypadaja podobnie

#fp - false positive, fn - false negative
# wiecej jest false negative
# fn moze porowadzic do nieleczenia choroby a fp do dodatowych niepotrzebnych badan - gorsze fn
#model wypada slabo bo powinien byc zoptymalizowany pod wzgledem minimalizacji fn nawet kosztem zwiekszenia fp