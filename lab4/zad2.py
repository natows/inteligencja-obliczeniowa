import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
df = pd.read_csv("c:\\Users\\natal\\Studies\\inteligencja-obliczeniowa\\lab4\\iris1.csv")

(train_set, test_set) = train_test_split(df, train_size=0.7, random_state=29)

train_inputs = train_set.iloc[:, 0:4].values 
train_classes = train_set.iloc[:, 4].values 
test_inputs = test_set.iloc[:, 0:4].values
test_classes = test_set.iloc[:, 4].values


scaler = StandardScaler()

train_inputs = scaler.fit_transform(train_inputs)
test_inputs = scaler.transform(test_inputs)

print("-----------------")
print("Model sieci z 4 neuronami wejsciowymi, 2 neuronami ukrytymi i 1 neuronem wyjsciowym")
mlp = MLPClassifier(hidden_layer_sizes=(2,), max_iter=10000, random_state=29)
 # propagacja wsteczna - wagi sa incjalizowane losowo, ale random state pozwala na powtorzenie wynikow
mlp.fit(train_inputs, train_classes)

train_predictions = mlp.predict(train_inputs)
print(f"sprawdzenie skutecznosci sieci na treningowych: {accuracy_score(train_classes,train_predictions)}") #czy jest przeuczony?? -raczej nie bo skutecznosc na testowych nie jest gorsza

test_predictions = mlp.predict(test_inputs)
print(f"sprawdzenie skutecznosci sieci na testowych: {accuracy_score(test_classes, test_predictions)}")

print("-----------------")
print("model sieci z 4 neuronami wejsciowymi, 3 neuronami ukrytymi i 1 neuronem wyjsciowym")
mlp = MLPClassifier(hidden_layer_sizes=(3,), max_iter=10000, random_state=29)

mlp.fit(train_inputs, train_classes)

train_predictions = mlp.predict(train_inputs)
print(f"sprawdzenie skutecznosci sieci na treningowych: {accuracy_score(train_classes,train_predictions)}")

test_predictions = mlp.predict(test_inputs)
print(f"sprawdzenie skutecznosci sieci na testowych: {accuracy_score(test_classes, test_predictions)}")

print("-----------------")
print("model sieci z 4 neuronami wejsciowymi, 2 warstwami ukrytymi po 3 neurony i 1 neuronem wyjsciowym")
mlp= MLPClassifier(hidden_layer_sizes=(3,3), max_iter=10000, random_state=29)

mlp.fit(train_inputs, train_classes)

train_predictions = mlp.predict(train_inputs)
print(f"sprawdzenie skutecznosci sieci na treningowych: {accuracy_score(train_classes,train_predictions)}")

test_predictions = mlp.predict(test_inputs)
print(f"sprawdzenie skutecznosci sieci na testowych: {accuracy_score(test_classes, test_predictions)}")

