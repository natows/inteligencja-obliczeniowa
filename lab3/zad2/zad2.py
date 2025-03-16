import pandas as pd 
from sklearn import tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

df = pd.read_csv("c:\\Users\\natal\\Studies\\inteligencja-obliczeniowa\\lab3\\iris1.csv")


# podzial na zbior testowy (30%) i treningowy (70%), ziarno losowosci = 29 
(train_set, test_set) = train_test_split(df, train_size=0.7, random_state=29) 



train_inputs = train_set.iloc[:, 0:4].values # cechy
train_classes = train_set.iloc[:, 4].values # etykieta klasy
test_inputs = test_set.iloc[:, 0:4].values 
test_classes = test_set.iloc[:, 4].values


clf = tree.DecisionTreeClassifier()

clf = clf.fit(train_inputs, train_classes)


plt.figure(figsize=(20,10))
tree.plot_tree(clf, filled=True, feature_names=df.columns[:4], class_names=df['variety'].unique())
plt.savefig("c:\\Users\\natal\\Studies\\inteligencja-obliczeniowa\\lab3\\zad2\\tree.png")
count = 0
length = test_set.shape[0]
for i in range(length): 
    if clf.predict([test_inputs[i]]) == test_classes[i]: 
        count = count + 1 

print(count)
print(count / length * 100, "%")

matrix = ConfusionMatrixDisplay.from_estimator(clf, test_inputs, test_classes)
matrix.plot()
plt.savefig("c:\\Users\\natal\\Studies\\inteligencja-obliczeniowa\\lab3\\zad2\\confusion_matrix.png")
