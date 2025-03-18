import pandas as pd 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from sklearn.naive_bayes import GaussianNB

df = pd.read_csv("c:\\Users\\natal\\Studies\\inteligencja-obliczeniowa\\lab3\\iris1.csv")

(train_set, test_set) = train_test_split(df, train_size=0.7, random_state=29) 

train_inputs = train_set.iloc[:, 0:4].values 
train_classes = train_set.iloc[:, 4].values 
test_inputs = test_set.iloc[:, 0:4].values
test_classes = test_set.iloc[:, 4].values

#klasyfikacja metoda k najblizszych sasiadow
for k in [3,5,11]:
    nbrs = KNeighborsClassifier(n_neighbors=k).fit(train_inputs, train_classes)
    count = 0
    length = test_set.shape[0]
    for i in range(length): 
        if nbrs.predict([test_inputs[i]]) == test_classes[i]: 
            count = count + 1 
    accuracy = count / length

    print(f"K-nearest neighbours method for k={k}: Accuracy: {accuracy:.2%}")
    with open("c:\\Users\\natal\\Studies\\inteligencja-obliczeniowa\\lab3\\summary.txt", "a") as file: 
        file.write(f"Metoda k-najblizszych sasiadow k={k}: Poprawnie sklasyfikowano {accuracy:.2%} przypadkow\n")
    matrix = ConfusionMatrixDisplay.from_estimator(nbrs, test_inputs, test_classes)
    matrix.plot()
    plt.savefig(f"c:\\Users\\natal\\Studies\\inteligencja-obliczeniowa\\lab3\\zad3\\confusion_matrix_k{k}.png")



#klasyfikacja metoda naive bayes
bayes = GaussianNB().fit(train_inputs, train_classes)
predicted = bayes.predict(test_inputs)
accuracy = accuracy_score(test_classes, predicted)
print(f"Gaussian Naive Bayes method: Accuracy: {accuracy:.2%}")
with open("c:\\Users\\natal\\Studies\\inteligencja-obliczeniowa\\lab3\\summary.txt", "a") as file: 
    file.write(f"Metoda naive bayes: Poprawnie sklasyfikowano {accuracy:.2%} przypadkow\n")
    file.close()
matrix = ConfusionMatrixDisplay.from_estimator(bayes, test_inputs, test_classes)
matrix.plot()
plt.savefig(f"c:\\Users\\natal\\Studies\\inteligencja-obliczeniowa\\lab3\\zad3\\confusion_matrix_bayes.png")

