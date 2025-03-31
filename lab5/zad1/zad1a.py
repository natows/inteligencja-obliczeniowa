import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.datasets import load_iris 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler, OneHotEncoder 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense 
from tensorflow.keras.utils import plot_model 
 
# Load the iris dataset 
iris = load_iris() 
X = iris.data 
y = iris.target 


# # Preprocess the data 

# # Scale the features 
scaler = StandardScaler() 
X_scaled = scaler.fit_transform(X)  #skaluje dane wejsciowe tak aby ich srednia wynosila 0 a odchylenie stand 1

 
# # Encode the labels 
encoder = OneHotEncoder(sparse_output=False) #one-hot encoding to kodowanie wartosci na wektory binarne  gdzie tylko jedna wartosc jest rowna 1 a reszta 0
#NIE KODUJEMY LICZB TYLKO KATEGORIE (wektory nie sa binarna reprezentacja liczb)
#koduje sie dlatego ze model po pierwsze nie rozumie tekstu a po drugie moze pomyslec ze np virginica reprezentowana przez 2 jest wazniejsza niz setosa 0
y_encoded = encoder.fit_transform(y.reshape(-1, 1))  #reshape bo onehotencoder przyjmuje tablice 2D 

 
# Split the dataset into training and test sets 
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.3, 
random_state=42) 
 
#pDefine the model 
model = Sequential([ #model wartwowy, kazda wartwa ukryta definiowana jedna po drugiej
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)), #pierwsza wartwa 64 neurony kazdy z funkcja aktywacji ReLU i kazdy neuron oczekuje 4 cech wejsciowych (bo x_train.shape[1])
    Dense(64, activation='relu'), #druga wartwa 64 neuronowa 
    Dense(y_encoded.shape[1], activation='softmax') #warstwa wyjsciowa, liczba neuronow rowna liczbie klas - y_encoded.shape[1] ==3
]) #softmax - funckja aktywacji uzywana w warstwach wyjsciowych by przeksztalcic surowe wartosci wyjsciowe (logity) do rozkladu prawdopodobienstwa 
#czyli np na wyjsciu dostajemy logity [2.1, 0.9, -1.2], po przejściu przez softmax (prawdopodobieństwa dla każdej klasy):[0.72, 0.25, 0.03]
 
#d) przy zamianie ReLU na tanh - accuracy spada do niecalych 98%
#przy zamianie ReLU na sigmoid - accuracy spada do niecalych  96%

# Compile the model 
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) 
#tu definiujemy JAK model ma sie uczyc i oceniac swoje wyniki
#adam (Adaptive Moment Estimation) - algorytm optymalizacji, sluzy do dostosowywania wag modelu podczas uczenia sieci
# Adam łączy w sobie dwie inne techniki: momentum (gdzie pamiętamy poprzednie zmiany wag) oraz adaptacyjne tempo uczenia się (gdzie zmieniamy tempo uczenia się w zależności od historii gradientu).

#loss - parametr funckji straty, mierzy jak bardzo przewidywania modelu odbiegaja od rzeczywistych danych
#cathegorical_crossentropy - funckja straty, uzywana w problemach klasyfikacji wieloklasowej(czyli jak mamy wiecej niz 2 klasy)

#parametr metrics sluzy do okreslenia jakie metryki chcemy sledzic by ocenic skutecznosc modelu podczas treningu
#accuracy - dokladnosc dopasowan przez model czyli stosunek poprawnych przewidywan do wszystkich przewidywan
#jest to najczesciej stosowana metryka, ale moze nie byc skuteczna przy zbiorach niezrownowazonych (np jest 90% setosa i 10% virginica to model bedzie przewidywal setosa w 90% przypadkow i bedzie mial 90% accuracy wiec niby super ale virginici nie przewidzi wogole czyli jest do dupy tak naprawde pomimo 90% accuracy)


# Train the model 
history = model.fit(X_train, y_train, epochs=100, validation_split=0.2) 
#epochs - liczba epok neuronowych czyli ile razy zbior treningowy bedzie przechodzil przzez siec
#po kazdej epoce model dostosowuje wagi w celu poprawy wynikow
#validation_split - czesc zbioru treningowego ktora bedzie uzyta do waidacji modelu - oceny jakosci w trakcie treningu

#funckja fit zwroci obiekt history ktory zawiera info o przebiegu treningu


# Evaluate the model on the test set 
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0) 
print(f"Test Accuracy: {test_accuracy*100:.2f}%") 
print(f"Test Loss: {test_loss:.4f}")
#verbose na 0  oznacza ze wynik oceny modelu nie bedzie wyswietlany w trakcie obliczen
#metoda evaluate zwraca wartosci straty i dokladnosci modelu na zbiorze testowym

# Plot the learning curve 
plt.figure(figsize=(12, 6)) 
plt.subplot(1, 2, 1) 
plt.plot(history.history['accuracy'], label='train accuracy') 
plt.plot(history.history['val_accuracy'], label='validation accuracy') 
plt.title('Model accuracy') 
plt.ylabel('Accuracy') 
plt.xlabel('Epoch') 
plt.grid(True, linestyle='--', color='grey') 
plt.legend() 
 
plt.subplot(1, 2, 2) 
plt.plot(history.history['loss'], label='train loss') 
plt.plot(history.history['val_loss'], label='validation loss') 
plt.title('Model loss') 
plt.ylabel('Loss') 
plt.xlabel('Epoch') 
plt.grid(True, linestyle='--', color='grey') 
plt.legend() 
 
plt.tight_layout() 
plt.show() 
 
# Save the model 
model.save('iris_model.keras') 
 
# Plot and save the model architecture 
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True) 