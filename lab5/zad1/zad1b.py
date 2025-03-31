import numpy as np 
from sklearn.datasets import load_iris 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler, OneHotEncoder 
from tensorflow.keras.models import load_model 

iris = load_iris() 
X = iris.data 
y = iris.target 
 

scaler = StandardScaler() #skalujemy dane aby mialy rozklad normalny N(0,1)
X_scaled = scaler.fit_transform(X) 
 
encoder = OneHotEncoder(sparse_output=False) 
y_encoded = encoder.fit_transform(y.reshape(-1, 1)) #enkodujemy etykiety do postaci one-hot, reshape bo onehotencoder przyjmuje pojedyncze wartosci w tablicy 2D
 

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.3, 
random_state=42) 
 

model = load_model('iris_model.keras') #wczytujemy model z pliku
 

model.fit(X_train, y_train, epochs=10) #ponownie trenujemy model na nowych danych treningowych przez 10 epok
 

model.save('updated_iris_model.keras') 
 

test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0) 
print(f"Test Accuracy: {test_accuracy*100:.2f}%") 
print(f"Test Loss: {test_loss:.4f}")