import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from tensorflow.keras.datasets import mnist 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D 
from tensorflow.keras.utils import to_categorical                                                                                                                                                                                                                  
from sklearn.metrics import confusion_matrix 
from tensorflow.keras.callbacks import History, ModelCheckpoint 
 
# Load dataset 
(train_images, train_labels), (test_images, test_labels) = mnist.load_data() 
 
# Preprocess data 
#domyslnie train i test_images sa 3D, a sieci konwolucyjne w keras oczekuja 4D 
#tablica jest tablica 60000 obrazow z czego kazdy to 28 tablic dlugosci 28
# 1 wymiar to liczba obrazow, 2 wymiar to wysokosc, 3 to szerokosc, a 4 to liczba kanalow(1 dla czarno-bialych)
#domyslnie wartosci w tablicach sa typu uint8 (od 0 do 255), a sieci neuronowe lepiej przetwarzaja floaty (wieksza precyzja wynikow)
#float 32 zajmuje mniej pamieci niz 64
#poprzez podzielenie na 255 normalizujemy wartosci do przedzialu 0-1
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1)).astype('float32') / 255 
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1)).astype('float32') / 255 
train_labels = to_categorical(train_labels) #to_cathegorical zmienia etykiety na kodowanie one-hot (zmienia liczby na wektory binarne) - okreslamy format wyjsciowy sieci, i niweluje ewentualne problemy z porownywaniem etykiet
test_labels = to_categorical(test_labels) 
original_test_labels = np.argmax(test_labels, axis=1)  # Save original labels for confusion matrix 
#np.argmax - zwraca indeksy najwiekszych wartosci wzdloz osi 1, czyli dla kazdej tablicy zwraca indeks 1, czyli poprostu etykiete

# Define model 
model = Sequential([ #siec konwolucyjna sekwencyjna
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)), 
    #warstwa konwolucyjna conv2d - wyodrebnia cechy(krawedzie,wzorce) poprzez nakladanie filtrow (jader) na obraz wejsciowy
    #splot(konwolucja) - mnozenie macierzy filtra (jadra) przez kazdy element obrazu, i zapisuje wyniki do mapy cech
    #mamy 32 filtry o wymiarach 3x3 wiec powstana 32 mapy cech
    #!! siec zaczyna od losowych filtrow i algorytm optymalizacji (tu adam) modyfikuje je podczas treningu
    MaxPooling2D((2, 2)), #warstwa maksymalnego probkowania (pooling) z oknem 2x2
    #ta warstwa redukuje rozmiar przestrzenny obrazu - tutaj dla wymiarow macierzy 2x2 , z kazdego takiego elementu obrazu wybierany jest najwiekszy element a reszta jest ignorowana
    #tutaj redukcja z map 26x26 do 13x13
    Flatten(), #konwertuje 3D mapy cech na 1D wektory - mamy 32 mapy 13x13 wiec 13*13*32 i powstaje sobie dlugi wektor ktory mozna podac klasycznej sieci neuronowej
    Dense(64, activation='relu'), #klasyczne warstwy geste
    Dense(10, activation='softmax') #warstwa wyjsciowa (10 neuronow dla cyfr 0-9) softmax przeksztalca na prawdopodobienstwa (suma 1)
]) 


checkpoint = ModelCheckpoint( #wbudowany callback do zapisywania modelu
    filepath="best_model.keras",
    monitor="val_accuracy",   #monitorujemy walidacyjna dokladnosc ktora lepiej odzwierciedla jak dobrze model sie uczy
    save_best_only=True,      
    mode="max",               
    verbose=1                 
)

# Compile model 
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) 
 
# Train model 
history = History() #tworzymy obiekt history do zapisywania wynikw treningu (strata, dokladnosc dla kazdej epoki)
model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_split=0.2, 
callbacks=[history, checkpoint]) 
#do sieci wchodza partie obrazow (batch_size) tu podany 64, domyslnie 32
#20 % danych treningowych zostanie przeznaczna na zbior walidacyjny czyli zbior do oceny jak dobrze model sie uczy

best_epoch = np.argmax(history.history['val_accuracy']) + 1  
best_val_accuracy = max(history.history['val_accuracy'])

print(f"Najlepszy model osiągnięto w epoce {best_epoch} z dokładnością walidacyjną {best_val_accuracy:.4f}")
 
# Evaluate on test set 
test_loss, test_acc = model.evaluate(test_images, test_labels) 
print(f"Test accuracy: {test_acc:.4f}")  #accuracy 98 %
 
# Predict on test images 
predictions = model.predict(test_images) 
predicted_labels = np.argmax(predictions, axis=1) 
 
# Confusion matrix 
cm = confusion_matrix(original_test_labels, predicted_labels) #najczesciej mylona jest 7 z 2 i 9 z 4 
plt.figure(figsize=(10, 7)) 
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues') 
plt.xlabel('Predicted') 
plt.ylabel('True') 
plt.title('Confusion Matrix') 
# plt.show()
plt.savefig("confusion_matrix.png") 
 
# Plotting training and validation accuracy 
plt.figure(figsize=(10, 5)) 
plt.subplot(1, 2, 1) 
  
plt.plot(history.history['accuracy'], label='Training Accuracy') 
plt.plot(history.history['val_accuracy'], label='Validation Accuracy') 
plt.xlabel('Epoch') 
plt.ylabel('Accuracy') 
plt.grid(True, linestyle='--', color='grey') 
plt.legend() 
 
# Plotting training and validation loss 
plt.subplot(1, 2, 2) 
plt.plot(history.history['loss'], label='Training Loss') 
plt.plot(history.history['val_loss'], label='Validation Loss') 
plt.xlabel('Epoch') 
plt.ylabel('Loss') 
plt.grid(True, linestyle='--', color='grey') 
plt.legend() 
 
plt.tight_layout() 
# plt.show()
plt.savefig("accuracy_loss.png") 
#niedouczenie (unfitting) nie wystepuje, accuracy testowe i walidacyjne przekracza 98%, a loss testowy i walidacyjny oscyluje w okolicach 0.05
#przeuczenie (overfitting) tez nie wystepuje - wyniki dla danych testowych i walidacyjnych sa zblizone


# Display 25 images from the test set with their predicted labels 
plt.figure(figsize=(10,10)) 
for i in range(25): 
    plt.subplot(5,5,i+1) 
    plt.xticks([]) 
    plt.yticks([]) 
    plt.grid(False) 
    plt.imshow(test_images[i].reshape(28,28), cmap=plt.cm.binary) 
    plt.xlabel(predicted_labels[i]) 
# plt.show() 
plt.savefig("predicted_labels.png")

