import pandas as pd 
from sklearn.model_selection import train_test_split
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

df = pd.read_csv("c:\\Users\\natal\\Studies\\inteligencja-obliczeniowa\\lab3\\iris1.csv")

(train_set, test_set) = train_test_split(df, train_size=0.7, random_state=29) 

# print(train_set)

train_inputs = train_set.iloc[:, 0:4].values 
train_classes = train_set.iloc[:, 4].values 
test_inputs = test_set.iloc[:, 0:4].values 
test_classes = test_set.iloc[:, 4].values


def classify_iris(sl, sw, pl, pw): 
    if 1.0 <= pl <= 1.9 and 0.1 <= pw <= 0.6: 
        return("Setosa") 
    elif 3.0 <= pl <= 5.0 and 1.0 <= pw <= 1.7 : 
        return("Versicolor") 
    else: 
        return("Virginica") 

count = 0 
len = test_set.shape[0] 
 
for i in range(len): 
    sl, sw, pl, pw = test_inputs[i]
    if classify_iris(sl, sw, pl, pw) == test_classes[i]: 
        count = count + 1 
 
print(count) 
print(count/len*100, "%") 

with open("c:\\Users\\natal\\Studies\\inteligencja-obliczeniowa\\lab3\\summary.txt", "a") as file: 
    file.write(f"Metoda wlasnego drzewa decyzyjnego: Poprawnie sklasyfikowano {count/len*100:.2f}% przypadkow\n") 
    file.close()