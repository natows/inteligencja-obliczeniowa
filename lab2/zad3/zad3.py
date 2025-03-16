import matplotlib.pyplot as plt , pandas as pd

from sklearn.preprocessing import MinMaxScaler, StandardScaler



def draw_plot(data, name):
    plt.scatter(data[data.variety =="Setosa"]["sepal.length"],data[data.variety =="Setosa"]["sepal.width"], c="red", label="Setosa")
    plt.scatter(data[data.variety =="Versicolor"]["sepal.length"],data[data.variety =="Versicolor"]["sepal.width"],c="green", label="Versicolor")
    plt.scatter(data[data.variety == "Virginica"]["sepal.length"],data[data.variety == "Virginica"]["sepal.width"],  c="blue", label="Virginica")
    plt.xlabel("Sepal Length")
    plt.ylabel("Sepal Width")
    plt.legend()
    plt.savefig(f"c:\\Users\\natal\\Studies\\inteligencja-obliczeniowa\\lab2\\zad3\\{name}.png")
    plt.show()



df = pd.read_csv("c:\\Users\\natal\\Studies\\inteligencja-obliczeniowa\\lab2\\iris1.csv")
pl1 = draw_plot(df, "original")

#standaryzacja min-max
#sposob 1 reczny

# df["sepal.length"] =(df["sepal.length"] - df["sepal.length"].min()) / (df["sepal.length"].max() - df["sepal.length"].min())
# df["sepal.width"] = (df["sepal.width"] - df["sepal.width"].min()) / (df["sepal.width"].max() - df["sepal.width"].min())

#sposob 2 z uzyciem skalera
scaler = MinMaxScaler()
df[["sepal.length", "sepal.width"]] = scaler.fit_transform(df[["sepal.length", "sepal.width"]])

pl2 = draw_plot(df, "min-max")


#standaryzacja z-

#sposob 1 reczny
# df["sepal.length"] = (df["sepal.length"] - df["sepal.length"].mean()) / df["sepal.length"].std()
# df["sepal.width"] = (df["sepal.width"] - df["sepal.width"].mean()) / df["sepal.width"].std()


#sposob 2 z uzyciem skalera
scaler = StandardScaler()
df[["sepal.length", "sepal.width"]] = scaler.fit_transform(df[["sepal.length", "sepal.width"]])

pl3 = draw_plot(df, "z-score")




#dane originalne maja wieksze wartosci, min-max i z-score maja podobne wartosci
#z-score centruje dane wokol 0, dane po transformacji maja srednia 0 i wariancje 1