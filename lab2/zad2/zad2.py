from sklearn import datasets
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

df = pd.read_csv("c:\\Users\\Admin\\inteligencja-obliczeniowa\\lab2\\iris1.csv")

X = df[["sepal.length", "sepal.width", "petal.length", "petal.width"]]
y = df["variety"]


def count_pca(components):
    pca = PCA(n_components=components).fit(X)
    print(X.columns[:components].tolist())
    print(pca.explained_variance_ratio_)
    return sum(pca.explained_variance_ratio_)

print(count_pca(4))
print(count_pca(3)) # 99% wariancji
print(count_pca(2)) #troche ponad 95% wariancji
print(count_pca(1)) #za malo

# z tego wynika ze aby zachowac 95% wariancji potrzebujemy 2 komponentow 
#potwierdzimy to wzorem


def count_loss():
    pca = PCA().fit(X)
    total_variance_sqrt = sum(pca.explained_variance_ ** 0.5)  

    for n in range(4, 0, -1):
        pca_n = PCA(n_components=n).fit(X)
        kept_variance_sqrt = sum(pca_n.explained_variance_ ** 0.5) 
        
        loss = (total_variance_sqrt - kept_variance_sqrt) / total_variance_sqrt  
        explained_variance = sum(pca_n.explained_variance_ratio_)
        
        print(f"Strata informacyjna przy {n} komponentach: {loss:.4f} (Wariancja: {explained_variance:.4f})")


            
count_loss()

newIris = PCA(n_components=2).fit_transform(X)

def draw_plot():
    plt.scatter(newIris[y=="Setosa"][:,0], newIris[y=="Setosa"][:,1], c="red", label="Setosa")
    plt.scatter(newIris[y=="Versicolor"][:,0], newIris[y=="Versicolor"][:,1], c="green", label="Versicolor")
    plt.scatter(newIris[y=="Virginica"][:,0], newIris[y=="Virginica"][:,1], c="blue", label="Virginica")
    plt.legend()
    plt.savefig("c:\\Users\\Admin\\inteligencja-obliczeniowa\\lab2\\zad2\\iris_plot.png")
    # plt.show()

draw_plot()

