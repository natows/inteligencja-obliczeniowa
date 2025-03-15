import pandas as pd, numpy as np

df = pd.read_csv("c:\\Users\\Admin\\inteligencja-obliczeniowa\\lab2\\zad1\\iris_with_errors.csv")

unfilled_data = df.isna().sum() + (df == "-").sum()
print(unfilled_data)


for column in ["sepal.length", "sepal.width", "petal.length", "petal.width"]:
    df[column] = pd.to_numeric(df[column], errors="coerce")
    mean = df[column].mean().round(1)
    df[column] = df[column].apply(lambda x: mean if pd.isna(x) or not (0<x<15) else x)
            


# wrong_types = []
# for i in df.variety.unique():
#     if i not in ["Setosa", "Versicolor", "Virginica"]:
#         wrong_types.append(i)
# print(wrong_types)

types = {"setosa": "Setosa", "Versicolour": "Versicolor", "virginica": "Virginica"}
for i in df.variety.unique():
    if i not in ["Setosa", "Versicolor", "Virginica"]:
        df["variety"] = df["variety"].replace(types)
        

df.to_csv("c:\\Users\\Admin\\inteligencja-obliczeniowa\\lab2\\iris_clean.csv", index=False)

        



