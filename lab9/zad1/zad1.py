from apyori import apriori
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('titanic.csv')
print(df.head())

data = df.iloc[:, 1:5]
print(data.head())
data = data.astype(str)
data = data.values.tolist() #apriori oczekuje listy list
print(data[0:5])


assoc_rules = apriori(data, min_support=0.005, min_confidence=0.8)
 # support - czestosc wystepowania danego zbioru we wszystkich danych - liczba okreslonych zbiorow przez liczbe zbiorow ogolnie
# confidence - ile razy wystapilo base gdy wystapilo add
#apriori - algorytm do znajdowania reguł asocjacyjnych w zbiorach danych
#min_support - minimalna czestotliwosc wystepowania, jesli nie pojawi sie w 0.5%mprzypadkow to ni e jest uwzgledniane
#min_confidence - minimalna pewnosc, ze reguła jest prawdziwa, jesli nie jest spełniona to reguła nie jest uwzgledniana

final_data = list(assoc_rules)

rules = []
for rule in final_data:
    for stat in rule.ordered_statistics:
        base = set(stat.items_base)
        add = set(stat.items_add)
        if base and add:
            rules.append({
                "rule": f"{base} -> {add}", # regula : jesli mamy to to mamy to
                "base": base, # zbior bazowy
                "add": add, # zbior dodawany
                "support": rule.support, # support - czestosc wystepowania danego zbioru we wszystkich danych - liczba okreslonych zbiorow przez liczbe zbiorow ogolnie
                "confidence": stat.confidence, # confidence - ile razy wystapilo base gdy wystapilo add
                # confidence = support(A U B) / support(A)
                "lift": stat.lift # lift - wspolczynnik podnoszenia, miara siły reguly
            })
rules_sorted_by_confidence = sorted(rules, key=lambda x: x['confidence'], reverse=True)

# for rule in rules_sorted_by_confidence: #wszystkie reguly malejaco pod wzgledem ufnosci
#     print(f"Rule: {rule['rule']}")
#     print(f"Support: {rule['support']}")
#     print(f"Confidence: {rule['confidence']}")
#     print(f"Lift: {rule['lift']}")
#     print()

survival_rules = [rule for rule in rules_sorted_by_confidence if "No" in rule['add'] or "Yes" in rule['add']]
# reguly dotyczace przezycia
for rule in survival_rules:
    print(f"Rule: {rule['rule']}")
    print(f"Support: {rule['support']}")
    print(f"Confidence: {rule['confidence']}")
    print(f"Lift: {rule['lift']}")
    print()
        

# Rule: {'Child', '2nd'} -> {'Yes'}
# Support: 0.01090909090909091
# Confidence: 1.0     # 100 % takich przypadkow przezywalo
# Lift: 3.0942334739803092   # 3 krotnie wyzsza szansa na przezycie

# Rule: {'Child', '2nd', 'Female'} -> {'Yes'}
# Support: 0.005909090909090909
# Confidence: 1.0       
# Lift: 3.0942334739803092

# Rule: {'Child', '2nd', 'Male'} -> {'Yes'}
# Support: 0.005
# Confidence: 1.0
# Lift: 3.0942334739803092

# Rule: {'Female', '1st'} -> {'Yes'}
# Support: 0.06409090909090909
# Confidence: 0.9724137931034483
# Lift: 3.0088753091808527

# Rule: {'Adult', 'Female', '1st'} -> {'Yes'}
# Support: 0.06363636363636363
# Confidence: 0.9722222222222221
# Lift: 3.0082825441475225

# Rule: {'Female', '1st'} -> {'Adult', 'Yes'}
# Support: 0.06363636363636363
# Confidence: 0.9655172413793103
# Lift: 3.247917325740799

# Rule: {'Adult', '2nd', 'Male'} -> {'No'}
# Support: 0.07
# Confidence: 0.9166666666666666
# Lift: 1.354376539064249

# Rule: {'Female', '2nd'} -> {'Yes'}
# Support: 0.042272727272727274
# Confidence: 0.8773584905660378
# Lift: 2.7147520101902716

# Rule: {'Female', 'Crew'} -> {'Yes'}
# Support: 0.00909090909090909
# Confidence: 0.8695652173913043
# Lift: 2.6906378034611387

# Rule: {'Female', 'Crew'} -> {'Adult', 'Yes'}
# Support: 0.00909090909090909
# Confidence: 0.8695652173913043
# Lift: 2.925142933120595

# Rule: {'Adult', 'Female', 'Crew'} -> {'Yes'}
# Support: 0.00909090909090909
# Confidence: 0.8695652173913043
# Lift: 2.6906378034611387

# Rule: {'2nd', 'Male'} -> {'No'}
# Support: 0.07
# Confidence: 0.8603351955307263
# Lift: 1.27114669588153

# Rule: {'2nd', 'Male'} -> {'Adult', 'No'}
# Support: 0.07
# Confidence: 0.8603351955307263
# Lift: 1.3162290891290667

# Rule: {'Adult', 'Female', '2nd'} -> {'Yes'}
# Support: 0.03636363636363636
# Confidence: 0.8602150537634408
# Lift: 2.66170621417661

# Rule: {'Adult', '3rd', 'Male'} -> {'No'}
# Support: 0.1759090909090909
# Confidence: 0.8376623376623377
# Lift: 1.2376475103137294

# Rule: {'3rd', 'Male'} -> {'No'}
# Support: 0.19136363636363637
# Confidence: 0.8271119842829078
# Lift: 1.222059345481798


# z regul asocjacyjnych wynika, że kobiety i dzieci miały większe szanse na przeżycie niż mężczyźni.


#counted survivors
survivors = pd.crosstab(df["Sex"], df['Survived'])
print(survivors)

# Survived    No  Yes
# Sex
# Female     126  344
# Male      1364  367




# top_rules = rules_sorted_by_confidence[:20]
# print(top_rules)
# top_rules = pd.DataFrame(top_rules)
# plt.figure(figsize=(15, 8))
# sns.barplot(data=top_rules, x="confidence", y="rule", palette="viridis")
# plt.xlabel("Confidence")
# plt.ylabel("Reguła (Antecedent → Consequent)")
# plt.title("Top 10 reguł asocjacyjnych dotyczących przeżycia")
# # plt.tight_layout()
# plt.show()
# to nie ma sensu bo wszystkie reguly maja okolo 100% pewnosci

survival_rules = pd.DataFrame(survival_rules)
plt.figure(figsize=(15, 8))
sns.barplot(data=survival_rules, x="confidence", y="rule", palette="viridis")
plt.xlabel("Confidence")
plt.ylabel("Reguła (Antecedent → Consequent)")
plt.title("Top 10 reguł asocjacyjnych dotyczących przeżycia")
plt.tight_layout()
plt.show()
plt.savefig('apriori.png')

