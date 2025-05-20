import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
# nltk.download('all')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

with open('article.txt', 'r', encoding='utf-8') as file:
    text = file.read()

#tokenizacja - split ale rozrozniajacy i slowa i znaki interpunkcyjne
tokens = word_tokenize(text)
# print("Tokens:", tokens)

#usuwanie stop words czyli slow ktore nie niosa ze soba zadnej informacji np w angielskim "the", "a", "an", "is", "are" itp

filtered_tokens = [word for word in tokens if word.lower() not in stopwords.words('english')]

print("ilosc slow:", len(filtered_tokens))

print("Unikalne słowa po usunięciu stop words:", set(filtered_tokens))

print("ilosc unikalnych slow:", len(set(filtered_tokens)))
stop_words = set(stopwords.words('english'))

custom_stop_words = ("n't", '.', "?", "'", '10', '12', '&', ',', '``', '(', ')', "'m", "''", '%', '”', '$', 'X', '100',  '—', 'X.', '“', '66.5', "'s")

stop_words.update(custom_stop_words)

filtered_tokens = [word for word in tokens if word.lower() not in stop_words]

print("ilosc slow:", len(filtered_tokens))

lemmatizer = WordNetLemmatizer() #ten lematyzer korzysta z bazy danych wordnet, lematyzacja uwzglednia gramatyke i forme slowa

lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
print("Lematyzacja:", lemmatized_tokens)


#stemming - algorytm ktory zmienia slowa do ich podstawowej formy, ale nie uwzglednia gramatyki i formy slowa, np. "running" -> "run", "better" -> "better"

processed_text = "".join(lemmatized_tokens)

word_count = Counter(lemmatized_tokens) # zlicza wystapienia kazdego slowa

top_10 = word_count.most_common(10)

print(word_count)
print("Top 10:", top_10)

x = [word[0] for word in top_10]
y = [word[1] for word in top_10]
plt.figure(figsize=(10, 6)) 
plt.bar(x,y)
plt.show()

wordcloud = WordCloud(
    width=800, 
    height=400, 
    background_color='white', 
    max_words=100, 
    contour_width=3, 
    contour_color='steelblue',
    colormap='viridis'
).generate_from_frequencies(word_count)

plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.tight_layout()
plt.savefig('word_cloud.png', dpi=300, bbox_inches='tight')
plt.show()