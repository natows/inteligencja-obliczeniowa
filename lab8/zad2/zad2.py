from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')
#vader to narzedzie ktore analizuje slowa i zwraca ich wage nacechowania, poprostu ma slownik gotowych slow w zprzypisanymi wagami
sentiment_analyzer = SentimentIntensityAnalyzer()

with open('neg_review.txt', 'r', encoding='utf-8') as file:
    neg_review = file.read()

with open('pos_review.txt', 'r', encoding='utf-8') as file:
    pos_review = file.read()


sentiment_1 = sentiment_analyzer.polarity_scores(neg_review)
sentiment_2 = sentiment_analyzer.polarity_scores(pos_review)
print("vader")
print("Negatywna recenzja:", sentiment_1)
print("Pozytywna recenzja:", sentiment_2)
# Negatywna recenzja: {'neg': 0.147, 'neu': 0.839, 'pos': 0.014, 'compound': -0.9139}
# Pozytywna recenzja: {'neg': 0.0, 'neu': 0.537, 'pos': 0.463, 'compound': 0.9836}

#po dodaniu slow nacechowanych emocjonalnie


# Negatywna recenzja: {'neg': 0.225, 'neu': 0.763, 'pos': 0.012, 'compound': -0.9702}
# Pozytywna recenzja: {'neg': 0.0, 'neu': 0.511, 'pos': 0.489, 'compound': 0.9894}


import text2emotion as te #tez gotowy slownik z emocjami ale wiecej kategorii

sentiment_1 = te.get_emotion(neg_review)
sentiment_2 = te.get_emotion(pos_review)
print("text2emotion")
print("Negatywna recenzja:", sentiment_1)
print("Pozytywna recenzja:", sentiment_2)
# Negatywna recenzja: {'Happy': 0.06, 'Angry': 0.17, 'Surprise': 0.11, 'Sad': 0.39, 'Fear': 0.28}
# Pozytywna recenzja: {'Happy': 0.56, 'Angry': 0.11, 'Surprise': 0.0, 'Sad': 0.22, 'Fear': 0.11}

#transformer - architektura modelu sztucznej inteligencji uzywana do analizy, przetwarzania i generowania tekstu, do porblemow NLP
# Transformer, zamiast przetwarzać tekst sekwencyjnie, patrzy na wszystkie słowa w zdaniu jednocześnie (równolegle), co znacznie przyspiesza proces uczenia się i pozwala modelowi lepiej rozumieć kontekst.


# BERT - Bidirectional Encoder Representations from Transformers - model transformera, który jest trenowany na dużych zbiorach danych tekstowych i może być dostosowywany do różnych zadań NLP, takich jak klasyfikacja tekstu, analiza sentymentu, odpowiedzi na pytania itp.
# BERT jest dwukierunkowy, co oznacza, że analizuje kontekst słowa zarówno z lewej, jak i z prawej strony, co pozwala mu lepiej rozumieć znaczenie słów w kontekście zdania.


from transformers import BertTokenizer, BertForSequenceClassification
import torch

tokenizer = BertTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

text_1= tokenizer(neg_review, return_tensors='pt', truncation=True, padding=True)
text_2= tokenizer(pos_review, return_tensors='pt', truncation=True, padding=True)

model = BertForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
#ten model przewiduje sentyment w skali 5 gwiazdek

with torch.no_grad(): 
    outputs_neg = model(**text_1) #rozpakowujemy jako kwargs bo text1 w formie slownika
    outputs_pos = model(**text_2)


scores_neg = outputs_neg.logits.softmax(dim=1)
scores_pos = outputs_pos.logits.softmax(dim=1)

print(scores_neg)
print(scores_pos)

# tensor([[8.5072e-01, 1.4152e-01, 6.9057e-03, 5.2947e-04, 3.2777e-04]])
# tensor([[0.0008, 0.0014, 0.0241, 0.4441, 0.5296]])

star = scores_neg.argmax()
print(f"Negatywna recenzja: {star.item()+1}")
star = scores_pos.argmax()
print(f"Pozytywna recenzja: {star.item()+1}")