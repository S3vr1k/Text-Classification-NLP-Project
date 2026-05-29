import pandas as pd
import re
import nltk
import joblib
import os

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline

data = pd.read_csv('data/dataset.csv')
russian_stopwords = stopwords.words('russian')

def clean_text(text): #очистка текста
    text = text.lower()
    text = re.sub(r'[^а-яА-Яa-zA-Z ]', '', text)
    words = text.split()
    filtered_words = [word for word in words if word not in russian_stopwords]
    return ' '.join(filtered_words)
data['text'] = data['text'].apply(clean_text)

X = data['text']
y = data['category']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Pipeline([ #векторизация
    ('vectorizer', TfidfVectorizer()),
    ('classifier', MultinomialNB())
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test) #прогноз
print('\nТочность модели:', round(accuracy_score(y_test, y_pred), 4))
print('\nОтчёт:\n', classification_report(y_test, y_pred))

os.makedirs('model', exist_ok=True) #сохранение модели
joblib.dump(model, 'model/classifier.pkl')
print('\nМодель успешно сохранена.')