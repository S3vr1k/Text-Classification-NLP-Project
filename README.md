# Text Classification NLP Project

## Описание проекта

Данный проект представляет собой систему автоматической классификации текстов по тематике с использованием методов машинного обучения и обработки естественного языка (NLP).

Программа анализирует введённый текст и определяет его тематику.

Поддерживаются:

* русский язык;

## Используемые тематики

* Sport
* Technology
* Politics
* Entertainment
* Science

---

# Используемые технологии

* Python
* Scikit-learn
* Pandas
* NLTK
* TF-IDF
* Naive Bayes
* Logistic Regression
* Linear SVM

---

# Возможности проекта

* Очистка текста
* Удаление stop-слов
* TF-IDF векторизация
* Классификация текстов
* Поддержка русского и английского языка
* Сравнение нескольких ML-моделей
* Вывод вероятностей классификации

---

# Структура проекта

```text
project/
│
├── data/
│   └── dataset.csv
│
├── model/
│
├── train.py
├── predict.py
└── requirements.txt
```

---

# Установка

## 1. Клонирование репозитория

```bash
git clone https://github.com/USERNAME/text-classifier.git
```

## 2. Переход в папку проекта

```bash
cd text-classifier
```

## 3. Установка библиотек

```bash
pip install -r requirements.txt
```

---

# Обучение модели

```bash
python train.py
```

После обучения модель автоматически сохранится в папку:

```text
model/classifier.pkl
```

---

# Запуск программы

```bash
python predict.py
```

---

# Пример работы

```text
Введите текст:
Scientists discovered a new planet

Тематика:
science
```

---

# Метрики качества

В проекте используются:

* Accuracy
* Precision
* Recall
* F1-score

---

# Используемые алгоритмы

## TF-IDF

Преобразует текст в числовые признаки.

## Multinomial Naive Bayes

Алгоритм машинного обучения для классификации текста.

## Logistic Regression

Модель линейной классификации.

## Linear SVM

Метод опорных векторов для классификации данных.
