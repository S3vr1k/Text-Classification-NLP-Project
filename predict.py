import joblib

model = joblib.load('model/classifier.pkl') #загрузка модели

print('=====================================')
print('КЛАССИФИКАЦИЯ ТЕКСТОВ ПО ТЕМАТИКЕ')
print('=====================================')

while True:
    text = input('\nВведите текст для классификации: ')

    prediction = model.predict([text])[0]
    probabilities = model.predict_proba([text])[0]

    categories = model.classes_

    print('\nПредполагаемая тематика:', prediction)

    print('\nВероятности:')

    for category, probability in zip(categories, probabilities):
        print(f'{category}: {round(probability * 100, 2)}%')

    again = input('\nПродолжить? (y/n): ')

    if again.lower() == 'n':
        break