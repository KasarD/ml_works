## Создание ИНС
### Обработка данных
1. Поиск данных, представляющих категории, в исходном датасете;
2. Преобразование символьных данных в цифровой формат при помощи **LabelEncoder** (from sklearn);
2.1. Если данные подразумевают сортировку по значению (то есть, если сортировка идет по алфавиту, то веса слов должны влиять на итоговый результат обучения), то этого будет достаточно;
2.2. Если же каждое слово имеет одинаковый вес, то необходимо использовать **OneHotEncoder**, который преобразует численные значения в их бинарный вид;
3. Данные разделяются на тренировочный и тестовый наборы (**train_test_split**);
4. Производится масштабирование данных к одному порядку (**StandartScaler**);
### Построение модели
1. Создается модель припомощи **Keras**;
2. Наполняем ее слоями;
3. Компилируем полученную модель;
4. Запускаем обучение.

Пример кода:
```python
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
# Сначала энкодер ест все страны, а потом преобразует их в числовой формат
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
# Аналогично и для пола человека
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
# Для исключения влияния категории (страны) на обучение, необходимо преобразовать
# Полученые результаты в бинарный вид, после чего удалить одну из колонок, исключив переобучаемость
onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
# Преобразуем все данные к одному порядку, для исключения влияния размерности
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
# Need to create a layer based structure for our ANN
from keras.models import Sequential
# Help to proceed values through all layers
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))

# Adding the second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
```