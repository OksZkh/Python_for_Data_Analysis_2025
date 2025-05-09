import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Считываем данный из файла
train_df = pd.read_csv('realty_data.csv')
train_df.head()

# Оставляем только имеющие значимость данные
df = train_df[['total_square','rooms','floor','price']]

# Заполняем пропуски 0, так как пустые значения относятся к студиям
df['rooms'] = df['rooms'].fillna(0)

# Разделяем данных
X = df[['total_square','rooms','floor']]
Y = df[['price']]

# Обучаем модель
model = LinearRegression()
model.fit(X, Y)


with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)


