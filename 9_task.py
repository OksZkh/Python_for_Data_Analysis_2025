import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

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

# Создаем интрефейс

st.title("Прогнозирование стоимости недвижимости")

total_square = st.number_input("Общая площадь (м²)", min_value=1)
rooms = st.number_input("Количество комнат", min_value=0, max_value=10)
floor = st.number_input("Этаж", min_value=1)

if st.button("Прогнозировать цену"):
    input_data = np.array([[total_square, rooms, floor]])
    predicted_price = float(model.predict(input_data)[0])

    st.write(f"Прогнозируемая цена недвижимости: {predicted_price:.2f} рублей")
