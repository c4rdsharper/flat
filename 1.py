import streamlit as st
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import csv
from io import StringIO
from sqlalchemy import create_engine
from sqlalchemy import create_engine, text as sql_text
import xgboost as xgb
import numpy as np
import matplotlib.ticker as ticker

#Титульник
st.title('АНАЛИЗ РЫНКА НЕДВИЖИМОСТИ Г.УЛАН-УДЭ')

#Считываем Датафрейм
df = pd.read_csv('cian')
del df["Unnamed: 0.1"]
del df["Unnamed: 0"]

con_pg = create_engine('postgresql+psycopg2://mycliaow:ntV1e4tt7w3EEIW1I-VS-vnzUTGmxezq@abul.db.elephantsql.com/mycliaow')
def psql_insert_copy(table, conn, keys, data_iter):
    # gets a DBAPI connection that can provide a cursor
    dbapi_conn = conn.connection
    with dbapi_conn.cursor() as cur:
        s_buf = StringIO()
        writer = csv.writer(s_buf)
        writer.writerows(data_iter)
        s_buf.seek(0)
        columns = ', '.join('"{}"'.format(k) for k in keys)
        if table.schema:
            table_name = '{}.{}'.format(table.schema, table.name)
        else:
            table_name = table.name
        sql = 'COPY {} ({}) FROM STDIN WITH CSV'.format(
            table_name, columns)
        cur.copy_expert(sql=sql, file=s_buf)

df.to_sql('cian',con_pg,index=False,if_exists='replace',method=psql_insert_copy)
def select(sql):
    with con_pg.connect() as conn:
        return pd.read_sql(sql_text(sql), conn)

def distr(n):
    if n == 1:
        return 'Октябрьский'
    elif n == 2:
        return 'Советский'
    else:
        return 'Железнодорожный'
def distreverse(n):
    if n == 'Октябрьский':
        return 1
    elif n == 'Советский':
        return 2
    else:
        return 3

#Слайдер
values = st.slider(
    'Выберите диапазон цен',
    float(df['price'].min()), float(df['price'].max()), (float(df['price'].min()), float(df['price'].max()))
)

#Вывод отфильтрованного датафрейма
filtered_df = df[(df['price'] >= values[0]) & (df['price'] <= values[1])]
st.dataframe(filtered_df)

#Слайдер графиков
on = st.toggle('Вывести графики')


#Вывод гистограммы
sns.set_theme()
fig_price, ax = plt.subplots()
sns.histplot(filtered_df['price'] / 1e6, bins=50, ax=ax)
ax.grid(True)
ax.set_title('Распределение цен')
ax.set_xlabel('Цена (млн. руб.)')
ax.set_ylabel('Кол-во')


sns.set_theme()
fig_meters, ax = plt.subplots()
sns.histplot(filtered_df['total_meters'], bins=50, ax=ax)
ax.grid(True)
ax.set_title('Распределение кв. м.')
ax.set_xlabel('Кв. м.')
ax.set_ylabel('Кол-во')


sns.set_theme()
fig_floor, ax = plt.subplots()
sns.histplot(filtered_df['floor'], bins=20, ax=ax, binwidth= 1)
ax.grid(True)
ax.set_title('Распределение этажей')
ax.set_xlabel('Этаж')
ax.set_ylabel('Кол-во')
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))


sns.set_theme()
fig_floor, ax = plt.subplots()
sns.histplot(filtered_df['floor'], bins=20, ax=ax, binwidth= 1)
ax.grid(True)
ax.set_title('Распределение этажей')
ax.set_xlabel('Этаж')
ax.set_ylabel('Кол-во')

#Корреляционная матрица
corrdf = (select('''SELECT floor,floors_count,rooms_count,total_meters,
case
when district = 'Октябрьский' then 1
when district = 'Советский' then 2
when district = 'Железнодорожный' then 3
end as district, price
FROM cian'''))
filtered_corrdf = corrdf[(corrdf['price'] >= values[0]) & (corrdf['price'] <= values[1])]
corrmatrix, ax = plt.subplots()
plt.title('Корреляционная матрица')
sns.heatmap(filtered_corrdf.corr(), ax=ax, annot = True)

if on:
    st.pyplot(fig_price)
    st.pyplot(fig_meters)
    st.pyplot(fig_floor)
    st.write(corrmatrix)

#Метрики
st.header('Наиболее типичные характеристики квартиры в выбранном диапазоне цен')
col1, col2, col3, col4 = st.columns(4)
col1.metric(label="Кол-во комнат", value=filtered_df['rooms_count'].median())
col2.metric(label="Кол-во метров", value=filtered_df['total_meters'].median())
col3.metric(label="Район", value=distr(filtered_corrdf['district'].mode()[0]))
col4.metric(label="Цена", value=filtered_df['price'].median())

#Модель
st.header('Узнать рекомендуемую цену')
model = xgb.XGBRegressor()
model.load_model('xgboost_model.json')
с1, с2, с3, с4, с5= st.columns(5)
floor = с1.selectbox("На каком этаже:", select('''select distinct floor from cian order by floor'''))
floors_count = с2.selectbox("Количество этажей:", select('''select distinct floors_count from cian order by floors_count'''))
rooms_count = с3.selectbox("Количество комнат:", select('''select distinct rooms_count from cian order by rooms_count'''))
total_meters = с4.number_input("Количество кв. м.")
district = distreverse(с5.selectbox("Район:", select('''SELECT distinct district
FROM cian''')))

new_data = pd.DataFrame({'floor': floor, 'floors_count': floors_count, 'rooms_count': rooms_count, 'total_meters': total_meters, 'district':district}, index = [0])
predictions = model.predict(new_data)
st.metric(label="Цена", value=predictions)

