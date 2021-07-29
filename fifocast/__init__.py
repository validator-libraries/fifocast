import telebot
from telebot import types
import sys
import webbrowser
from io import BytesIO
import requests
from PIL import Image
import os
import fifocast
import plotly.graph_objects as go
import bs4
from pprint import pprint
from telegraph import Telegraph
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier



def coords_to_address(x, y):
    geocoder_request = f"https://geocode-maps.yandex.ru/1.x/?apikey=40d1649f-0493-4b70-98ba-98533de7710b&geocode={x},{y}&format=json"

    # Выполняем запрос.
    response = requests.get(geocoder_request)

    if response:
        # Преобразуем ответ в json-объект
        json_response = response.json()

        # Получаем первый топоним из ответа геокодера.
        # Согласно описанию ответа, он находится по следующему пути:
        toponym = json_response["response"]["GeoObjectCollection"]["featureMember"][0]["GeoObject"]
        # Полный адрес топонима:
        toponym_address = toponym["metaDataProperty"]["GeocoderMetaData"]["text"]
        # Координаты центра топонима:
        toponym_coodrinates = toponym["Point"]["pos"]
        # Печатаем извлечённые из ответа поля:
        return toponym_address



def address_to_coords(address):
        geocoder_request = f"https://geocode-maps.yandex.ru/1.x/?apikey=40d1649f-0493-4b70-98ba-98533de7710b&geocode={address}&format=json"

    # Выполняем запрос.
        response = requests.get(geocoder_request)
        if response:
        # Преобразуем ответ в json-объект
            json_response = response.json()

        # Получаем первый топоним из ответа геокодера.
        # Согласно описанию ответа, он находится по следующему пути:
            toponym = json_response["response"]["GeoObjectCollection"]["featureMember"][0]["GeoObject"]
        # Полный адрес топонима:
            toponym_address = toponym["metaDataProperty"]["GeocoderMetaData"]["text"]
        # Координаты центра топонима:
            toponym_coodrinates = toponym["Point"]["pos"]
        # Печатаем извлечённые из ответа поля:
            return toponym_coodrinates
        else:
            print("Ошибка выполнения запроса:")
            print(geocoder_request)
            print("Http статус:", response.status_code, "(", response.reason, ")")




def create_static_map(x, y):
    URL = f"https://static-maps.yandex.ru/1.x/?l=sat&ll={x},{y}&pt={x},{y},pm2rdm&spn=0.004,0.004"
    response = requests.get(URL)

    image = Image.open(BytesIO(
    response.content))
    image.save('map_point.png')




def accuracy_generator_img(accuracy):    ### accuracy - процент вероятности
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = accuracy,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "", 'font': {'size': 24}},

        gauge = {
            'axis': {'range': [None, 100]},
             'bar': {'color': "black"},
            'bgcolor': "white",
            'borderwidth': 4,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 10], 'color': 'lightgreen'},
                {'range': [10, 20], 'color': 'green'},
                {'range': [20, 30], 'color': 'darkgreen'},
                {'range': [30, 40], 'color': 'royalblue'},
                {'range': [40, 50], 'color': 'blue'},
                {'range': [50, 60], 'color': 'yellow'},
                {'range': [60, 70], 'color': 'orange'},
                {'range': [70, 80], 'color': 'orangered'},
                {'range': [80, 90], 'color': 'red'},
                {'range': [90, 100], 'color': 'darkred'}],
            'threshold': {
                'line': {'color': "white", 'width': 13},
                'thickness': 0.75,
                'value': accuracy}}))

    fig.update_layout(paper_bgcolor = "black", font = {'color': "white", 'family': "Arial"})

    fig.to_image(format="png", engine="kaleido")
    fig.write_image("accuracy.png")



### функция для преобразования номера месяца в название месяца
def to_month(num_month):
    months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    return months[num_month - 1]

### функция для преобразования даты в день недели
def to_day_week(date):
    month = int(date.split('/')[0])
    day = int(date.split('/')[1])
    year = int(date.split('/')[2])
    today = datetime.datetime(year, month, day)
    num_day_week = today.weekday()
    days_week = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']
    return days_week[num_day_week]



def all_graphs():
    os.mkdir("images_graphs")
    df = pd.read_csv(fifocast.__path__[0] + "\Data.csv")

    ### Переименновываем колонки на англ.
    df.rename(columns={df.columns[0]: "Fire number", df.columns[1]: "Date",
                   df.columns[4]: "Area", df.columns[5]: "Temp", df.columns[6]: "Rel_hum",
                   df.columns[7]: "Soil_moist", df.columns[8]: "Atm_pres", df.columns[9]: "V_type"}, inplace = True)

    ### График кол-ва пожаров каждого типа растительности
    V_type_graph = sns.countplot(x=df['V_type'], hue=df['V_type']);

    fig = V_type_graph.get_figure()      ### сохраняем график
    fig.savefig(os.getcwd() + '\images_graphs\V_type_graph.png')
    plt.clf()

    ### Добавляем новый признак (номер месяца)
    # date.split('/')[0] - номер месяца
    df['Num_month'] = df['Date'].apply(lambda date: int(date.split('/')[0]))


    ### График кол-ва пожаров по месяцам каждой растительности
    ### to_month(num_month) - преобразуем номер в название месяца

    V_type_graph_month = sns.countplot(x=df['Num_month'].apply(lambda num_month: to_month(num_month)), hue=df['V_type']);

    fig1 = V_type_graph_month.get_figure()    ### сохраняем график
    fig1.savefig(os.getcwd() + '\images_graphs\V_type_graph_month.png')
    plt.clf()


    ### График кол-ва пожаров по дням недели каждой растительности
    ### to_day_week(date) - преобразуем дату в название дня недели
    V_type_graph_week = sns.countplot(x=df['Date'].apply(lambda date: to_day_week(date)), hue=df['V_type']);

    fig2 = V_type_graph_week.get_figure()    ### сохраняем график
    fig2.savefig(os.getcwd() + '\images_graphs\V_type_graph_week.png')
    plt.clf()


def fire_area(date, X, Y, temp, humidity, soil_moisture, pressure_pa, V_type):

    ### Данные за 2020 год
    df = pd.read_csv(fifocast.__path__[0] + "\Data.csv")


    ### Переименовываем колонки на англ. яз.
    df.rename(columns={df.columns[0]: "Fire number", df.columns[1]: "Date",
                   df.columns[4]: "Area", df.columns[5]: "Temp", df.columns[6]: "Rel_hum",
                   df.columns[7]: "Soil_moist", df.columns[8]: "Atm_pres", df.columns[9]: "V_type"}, inplace = True)

    ### Удаляем незначительные признаки и меняем значения признака V_type
    df.drop(['Fire number', 'X', 'Y', 'Date'], axis=1, inplace=True)

    df = df[(df['V_type'] == V_type)]      ### оставляем только определенные пожары

    df['V_type'] = df['V_type'].map({'Луг': 0, 'Лиственный лес': 1, 'Темнохвойный лес': 2})


    # Создание модели для прогнозирования
    ### Data Frame Test модель
    Data= [['1', date, X, Y, '0', temp, humidity, soil_moisture, pressure_pa, V_type]]
    df_test = pd.DataFrame(Data, columns=["Номер пожара","Дата","X","Y","Площадь, га","температура, °С","относительная влажность, %","влажность почвы (в слое 0-10см), %","атмосферное давление, мбар","Тип растительности"])

    df_test.rename(columns={df_test.columns[0]: "Fire number", df_test.columns[1]: "Date",
                   df_test.columns[4]: "Area", df_test.columns[5]: "Temp", df_test.columns[6]: "Rel_hum",
                   df_test.columns[7]: "Soil_moist", df_test.columns[8]: "Atm_pres", df_test.columns[9]: "V_type"}, inplace = True)

    df_test.drop(['Fire number', 'X', 'Y', 'Date'], axis=1, inplace=True)
    df_test['V_type'] = df_test['V_type'].map({'Луг': 0, 'Лиственный лес': 1, 'Темнохвойный лес': 2})

    y_train = df['Area'].astype('int')
    X_train = df.drop('Area', axis=1)
    y_valid = df_test['Area'].astype('int')
    X_valid = df_test.drop('Area', axis=1)

    ### Модель

    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X_train, y_train)


    area_fire_pred = model.predict(X_valid)[0]

    if area_fire_pred < 0:
        area_fire_pred = 0

    accuracy = model.score(X_train, y_train)  ### accuracy пожара

    return (area_fire_pred, accuracy)





def telegraph_article(address, v_type):
        longitude, latitude = address_to_coords(address).split(' ')



        headers = {'X-Yandex-API-Key': 'a141c1f3-a2b3-4d92-83d2-662ae921db9d'}

        link = f'https://api.weather.yandex.ru/v2/forecast?lat={latitude}&lon={longitude}&extra=true'

        response = requests.get(link, headers = headers, verify = True)

        json_response = response.json()

        # температура воздуха
        temp = json_response['fact']['temp']

        # температура почвы
        soil_temp = json_response['fact']['soil_temp']

        # влажность почвы %
        soil_moisture = int(float(json_response['fact']['soil_moisture']) * 100)

        # относительная влажность %
        humidity = json_response['fact']['humidity']


        # атмосферное давление
        pressure_pa = json_response['fact']['pressure_pa']


        # дата (Месяц/День/Год)
        date_now = datetime.datetime.now()
        date = str(date_now.month) + '/' + str(date_now.day) + '/' + str(date_now.year)




        ### Площадь пожара и точностть прогноза
        f_a, accuracy_fire = fire_area(date, latitude, longitude, temp, humidity, soil_moisture, pressure_pa, v_type)
        f_a = int(f_a)                      ###  переводим площадь в целочисленный вид
        accuracy_fire = int(accuracy_fire * 100)    ### точность прогноза переводим в проценты
        accuracy_generator_img(accuracy_fire)     # создаем спидометр





        # создаем телеграф статью для описания прогноза
        telegraph = Telegraph()

        telegraph.create_account(short_name='Valy')

        with open(os.path.join(fifocast.__path__[0] + '/forest_fire.jpg'), 'rb') as f:     ### фотка банера
            path_ban = requests.post(
                    'https://telegra.ph/upload', files={'file':
                                                        ('file', f,
                                                        'image/jpg')}).json()[0]['src']



        with open('accuracy.png', 'rb') as f:     ### фотка спидометра
            path_spidom = requests.post(
                    'https://telegra.ph/upload', files={'file':
                                                        ('file', f,
                                                        'image/png')}).json()[0]['src']


        create_static_map(longitude, latitude)   ### Создаем карту точки геолокации
        with open('map_point.png', 'rb') as f:   ### карта геолокации точки
            map_point = requests.post(
                    'https://telegra.ph/upload', files={'file':
                                                        ('file', f,
                                                        'image/png')}).json()[0]['src']


        all_graphs()  ### Создаем графики
        with open(os.getcwd() + '\images_graphs\V_type_graph.png', 'rb') as f:   ### фотка графика V_type кол-ва пожаров каждой растительности
            V_type_graph = requests.post(
                    'https://telegra.ph/upload', files={'file':
                                                        ('file', f,
                                                        'image/png')}).json()[0]['src']


        with open(os.getcwd() + '\images_graphs\V_type_graph_month.png', 'rb') as f:  ### фотка графика V_type кол-ва пожаров за каждый месяц
            V_type_graph_month = requests.post(
                    'https://telegra.ph/upload', files={'file':
                                                        ('file', f,
                                                        'image/png')}).json()[0]['src']


        with open(os.getcwd() + '\images_graphs\V_type_graph_week.png', 'rb') as f:  ### фотка графика V_type кол-ва пожаров за каждый день недели
            V_type_graph_week = requests.post(
                    'https://telegra.ph/upload', files={'file':
                                                        ('file', f,
                                                        'image/png')}).json()[0]['src']





        ### Создаем статью в Телеграфе
        response = telegraph.create_page(
            f'Прогноз пожара',
            html_content="<img src='{}'/>".format(path_ban) + f"<h3>Точность прогноза {accuracy_fire}%</h3>" + "<img src='{}'/>".format(path_spidom) +

                  f"<h3>Вероятная площадь пожара: {f_a} га</h3> \
                  <h3>Содержание</h3> \
                  <ul><li><a href='#Текущие-метеоданные'>Текущие метеоданные</a></li> \
                  <li><a href='#Пожары-по-типам-растительности'>Пожары по типам растительности</a></li> \
                  <li><a href='#Пожары-по-типам-растительности-за-каждый-месяц'>Пожары по типам растительности за каждый месяц</a></li> \
                  <li><a href='#Пожары-по-типам-растительности-за-каждый-день-недели'>Пожары по типам растительности за каждый день недели</a></li></ul> \
                  \
                  <h3>Текущие метеоданные</h3> \
                  <p><em><b><u>Адрес: </u></b></em>{coords_to_address(longitude, latitude)}</p>" +
                  "<img src='{}'/>".format(map_point) +
                  f"<p>Температура воздуха: {temp}°</p> \
                  <p>Температура почвы: {soil_temp}°</p> \
                  <p>Влажность почвы: {soil_moisture}%</p>\
                  <p>Относительная влажность: {humidity}%</p>\
                  <p>Атмосферное давление: {pressure_pa}</p>\
                  <p>Точность прогноза: {accuracy_fire}%</p>\
                  \
                  <h3>Пожары по типам растительности</h3>" +
                  "<img src='{}'/>".format(V_type_graph) +

                  "<h3>Пожары по типам растительности за каждый месяц</h3>" +
                  "<img src='{}'/>".format(V_type_graph_month) +

                  "<h3>Пожары по типам растительности за каждый день недели</h3>" +
                  "<img src='{}'/>".format(V_type_graph_week)
        )

        link_article = 'https://telegra.ph/{}'.format(response['path'])
        print(link_article)
        webbrowser.open(link_article)