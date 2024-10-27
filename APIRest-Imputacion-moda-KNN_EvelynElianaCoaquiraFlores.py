import requests
import csv
import pandas as pd

from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder

URL = "https://api.mediastack.com/v1/news"

api_key = "2fa83844c4d87b909d954103b2511c1e"

params = {
    'sources':'cnn',
    'language':'en',
    'access_key': api_key
}
response = requests.get(URL,params=params)

if response.status_code == 200:

    news_list = response.json()['data']

    with open('news.csv','w',newline='',encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['title','description','category','country','url'])
        writer.writeheader()
        for news in news_list:
            writer.writerow({
                'title':news['title'],
                'description':news['description'],
                'category':news['category'],
                'country':news['country'],
                'url':news['url']
            })

        # Guardar en DataFrame
        news_df = pd.DataFrame(news_list)
        news_df = news_df[['title', 'description','category','country', 'url']]  # Filtrar solo columnas necesarias
        print(news_df)  # Para verificar el DataFrame

        #imputacion por moda
        moda = news_df['category'].mode()[0]
        news_df['category'].fillna(moda, inplace=True)

        ###imputacion knn
        # Codificar variables categóricas para que KNN pueda procesarlas
        df_encoded = news_df.apply(LabelEncoder().fit_transform)

        # Aplicar imputación KNN
        imputer = KNNImputer(n_neighbors=5)
        df_imputed = imputer.fit_transform(df_encoded)

        # Convertir de nuevo a DataFrame
        df_imputed = pd.DataFrame(df_imputed, columns=news_df.columns)

        ###Inputacion de datos knn (columnas especificas)
        # Seleccionar solo las columnas que queremos imputar
        columnas_a_imputar = ['category', 'country']

        # Codificar las columnas cualitativas
        label_encoders = {}
        for col in columnas_a_imputar:
            if news_df[col].dtype == 'object':  # Solo columnas categóricas
                le = LabelEncoder()
                news_df[col] = le.fit_transform(news_df[col].astype(str))
                label_encoders[col] = le

        # Inicializar el imputador KNN y aplicar imputación
        imputer = KNNImputer(n_neighbors=3)
        news_df[columnas_a_imputar] = imputer.fit_transform(news_df[columnas_a_imputar])

        # Si quieres revertir la codificación de columnas categóricas
        for col in label_encoders:
            news_df[col] = label_encoders[col].inverse_transform(news_df[col].astype(int))

        print(news_df)

        ###
        ##imputer = KNNImputer(n_neighbors=5)
        ##df_imputed = imputer.fit_transform(news_df)
else:
    print(f'Error al obtener noticias: ', {response.status_code})