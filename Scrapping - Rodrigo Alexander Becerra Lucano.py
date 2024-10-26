import requests
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#Web de libros enfocada en practicar scrapping
base_url= 'https://books.toscrape.com/catalogue/page-{}.html'

df= []

for page in range(1, 6):
    url= base_url.format(page)
    print(f'Extrayendo datos de: {url}')

    #Solicitud a la página
    response= requests.get(url)
    #Verificar solicitud
    response.raise_for_status()

    #Analizar contenido de la página
    soup= BeautifulSoup(response.text, 'html.parser')

    #Extraer elementos con la etiqueta article y la clase product_pod
    libros= soup.find_all('article', class_= 'product_pod')

    i= 1
    for libro in libros:
        titulo= libro.h3.a['title']
        precio= libro.find('p', class_= 'price_color').text
        print(f'Título {i}: {titulo} \nPrecio {i}: {precio}')

        #convertir los datos
        precio= float(precio.lstrip('Â£'))
        
        #agregar los datos al df
        df.append({'Titulo': titulo, 'Precio': precio})
        i+= 1
        
    print('-' * 30)

df= pd.DataFrame(df)
print(df)

#Normalización
scaler= MinMaxScaler()
df['Precio_Norm']= scaler.fit_transform(df[['Precio']])

#Estandarización
scaler= StandardScaler()
df['Precio_St']= scaler.fit_transform(df[['Precio']])

#Binning
rango= [0, 20, 40, 60, 80, 100]
categoria= ['Muy Barato', 'Barato', 'Normal', 'Caro', 'Muy Caro']
df['Precio_Categoria']= pd.cut(df['Precio'], bins= rango, labels= categoria)

print(df)
