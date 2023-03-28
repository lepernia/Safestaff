#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 19:57:11 2023

@author: leonardoperniaespinoza
"""

import sklearn.preprocessing as preprocessing
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder #Para cambiar una variable categorica en entero.
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier as knn
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn import svm

# #### Modelo predictivo 1 - entrenamiento abandono ####

# Importamos la data
data_original=pd.read_csv("HR_comma_sep.csv",sep=";")

# Y visualizamos los 5 primero registros del data set para inspeccionar las variables.
data_original.head()

# Partimos el data set en dos partes.
# Una que tenga 1000 empleados que se han ido y 3000 que se han quedado. Particiones aleatorias.
# La otra es el resto de lo que no se ha cogido en la particion anterior.
import numpy as np

# Seleccionar las filas con left=1 y left=0
tratado_left_1 = data_original.query('left == 1')
tratado_left_0 = data_original.query('left == 0')

# Elegir 1000 filas de la categoria left=1 y 3000 filas de left=0
np.random.seed(42)
parte_1 = pd.concat([tratado_left_1.sample(1000,random_state = 42),tratado_left_0.sample(3000,random_state = 42)])

# El resto
parte_2 = data_original.loc[~data_original.index.isin(parte_1.index)]

# Convertimos las variables categoricas en variables numericas.
df=parte_1 #renombramos los datos que vamos a trabajar a "df"

df["salary"] = pd.factorize(df["salary"])[0]
df["Department"] = pd.factorize(df["Department"])[0]

# Separamos la variable dependiente que queremos predecir del resto de variables.
# Indicamos que columna es nuestro target....
target = df.left
print(target)

# ...y cogemos el resto de variables, quitando la columna left
features = df.drop("left", axis=1)
print(features)

# Aplicamos PCA para reducir la dimensionalidad
from sklearn.decomposition import PCA
pca = PCA(n_components=5)
pca.fit(features)
features_pca = pca.transform(features)

# Dividimos los datos en los subgrupos  training, test y validacion: las "x" seran los datos de las caracteristicas y los "y" los datos del target.
# x_train= datos para training de features
# y_train= datos para training de target
# x_test= datos para test de features
# y_test= datos

y_train, y_test,x_train,x_test = train_test_split(target, features,test_size = 0.20, random_state = 0,stratify=target) 

# In[]

#Construimos el modelo de regresion logistica

from sklearn import linear_model

regresion= linear_model.LogisticRegression()
regresion.fit(x_train,y_train)

# In[]

#Utilizamos cross_val_score para evaluar nuestro modelo. La metrica usada es accuray y nos da una precision del 100% en cada evaluacion. Le hemos indicado que realice  
# la muestra con datos aleatorio 5 veces. Parece que el modelo se ha aprendido de memoria los datos por lo que hay un claro ejemplo de overfitting.

precision_regresion = cross_val_score(regresion,x_test,y_test,cv=5)
precision_regresion.mean()

# In[]

# Inicializar el modelo de gradient boosting
from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier()

# In[]

# Ajustar el modelo con los datos de entrenamiento
gb.fit(x_train,y_train)

# In[]

# Evaluar el modelo con los datos de prueba
accuracy = gb.score(x_test, y_test)
print("Accuracy: ", accuracy)

# In[]

predict_y_test=gb.predict_proba(x_test)

# In[]

#Probamos el modelo con la parte que hemos separado del data set data_original. 
#Lo llamamos "df2" y hacemos el mismo tratamiento que al que hemos usado para crar 
#el modelo.

df2=parte_2

df2["salary"] = pd.factorize(df2["salary"])[0]
df2["Department"] = pd.factorize(df2["Department"])[0]

#Separamos el target del resto de caracteristicas independientes.
target2 = df2.left

features2 = df2.drop("left", axis=1)

# In[]
#Creamos la prediccion del target de features2 que son las caracteristicas de la data no usada en el entrenamiento.

proba_features2 = gb.predict_proba(features2)

# In[]

# Evaluar el modelo con los datos de prueba
accuracy = gb.score(features2, target2)
print("Accuracy: ", accuracy)


# In[] #### Modelo predictivo 2 - predicción tiempo de abandono #### 

## Importamos las librerías necesarias
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Leemos el conjunto de datos
df8 = pd.read_csv('HR_comma_sep.csv', sep = ';')

# Seleccionamos las variables predictoras y la variable objetivo
X8 = df8[['satisfaction_level', 'last_evaluation', 'number_project','average_montly_hours', 'Work_accident', 'promotion_last_5years', 'salary']]
y8 = df8['time_spend_company']

# Convertimos la variable salary en variables ficticias (dummies)
X8 = pd.get_dummies(X8, columns=['salary'])

# Dividimos los datos en conjunto de entrenamiento y prueba
X8_train, X8_test, y8_train, y8_test = train_test_split(X8, y8, test_size=0.2, random_state=42)

# Creamos el modelo de regresión lineal y lo entrenamos con los datos de entrenamiento
reg = LinearRegression().fit(X8_train, y8_train)

# Hacemos predicciones con los datos de prueba
y8_pred = reg.predict(X8_test)

# Evaluamos el modelo
print('Error cuadrático medio: %.2f' % mean_squared_error(y8_test, y8_pred))
print('Coeficiente de determinación: %.2f' % r2_score(y8_test, y8_pred))


# Leemos el conjunto de datos


# In[]
# comenzamos con el desarrollo de la app 

import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
import seaborn as sns
import xlrd

#### Tile For the Web App
# Mostrar el gráfico en Streamlit con un tamaño reducido
image = Image.open("logo_safestaff.png")

new_size = (500,200)
image = image.resize(new_size)
st.image(image,use_column_width=True)

st.write("""
         # Aplicación para predecir el abandono del personal de tu organización #
         """)
      
if st.button("Quienes somos"):
    image2 = Image.open("Logo_EAE.png")
    new_size2 = (150,50)
    image2 = image2.resize(new_size2)
    st.image(image2,use_column_width=False)
    st.write("""
        Name: Grupo de creadores - Proyecto EAE Business School
            
        Huwen Ely Armone Petrovich 
        
        Laura María Extremera Díez
        
        Silvia Patricia Fernández Jaudenes
        
        Diego Ortíz Boyano 
        
        Leonardo Pernía Espinoza
         """)
## 
st.header(
   """Clasificación empleados""")

# In[18]:

np.random.seed(0)

#load the datasets for Employee who has left.
#left = pd.read_excel("Hash-Analytic-Python-Analytics-Problem-case-study-1.xlsx",sheet_name=2)
#left.head(4)

left = pd.read_csv("employee_who_left.csv")
left.head(4)

#load the datasets for Employee who is still existing.
#existing = pd.read_excel("Hash-Analytic-Python-Analytics-Problem-case-study-1.xlsx",sheet_name=1)

existing = pd.read_csv("existing_employee.csv")

## Add the atrribute Churn to Existing Employeee dataset
existing['Churn']= 'No'
existing.head(2)

## Add the attribute churn to Employee who has left dataset
left['Churn']='Yes'
left.head(2)

## Chart
fig1,ax= plt.subplots(figsize=(12,5))
ax.bar(left['dept'].value_counts().index,left['dept'].value_counts().values)
plt.ylabel("Nro. empleados que han abandonado la organización")
plt.xlabel("Departamentos")
plt.title("Volumen de empleados por departamento")
plt.grid()
st.pyplot(fig1)

from sklearn.metrics import accuracy_score, classification_report

# #### b) Random forest

from sklearn.ensemble import RandomForestClassifier

## Trainign the Model
rforest = RandomForestClassifier()
rforest.fit(x_train,y_train)

#rf = RandomForestClassifier()
#rf.fit(X9_train,y9_train)

## Testing the Model
y_pred_rforest = rforest.predict(x_train)

#Cambiamos la variable categorida ordinal (salary) por 1 (low), 2 (medium) y 3 (high)
# creamos un diccionario para mapear los valores de la columna categorica salario con sus valores binarios

mapeo = {'low': 1, 'medium': 2, 'high': 3}

df['salary'] = df['salary'].map(mapeo)

df.replace({'salary': mapeo})

df.head(5)

# Para las categoricas no ordinales crearemos variables dummies usando one-hot encoding

dummys = pd.get_dummies(df.Department, prefix='Department')

dummys.head()
                      
################ 3er Modelo Clustering Clasificación de empleado por clustering

# In[] ### 3er. Modelo Kmeans - clustering 

st.header(
   """Atributos con mayor relación""")
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import streamlit as st

# carga de los datos

df_clust = pd.read_csv('HR_comma_sep.csv', sep=";")
df_clust.head(5)
    
#Vamos a reducir el número de variables agrupando technical, support e IT bajo el mismo nombre de IT

df_clust['Department'] = df_clust['Department'].replace(['technical', 'support'], 'IT')

#Cambiamos la variable categorida ordinal (salary) por 1 (low), 2 (medium) y 3 (high)
# creamos un diccionario para mapear los valores de la columna categorica salario con sus valores binarios

mapeo = {'low': 1, 'medium': 2, 'high': 3}

df_clust['salary'] = df_clust['salary'].map(mapeo)

df_clust.replace({'salary': mapeo})

df_clust.head(5)

# Para las categoricas no ordinales crearemos variables dummies usando one-hot encoding

dummys = pd.get_dummies(df.Department, prefix='Department')

dummys.head()

# unimos nuestro df original con las nuevas feature dummies creadas y eliminamos la antigua columna de departamento

df_clust = pd.concat([df_clust, dummys], axis = 1)

df_clust.drop('Department', axis = 1)

#df_clust.columns.values
#df_clust.drop('Department', axis = 1)

# Vamos a comenzar por seleccionar las columnas que utilizaremos en la segmentación de empleados
import seaborn as sns
sns.set()
from sklearn.preprocessing import StandardScaler
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


df_clust = pd.read_csv('HR_comma_sep.csv', sep=";")

cluster_data = df_clust[['satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours', 'time_spend_company', 'Work_accident', 'promotion_last_5years', 'salary']]
#plt.figure(figsize=(12,9))
fig1 = sns.heatmap(cluster_data.corr(),annot=True)
plt.title('Correlation Heatmap',fontsize=14)
plt.yticks(rotation =0)

st.pyplot(fig1.figure)


#data = pd.read_csv('cluster_data.csv')
#corr_matrix = data.corr()
#fig = sns.heatmap(corr_matrix)

# =============================================================================
# # Para K Means aplicamos una estandarizacion a los valores de nuestro dataset con el objetivo de que todas las variables tengan el mismo peso
# 
# scaler = StandardScaler()
# 
# clusters_std= scaler.fit_transform(cluster_data)
# 
# # añadimos los titulos de las columna
# 
# clusters_std = pd.DataFrame(data = clusters_std, columns = cluster_data.columns)
# 
# # Para saber el numero optimo de clusters usamos el metodo del codo
# 
# WCSS = []
# for i in range(1,20):
#     model = KMeans(n_clusters = i,init = 'k-means++')
#     model.fit(clusters_std)
#     WCSS.append(model.inertia_)
# fig = plt.figure(figsize = (7,7))
# plt.plot(range(1,20),WCSS, linewidth=4, markersize=12,marker='o',color = 'green')
# plt.xticks(np.arange(20))
# plt.xlabel("Number of clusters")
# plt.ylabel("WCSS")
# plt.show()
# 
# 
# #Configuramos nuestro k means
# 
# kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
# kmeans = kmeans.fit(clusters_std)
# cluster_data.loc[:,"cluster"] = kmeans.labels_
# cluster_data.head(5)
# 
# # Para obtener los centroides del cluster utilizamos el siguiente método
# 
# centroides = kmeans.cluster_centers_
# 
# # Este método nos devuelve un array de dimensiones NxK, siendo N el número de atributos y k el número de clasuters. 
# #centroides
# 
# # Construimos una tabla para conocer mejor las características del centroide de cada cluster.
# 
# indices=['centroide_1','centroide_2','centroide_3','centroide_4', 'centroide_5']
# atributos = ['satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours', 'time_spend_company', 'Work_accident', 'promotion_last_5years', 'salary']
# df_clust=pd.DataFrame(data=centroides,index=indices,columns= [atributos])
# 
# cluster_analysis = cluster_data.groupby(['cluster']).mean()
# cluster_analysis
# 
# #vamos a hacer PCA
# #Usamos el fragmento del dataset sin estandarizar valores
# 
# from sklearn.decomposition import PCA
# 
# pca = PCA(n_components = 2)
# cluster_data_pca = pd.DataFrame(pca.fit_transform(cluster_data_org))
# 
# cluster_data_pca
# 
# clusters_pca_components = pd.DataFrame(pca.components_, columns = atributos)
# clusters_pca_components
# 
# # el PCA nos muestra que las dos features que más correlación producen son las horas mensuales trabajadas (variable 0) 
# # y el nivel de satisfacción (1)
# 
# # Utilizamos el método del codo para determinar el número óptimo de clusters
# 
# # =============================================================================
# WCSS = []
# for i in range(1,20):
#    model = KMeans(n_clusters = i,init = 'k-means++')
#    model.fit(cluster_data_pca)
#    WCSS.append(model.inertia_)
# # fig = plt.figure(figsize = (7,7))
# # plt.plot(range(1,20),WCSS, linewidth=4, markersize=12,marker='o',color = 'green')
# # plt.xticks(np.arange(20))
# # plt.xlabel("Number of clusters")
# # plt.ylabel("WCSS")
# # plt.show()
# # 
# # 
# # #Probaremos k means con 4 clusters
# # 
# # # Primero hacemos el fit de KMean
# model = KMeans(n_clusters=4)
# model.fit(cluster_data_pca)
# # 
# # # Nos guardamos en labels las etiquetas de cada clusters
# # labels = model.labels_
# # 
# # # Plot the results
# cluster_data_pca_np = cluster_data_pca.values
# plt.scatter(cluster_data_pca_np[:, 0], cluster_data_pca_np[:, 1], c=labels)
# plt.xlabel("First Principal Component")
# plt.ylabel("Second Principal Component")
# # 
# centroides_pca = model.cluster_centers_
# # 
# # # Construimos una tabla para conocer mejor las características del centroide de cada cluster.
# indices=['centroide_1','centroide_2','centroide_3','centroide_4']
# atributos = ['pca_0','pca_1']
# df_clust=pd.DataFrame(data=centroides_pca,index=indices,columns= [atributos])
# df_clust
# # =============================================================================
# =============================================================================



##########################


# ### Predicting the Existing Employee with the Probability of leaving
# Mostrar el gráfico en Streamlit con un tamaño reducido

st.sidebar.image("logo_safestaff.png", use_column_width=True)
st.sidebar.header('Ingrese los valores de entrada')    
st.sidebar.subheader("""Indicar las características del perfil""")
#### Tile For the Web App

def user_input():
    satisfaction_level = st.sidebar.number_input("Nivel de satisfacción",min_value=1, max_value= 5,value=2)
    last_evaluation = st.sidebar.number_input("Última evaluación",min_value=1, max_value= 5,value=3)
    number_project =st.sidebar.number_input('Número de proyectos',min_value=0, max_value= 10,value=5)
    average_montly_hours = st.sidebar.number_input('Promedio horas trabajadas al mes',min_value=0, max_value= 350,value=160)
    time_spend_company  = st.sidebar.number_input('Antiguedad',min_value=0, max_value= 20,value=5)
    Work_accident =st.sidebar.selectbox('Accidentes laborales',(0, 1))
    promotion_last_5years = st.sidebar.selectbox('Promociones últimos 5 años',(0, 1))
    department = st.sidebar.selectbox('Departamento',("Ventas","Técnico","Soporte","IT","RRHH","Contabilidad","Marketing","Producción","randD","Administración"))
    salary =  st.sidebar.selectbox('Nivel salarial',("low","medium","high"))
 
    
    ### Dictionaries of Input
    input_user = {"Nivel de satisfacción":satisfaction_level ,"Última evaluación":last_evaluation, "Número de proyectos":number_project,"Promedio de horas trabajadas al mes":average_montly_hours,"Antiguedad":time_spend_company,"Accidentes laborales":Work_accident,"Promociones en los últimos 5 años":promotion_last_5years, "Departamento":department,"Nivel salarial":salary}
               
    ### Converting to a Dataframes
    input_user = pd.DataFrame(input_user,index=[0])
    return input_user

input_value = user_input()                               

print(input_value.info())
        
# Label Encoding will be used for columns with 2 or less unique values

## Encoding The  Categorical Variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le1= LabelEncoder()

le1_count = 0
for col in input_value.columns:
    if input_value[col].dtypes == 'object':
        le1.fit(input_value[col])
        input_value[col] = le1.transform(input_value[col])
        le1_count += 1


print('{} columns were label encoded.'.format(le1_count))

Prediction2 = reg.predict(input_value)
if st.sidebar.button("Predecir"):
    Prediction = rforest.predict(input_value)
    
    if Prediction == 0:
        result = pd.DataFrame({"Abandono?":Prediction,"Info":"El empleado no dejaría la organización"})
        #result2 = pd.DataFrame({"Info": "El empleado dejará la organización en", "Años": Prediction2 })
    else:   
        result = pd.DataFrame({"Abandono?":Prediction,"Info":"El empleado dejará la organización"})                      
        
        result2 = pd.DataFrame({"Info": "El empleado dejará la organización en", "Años": Prediction2 })
    st.write("""
             # Resultado de la clasificación:
                 """)
    st.write(" Desgaste: ")  
    if Prediction == 0:
        st.dataframe(result)
    else:
        st.dataframe(result)
        st.dataframe(result2)
    



