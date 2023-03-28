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


st.sidebar.header('Ingrese los valores de entrada')

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
                        
## 3er Modelo Clustering Clasificación de empleado por clustering



    

# ### Predicting the Existing Employee with the Probability of leaving
st.sidebar.subheader("""Indicar las características del perfil""")
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
    



