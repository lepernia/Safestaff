#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 12:50:21 2023

@author: leonardoperniaespinoza
"""

# Import de todas las bibliotecas 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import streamlit as st
from PIL import Image
import xlrd

#SKlearn
import sklearn.preprocessing as preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder #Para cambiar una variable categorica en entero.
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.metrics import accuracy_score
from sklearn import metrics   
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn import linear_model
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
sns.set()
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


#Import de ficheros
file_import = pd.read_csv("P&O historico empresa SSA.csv",sep=";")
file_left = pd.read_csv("employee_who_left.csv")
file_left.head(4)
file_existing = pd.read_csv("existing_employee.csv")


########### MODELO 1 entrenamiento abandono ##################
file_import.head()

# Partimos el data set en dos partes.
# Una que tenga 1000 empleados que se han ido y 3000 que se han quedado. Particiones aleatorias.
# La otra es el resto de lo que no se ha cogido en la particion anterior.

# Seleccionar las filas con left=1 y left=0
tratado_left_1 = file_import.query('left == 1')
tratado_left_0 = file_import.query('left == 0')

# Elegir 1000 filas de la categoria left=1 y 3000 filas de left=0
np.random.seed(42)
parte_1 = pd.concat([tratado_left_1.sample(1000,random_state = 42),tratado_left_0.sample(3000,random_state = 42)])

# El resto 11000 registro
parte_2 = file_import.loc[~file_import.index.isin(parte_1.index)]

# Convertimos las variables categoricas en variables numericas.
df = parte_1 #renombramos los datos que vamos a trabajar a "df"
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

pca = PCA(n_components=5)
pca.fit(features)
features_pca = pca.transform(features)

# Dividimos los datos en los subgrupos  training, test y validacion: las "x" 
#seran los datos de las caracteristicas y los "y" los datos del target.
y_train, y_test,x_train,x_test = train_test_split(target, features,test_size = 0.20, random_state = 0,stratify=target) 

# In[]
#Construimos el modelo de regresion logistica

regresion= linear_model.LogisticRegression()
regresion.fit(x_train,y_train)

# In[]
#Utilizamos cross_val_score para evaluar nuestro modelo. La metrica usada es 
#accuray y nos da una precision del 100% en cada evaluacion. Le hemos indicado que realice  
# la muestra con datos aleatorio 5 veces. Parece que el modelo se ha aprendido de memoria 
#los datos por lo que hay un claro ejemplo de overfitting.
precision_regresion = cross_val_score(regresion,x_test,y_test,cv=5)
precision_regresion.mean()


# In[]
# Inicializar el modelo de gradient boosting
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
#Probamos el modelo con la parte que hemos separado del data set dataframe "file_import". 
#Lo llamamos "df2" y hacemos el mismo tratamiento que al que hemos usado para crar 
#el modelo.
df2=parte_2
# =============================================================================
df2["salary"] = pd.factorize(df2["salary"])[0]
df2["Department"] = pd.factorize(df2["Department"])[0]
# 
#Separamos el target del resto de caracteristicas independientes.
target2 = df2.left
features2 = df2.drop("left", axis=1)
# 
# In[]
#Creamos la prediccion del target de features2 que son las caracteristicas de la data no usada en el entrenamiento.
proba_features2 = gb.predict_proba(features2)
# 
# In[]
# Evaluar el modelo con los datos de prueba
accuracy = gb.score(features2, target2)
print("Accuracy: ", accuracy)
# =============================================================================


# In[18]:
##### se comienza a evaluar los resultados RFOREST ########
np.random.seed(0)

#load the datasets for Employee who has left.
#left = pd.read_excel("Hash-Analytic-Python-Analytics-Problem-case-study-1.xlsx",sheet_name=2)
#left.head(4)

left = features2 #pd.read_csv("employee_who_left.csv")
left.head(4)

#load the datasets for Employee who is still existing.
#existing = pd.read_excel("Hash-Analytic-Python-Analytics-Problem-case-study-1.xlsx",sheet_name=1)

existing = features #pd.read_csv("existing_employee.csv")

## Add the atrribute Churn to Existing Employeee dataset
existing['Churn']= 'No'
existing.head(2)

## Add the attribute churn to Employee who has left dataset
left['Churn']='Yes'
left.head(2)



# #### b) Random forest
## Trainign the Model
rforest = RandomForestClassifier()
rforest.fit(x_train,y_train)

#rf = RandomForestClassifier()
#rf.fit(X9_train,y9_train)

## Testing the Model
y_pred_rforest = rforest.predict(x_train)

# =============================================================================
# #Cambiamos la variable categorida ordinal (salary) por 1 (low), 2 (medium) y 3 (high)
# # creamos un diccionario para mapear los valores de la columna categorica salario con sus valores binarios
# 
# mapeo = {'low': 1, 'medium': 2, 'high': 3}
# 
# df['salary'] = df['salary'].map(mapeo)
# 
# df.replace({'salary': mapeo})
# 
# df.head(5)
# =============================================================================

# Para las categoricas no ordinales crearemos variables dummies usando one-hot encoding

# dummys = pd.get_dummies(df.Department, prefix='Department')

# dummys.head()

########### MODELO 3 Kmeans - Clustering ##############################

# carga de los datos
#df_clust = pd.read_csv('HR_comma_sep.csv', sep=";")
#df_clust.head(5)
    
# =============================================================================
# #Vamos a reducir el número de variables agrupando technical, support e IT bajo el mismo nombre de IT
# df_clust = file_import
# df_clust['Department'] = df_clust['Department'].replace(['technical', 'support'], 'IT')
# 
# #Cambiamos la variable categorida ordinal (salary) por 1 (low), 2 (medium) y 3 (high)
# # creamos un diccionario para mapear los valores de la columna categorica salario con sus valores binarios
# # mapeo = {'low': 1, 'medium': 2, 'high': 3}
# # df_clust['salary'] = df_clust['salary'].map(mapeo)
# # df_clust.replace({'salary': mapeo})
# # df_clust.head(5)
# df_clust["salary"] = pd.factorize(df_clust["salary"])[0]
# df_clust"Department"] = pd.factorize(df_clust["Department"])[0]
# 
# # Para las categoricas no ordinales crearemos variables dummies usando one-hot encoding
# # dummys = pd.get_dummies(df.Department, prefix='Department')
# # dummys.head()
# 
# # unimos nuestro df original con las nuevas feature dummies creadas y 
# #eliminamos la antigua columna de departamento
# df_clust = pd.concat([df_clust, dummys], axis = 1)
# df_clust.drop('Department', axis = 1)
# 
# #df_clust.columns.values
# #df_clust.drop('Department', axis = 1)
# =============================================================================

# Vamos a comenzar por seleccionar las columnas que utilizaremos en la segmentación de empleados
df_clust = file_import#pd.read_csv('HR_comma_sep.csv', sep=";")
cluster_data = df_clust[['satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours', 'time_spend_company', 'Work_accident', 'promotion_last_5years', 'salary']]
# #Vamos a reducir el número de variables agrupando technical, support e IT bajo el mismo nombre de IT
df_clust["Department"] = pd.factorize(df_clust["Department"])[0]
#plt.figure(figsize=(12,9))
fig1 = sns.heatmap(cluster_data.corr(),annot=True)
plt.title('Relación de atributos',fontsize=10)
plt.yticks(rotation =0)

########### MODELO 2 entrenamiento tiempo de abandono ########

# Leemos el conjunto de datos
df3 = file_import  #pd.read_csv('HR_comma_sep.csv', sep = ';')

# Seleccionamos las variables predictoras y la variable objetivo
X3 = df3[['satisfaction_level', 'last_evaluation', 'number_project','average_montly_hours', 'time_spend_company','Work_accident', 'promotion_last_5years', 'Department', 'salary', ]]
y3 = df3['time_spend_company']

# Convertimos la variable salary en variables ficticias (dummies)
#X3 = pd.get_dummies(X3, columns=['salary'])
X3["salary"] = pd.factorize(X3["salary"])[0]

# Dividimos los datos en conjunto de entrenamiento y prueba
X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size=0.2, random_state=42)

# Creamos el modelo de regresión lineal y lo entrenamos con los datos de entrenamiento
reg = LinearRegression().fit(X3_train, y3_train)

# Hacemos predicciones con los datos de prueba
y3_pred = reg.predict(X3_test)

# Evaluamos el modelo
print('Error cuadrático medio: %.2f' % mean_squared_error(y3_test, y3_pred))
print('Coeficiente de determinación: %.2f' % r2_score(y3_test, y3_pred))

################ APP SAFESTAPP PEOPLE ANALITYCS ######################## 

#cuerpo principal
#### Tile For the Web App
# Mostrar el gráfico en Streamlit con un tamaño reducido
image = Image.open("logo_safestaff.png")

new_size = (500,200)
image = image.resize(new_size)
st.image(image,use_column_width=True)

st.write("""
         # Aplicación para predecir el abandono del personal de tu organización #
         """)  

# Inicializar la variable auxiliar
button_pressed = False

# Crear boton "Quienes somos" 

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

# lateral
st.sidebar.image("logo_safestaff.png", use_column_width=True)
data_up= st.sidebar.file_uploader('Upload file', type="csv")
st.sidebar.header('Ingrese los valores de entrada')    
st.sidebar.subheader("""Indicar las características del perfil""")

##
if data_up is not None:
    st.header(
   """Atributos con mayor relación""")
    st.pyplot(fig1.figure) # grafico de correlación en el cuerpo de la app

## 
    st.header(
   """Clasificación empleados"""
   )
# =============================================================================
if data_up is not None:
# ## Chart

#     fig1,ax= plt.subplots(figsize=(12,5))
#     ax.bar(left['Department'].value_counts().index,left['Department'].value_counts().values)
   
#     # Define las etiquetas para cada valor del eje X
#     etiquetas = ["Ventas","Técnico","Soporte","IT","RRHH","Contabilidad","Marketing","Producción","randD","Administración"]

# # Utiliza el método set_xticklabels() para asignar las etiquetas a los valores del eje X
#     ax.set_xticklabels(etiquetas)

# # Rota las etiquetas en caso de que sean muy largas para que se ajusten correctamente
#     plt.xticks(rotation=90)

    
    # plt.ylabel("Nro. empleados que han abandonado la organización")
    # plt.xlabel("Departamentos")
    # plt.title("Volumen de empleados por departamento")
    # plt.grid()
    # st.pyplot(fig1)
    

    import plotly.express as px

    data_attrition = pd.read_csv("existing_employee.csv",sep=",")

    fig55 = px.bar(data_attrition, x="dept", y="satisfaction_level", color='dept', barmode='stack')

    fig55.update_layout(title='Recurrencia e Abandono por Departamento', xaxis_title='Departamento', yaxis_title='Recurrencia de Abandono')

    st.plotly_chart(fig55)

# =============================================================================

def user_input():
    satisfaction_level = st.sidebar.number_input("Nivel de satisfacción",min_value=1, max_value= 100,value=45)
    last_evaluation = st.sidebar.number_input("Puntuación ultima evaluación",min_value=1, max_value= 100,value=60)
    number_project =st.sidebar.number_input('Participación proyectos al año',min_value=0, max_value= 10,value=5)
    average_montly_hours = st.sidebar.number_input('Promedio horas trabajadas al mes',min_value=0, max_value= 350,value=160)
    time_spend_company  = st.sidebar.number_input('Antiguedad en años',min_value=0, max_value= 20,value=5)
    Work_accident =st.sidebar.selectbox('Acidentes laborales',("Si", "No"))
    promotion_last_5years = st.sidebar.selectbox('Promoción los ultimos 5 años?',("Si", "No"))
    Department = st.sidebar.selectbox('Departamento',("Ventas","Técnico","Soporte","IT","RRHH","Contabilidad","Marketing","Producción","randD","Administración"))
    salary =  st.sidebar.selectbox('Nivel salarial', ("low","medium","high"))
    
    # Diccionario de entrada
    input_dict = {
        "satisfaction_level": satisfaction_level,
        "last_evaluation": last_evaluation,
        "number_project": number_project,
        "average_montly_hours": average_montly_hours,
        "time_spend_company": time_spend_company,
        "Work_accident": Work_accident,
        "promotion_last_5years": promotion_last_5years,
        "Department": Department,
        "salary": salary
    }
    # Convertir a un DataFrame
    input_df = pd.DataFrame(input_dict, index=[0])
    input_value = input_df # asignar input_df a input_value
    return input_value
input_value = user_input()
print(input_value.info())


# =============================================================================
 # Label Encoding will be used for columns with 2 or less unique values
 ## Encoding The  Categorical Variables

le1= LabelEncoder()
le1_count = 0
for col in input_value.columns:
     if input_value[col].dtypes == 'object':
         le1.fit(input_value[col])
         input_value[col] = le1.transform(input_value[col])
         le1_count += 1
print('{} columns were label encoded.'.format(le1_count))
   
result2 = None
Prediction2 = reg.predict(input_value)
if st.sidebar.button("Predecir"):
    Prediction = rforest.predict(input_value)
    if Prediction == 0:
        Prediction = "No"
        result = pd.DataFrame({"Abandono?":Prediction,"Info":"El empleado no dejaría la organización"}, index=[0])
        #result2 = pd.DataFrame({"Info": "El empleado dejará la organización en", "Años": Prediction2 })
    else:   
          result = pd.DataFrame({"Abandono?":Prediction,"Info":"El empleado dejará la organización"}, index=[0])                            
          result2 = pd.DataFrame({"Info": "El empleado dejará la organización en", "Años":Prediction2}, index=[0])
        
    st.write("""
             # Resultado de la clasificación:
                 """)
    
    st.write("Desgaste:")  
    if Prediction == 0:
        st.dataframe(result)
    else:  
        Prediction = "Si"
        st.dataframe(result)
        if result2 is not None:
            st.dataframe(result2)
    
    