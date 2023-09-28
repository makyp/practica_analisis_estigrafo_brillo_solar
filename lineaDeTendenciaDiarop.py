# Imports necesarios
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv("./DataFrameDiarios.csv")
#veamos cuantas dimensiones y registros contiene
print("Dimensiones: ",data.shape)

print(data.head())
# Ahora veamos algunas estadísticas de nuestros datos
print(data.describe())

# Vamos a RECORTAR los datos en la zona donde se concentran más los puntos
# esto es en el eje X: entre 0 y 1000
# y en el eje Y: entre 0 y 1000
filtered_data = data[(data['ENE'] <= 400) & (data['AÑO'] <= 2030)]
 
colores=['orange','blue']
tamanios=[20,20]
 
f1 = filtered_data['ENE'].values
f2 = filtered_data['AÑO'].values
f4 = filtered_data['FEB'].values
f3 = filtered_data['ABR'].values
f5 = filtered_data['MAR'].values
f6 = filtered_data['MAY'].values
f7 = filtered_data['JUN'].values
f8 = filtered_data['JUL'].values
f9 = filtered_data['AGO'].values
f10 = filtered_data['SEP'].values
f11 = filtered_data['OCT'].values
f12 = filtered_data['NOV'].values

# Vamos a pintar en colores los puntos por debajo y por encima de la media que 4.05 respeto enero
plt.xlabel('Eje X: Brillo Solar')
plt.ylabel('Eje Y: AÑO')
plt.title('Gráfico linea de tendencia')
asignar=[]
# Vamos a pintar en colores los puntos por debajo y por encima de la media que 4.05 respeto enero
for index, row in filtered_data.iterrows():
    if(row['ENE']>4.0):
        asignar.append(colores[0])
    else:
        asignar.append(colores[1])
#Personalizar el grafico    
plt.scatter(f1, f2, c=asignar, s=tamanios[0])

# Vamos a pintar en colores los puntos por debajo y por encima de la media que 4.05 respeto Febrero
asignar=[]
for index, row in filtered_data.iterrows():
    if(row['FEB']>4.0):
        asignar.append(colores[0])
    else:
        asignar.append(colores[1])
plt.scatter(f4, f2, c=asignar, s=tamanios[0])

# Vamos a pintar en colores los puntos por debajo y por encima de la media que 4.05 respeto Marzo
asignar=[]
for index, row in filtered_data.iterrows():
    if(row['MAR']>4.0):
        asignar.append(colores[0])
    else:
        asignar.append(colores[1])
   
plt.scatter(f5, f2, c=asignar, s=tamanios[0])

asignar=[]
# Vamos a pintar en colores los puntos por debajo y por encima de la media que 4.05 respeto ABRIL
for index, row in filtered_data.iterrows():
    if(row['ABR']>4.0):
        asignar.append(colores[0])
    else:
        asignar.append(colores[1])
#Personalizar el grafico    
plt.scatter(f3, f2, c=asignar, s=tamanios[0])

# Vamos a pintar en colores los puntos por debajo y por encima de la media que 4.05 respeto MAYO
asignar=[]
for index, row in filtered_data.iterrows():
    if(row['MAY']>4.0):
        asignar.append(colores[0])
    else:
        asignar.append(colores[1])
plt.scatter(f6, f2, c=asignar, s=tamanios[0])

# Vamos a pintar en colores los puntos por debajo y por encima de la media que 4.05 respeto JUNIO
asignar=[]
for index, row in filtered_data.iterrows():
    if(row['JUN']>4.0):
        asignar.append(colores[0])
    else:
        asignar.append(colores[1])
   
plt.scatter(f7, f2, c=asignar, s=tamanios[0])
# Vamos a pintar en colores los puntos por debajo y por encima de la media que 4.05 respeto julio
asignar=[]
for index, row in filtered_data.iterrows():
    if(row['JUL']>4.0):
        asignar.append(colores[0])
    else:
        asignar.append(colores[1])
#Personalizar el grafico    
plt.scatter(f8, f2, c=asignar, s=tamanios[0])

# Vamos a pintar en colores los puntos por debajo y por encima de la media que 4.05 respeto agosto
asignar=[]
for index, row in filtered_data.iterrows():
    if(row['AGO']>4.0):
        asignar.append(colores[0])
    else:
        asignar.append(colores[1])
plt.scatter(f9, f2, c=asignar, s=tamanios[0])

# Vamos a pintar en colores los puntos por debajo y por encima de la media que 4.05 respeto septiembre
asignar=[]
for index, row in filtered_data.iterrows():
    if(row['SEP']>4.0):
        asignar.append(colores[0])
    else:
        asignar.append(colores[1])
   
plt.scatter(f10, f2, c=asignar, s=tamanios[0])
# Vamos a pintar en colores los puntos por debajo y por encima de la media que 4.05 respeto octubre
asignar=[]
for index, row in filtered_data.iterrows():
    if(row['OCT']>4.0):
        asignar.append(colores[0])
    else:
        asignar.append(colores[1])   
plt.scatter(f11, f2, c=asignar, s=tamanios[0])

# Vamos a pintar en colores los puntos por debajo y por encima de la media que 4.05 respeto noviembre
asignar=[]
for index, row in filtered_data.iterrows():
    if(row['NOV']>4.0):
        asignar.append(colores[0])
    else:
        asignar.append(colores[1])
plt.scatter(f12, f2, c=asignar, s=tamanios[0])

# Asignamos nuestra variable de entrada X para entrenamiento y las etiquetas Y.
dataX =filtered_data[["SEP"]]
X_train = np.array(dataX)
y_train = filtered_data['AÑO'].values

   # Asignamos nuestra variable de entrada X para entrenamiento y las etiquetas Y.
dataX =filtered_data[["SEP"]]
X_train = np.array(dataX)
y_train = filtered_data['AÑO'].values
 
# Creamos el objeto de Regresión Linear
regr = linear_model.LinearRegression()
 
# Entrenamos nuestro modelo
regr.fit(X_train, y_train)
 
# Hacemos las predicciones que en definitiva una línea (en este caso, al ser 2D)
y_pred = regr.predict(X_train)

# Trazar la línea de tendencia
plt.plot(X_train, y_pred, color='red', linewidth=2, label='Línea de tendencia')


# Veamos los coeficienetes obtenidos, En nuestro caso, serán la Tangente
print('Coefficients: \n', regr.coef_)
# Este es el valor donde corta el eje Y (en X=0)
print('Independent term: \n', regr.intercept_)
# Error Cuadrado Medio
print("Mean squared error: %.2f" % mean_squared_error(y_train, y_pred))
# Puntaje de Varianza. El mejor puntaje es un 1.0
print('Variance score: %.2f' % r2_score(y_train, y_pred))
plt.show()