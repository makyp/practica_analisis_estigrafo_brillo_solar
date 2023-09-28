import pandas as pd
import numpy as np
import openpyxl
import scipy as st
from scipy.stats import skew, kurtosis

#Indica el nombre de las hojas del archivo Excel
wb = openpyxl.load_workbook("DataFrame.xlsx")
print(wb.sheetnames)

#Carga el archivo en su hoja mensual
df = pd.read_excel(io="DataFrame.xlsx",sheet_name="Mensual",header=0,names=None, index_col=None, engine="openpyxl" )    
print(df.head())

#Calculo de la media general de todos los meses
columnas_a_excluir = ["AÑO", "CODIGO"]#Se excluye el año y el codigo
media_total = df.drop(columns=columnas_a_excluir).mean().mean()
print("Media: ",media_total)

#Calculo de la mediana general de todos los meses
mediana_total = df.drop(columns=columnas_a_excluir).median().median()
print("Mediana: ",mediana_total)

#Calculo de la moda de todos los meses segun el mes
moda = df.drop(columns=columnas_a_excluir).mode().iloc[1]
print("Moda:\n",moda)

#Calculo de la moda a nivel general
serie_unidimensional = df.drop(columns=columnas_a_excluir).values.flatten()
moda_total = pd.Series(serie_unidimensional).mode().iloc[0]
print("moda general ", moda_total)

#Calculo de la desviación estandar de todos los meses
desviación_estandar = df.drop(columns=columnas_a_excluir).std().std()
print("Desviación estandar: ",desviación_estandar)

#Calculo de la Varianza de todos los meses
Varianza = df.drop(columns=columnas_a_excluir).var().var()
print("Varianza: ",Varianza)

#Calculo de la Curtosis de todos los meses
Curtosis = kurtosis(df.drop(columns=columnas_a_excluir))
print("Curtosis: ",Curtosis)

#Calculo de la Asimetria de todos los meses
Asimetria = skew(df.drop(columns=columnas_a_excluir))
print("Asimetria: ",Asimetria)

#Calculo del rango
datos =df.drop(columns=columnas_a_excluir)
def calcula_rango(datos):
    return np.max(datos, axis=0)
print("Rangos por columnas:\n",calcula_rango(datos))

#Valor minimo diferente de 0 y valor maximo
datos_sin_cero = datos[datos != 0]
if datos_sin_cero.size > 0:
    valor_minimo = np.min(datos_sin_cero)
    valor_maximo = np.max(datos_sin_cero)
    print("Valor mínimo (sin incluir cero):", valor_minimo)
    print("Valor máximo", valor_maximo)
else:
    print("No hay valores diferentes de cero en el conjunto de datos.")









