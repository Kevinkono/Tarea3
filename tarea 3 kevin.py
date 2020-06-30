# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 19:33:39 2020

@author: Kevin Picado soto A94779
"""

#paquetes utlizados 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.pyplot import *
from pylab import *
from mpl_toolkits.mplot3d.axes3d import Axes3D

################################################################
################################################################
#   Punto #1
# A partir de los datos del archivo xy.csv se procede a encontrar 
#la mejor curva de ajuste  para las funciones de densidad marginales de X y Y.

################################################################
################################################################


#se llama el archivo tipo csv y se nombran las columnas 
datosxy = pd.read_csv('xy.csv',skiprows=0,names=['5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25'],header=0)

#se establecen la sumatoria de todas las columnas y filas 
#para encontra la PMF en X y Y
Fy=np.sum(datosxy,axis=0)#PMF Y
Fx=np.sum(datosxy,axis=1)#PMF X

#se establecen los vectores de Xs y Ys
Xs=np.linspace(5,15,11)#se hace un vector con el tamano para todas las X 
Ys=np.linspace(5,25,21)#se hace un vector con el tamano para todas las Y
  
print('Los datos dela PMF de X:')
print(Fx)
     
print()   
print('Los datos dela PMF de Y:')
print(Fy)

#Se grafico para ver el comportamiento original de las funciones maginales X y Y
plt.plot(Ys, Fy)
plt.title("Curva con ruido para PMF de Y")
plt.show()


plt.plot(Xs, Fx)
plt.title("Curva con ruido para PMF de X")
plt.show()


#Se deduce que el compartamiebto de los datos es gaussiano cada curva por lo 
#que se procede a definir la funcion gaussiana para determinar sus parametros 
# de ajjuste mu y sigma 
def gaus(x,mu,sigma):
    return 1/(np.sqrt(2*np.pi*sigma**2))*np.exp(-(x-mu)**2/(2*sigma**2))


#se hallan los parametros de las curvas de ajuste para Fx y Fy conel modelo
parax,_=curve_fit(gaus,Xs,Fx)
paray,_=curve_fit(gaus,Ys,Fy)

# se nombran adecuadamente los parametros obtenidos 
mux=parax[0]
sigx=parax[1]

muy=paray[0]
sigy=paray[1]

print('Parametros para Funcion marginal de X ')
print('mu=',mux,'    sigma=',sigx)
print()
print('Parametros para Funcion marginal de Y ')
print('mu=',muy,'    sigma=',sigy)
print()
################################################################
################################################################
#   Punto #3
# Se  calculan los valores de correlación, covarianza y coeficiente de (Pearson) 
#para los datos

################################################################
################################################################


# Con esto se llama los datos del archivo xyp.csv
datosxyp = pd.read_csv('xyp.csv',skiprows=0,names=['x','y','p'],header=0,)

# se establecen los vetores X, Y , P para los datos de las columnas de xyp.csv
X=datosxyp['x']#vetor columna X
Y=datosxyp['y']#vetor columna Y
P=datosxyp['p']#vetor columna P



cor =sum((X*Y)*P)#se calcula la correlacion 
cov =sum(((X-mux)*(Y-muy))*P) # se cacula la covarianza 
copear=(cov)/(sigx*sigy)#se calcula el coef de pearson

print('Elvalos de la correlacion:',cor)
print('El valor de la covarianza:',cov)
print('EL valor del coef de correlacion (Pearson):',copear)


################################################################
################################################################
#   Punto #4 
#Se procede a graficar  las funciones de densidad marginales
# en 2D para X y para Y, Y la función de densidad conjunta XY (3D)

################################################################
################################################################

#con los parametros se procede sacar las curvas de ajuste 
plt.plot(Xs,gaus(Xs,mux,sigx))#curva ajustada sin ruido  Fx
plt.title("Curva para PMF de X")
plt.show()
plt.plot(Ys,gaus(Ys,muy,sigy))#curva ajustada sin ruido  Fy
plt.title("Curva para PMF de Y")
plt.show()


#Se saca garfica la funcion conjunta 3D
Xs,Ys =np.meshgrid(Xs, Ys)
# se obtiene la funcion conjunta de x y 
XY=(1/((2*np.pi)*(sigx)*(sigy)))*np.exp((-1/2)*((((Xs-mux)**2)/( sigx**2)) + (((Ys-muy)**2)/( sigy**2))))
fig=plt.figure(figsize=(7,7))
FXY=plt.axes(projection='3d')
FXY.plot_surface(Xs, Ys, XY,cmap='rainbow' )
FXY.set_title('Funcion de densidad conjunta')
FXY.set_xlabel('eje X')
FXY.set_ylabel('eje Y ')
FXY.set_zlabel('eje P() conjunta ')

