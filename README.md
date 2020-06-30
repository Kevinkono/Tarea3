
# ** Variables Aleatorias Multiples **
Universidad de Costa Rica

Modelos Probabilisticos de señales y sistemas

Kevin Picado Soto A94779

####  TAREA 3
1. (25 %) A partir de los datos, encontrar la mejor curva de ajuste (modelo probabilístico) para las funciones de densidad marginales de X y Y.
2. (25 %) Asumir independencia de X y Y. Analíticamente, ¿cuál es entonces la expresión de la función de densidad conjunta que modela los datos?
3. (25 %) Hallar los valores de correlación, covarianza y coeficiente de correlación (Pearson) para los datos y explicar su significado.
4. (25 %) Graficar las funciones de densidad marginales (2D), la función de densidad conjunta (3D).

#### Solución

1. 
Para el primer enunciado  se procede a buscar las funciones marginales de X y Y , para esto se inicia importando las librerias necesarias como pandas para leer los archivos .csv , numpy para las opreciones matematicas y matplotlib para graficar.
Este primer inciso al realizar la sumatoria de los datos  por filas y columnas se obtienen los vectores necesarios para graficar las funciones marginales de X y Y por medio del siguiente codigo.```python
###### #paquetes utlizados 
import pandas as pd \
import numpy as np \
import matplotlib.pyplot as plt \
from scipy.optimize import curve_fit \
from matplotlib.pyplot import * \
from pylab import * \
from mpl_toolkits.mplot3d.axes3d import Axes3D \
###### #se llama el archivo tipo csv y se nombran las columnas \
datosxy = pd.read_csv('xy.csv',skiprows=0,names=['5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25'],header=0) \
###### #se establecen la sumatoria de todas las columnas y filas \
###### #para encontra la PMF en X y Y \
Fy=np.sum(datosxy,axis=0)#PMF Y \
Fx=np.sum(datosxy,axis=1)#PMF X \
###### #se establecen los vectores de Xs y Ys\
Xs=np.linspace(5,15,11)#se hace un vector con el tamano para todas las X \
Ys=np.linspace(5,25,21)#se hace un vector con el tamano para todas las Y \
print('Los datos dela PMF de X:') \
print(Fx) \   
print()  \ 
print('Los datos dela PMF de Y:') \
print(Fy) \
###### #Se grafico para ver el comportamiento original de las funciones maginales X y Y \
plt.plot(Ys, Fy) \
plt.title("Curva con ruido para PMF de Y") \
plt.show() \
plt.plot(Xs, Fx) \
plt.title("Curva con ruido para PMF de X") \
plt.show()
```

