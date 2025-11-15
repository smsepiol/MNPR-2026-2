# -*- coding: utf-8 -*-
"""
Created on Fri Nov 14 17:08:56 2025

@author: omarr
"""

import pandas as pd
import numpy as np
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.renders_default = 'browser'

# Paso  0 Realizar la prueba de Lilliefors
# H0: Los datos siguen una distribución Normal
# H0: Los datos NO siguen una distribución Normal

# Paso 1: Cargar los datos
df = pd.read_csv("^MXX.csv")
df = df.dropna() # Eliminar valores núlos

# Paso adicional
fig = px.line(df, x=df.index, y='Close')
fig.show()

fig = px.histogram(df, x='Close')
fig.show()

# Paso 2. Obtner los rendimientos diarios
df["Return"] = df["Close"].pct_change()


# Paso adicional
fig = px.line(df, x=df.index, y='Return')
fig.show()

fig = px.histogram(df, x='Return')
fig.show()

# Paso 3. Extraer rendimientos y ordenarlos
x = df["Return"].dropna()
x = np.sort(x)

# Paso 3. Construir la función empírica

def empirical(data):
    n = len(data)
    muestra = np.sort(data)
    empirical_values = []
    for x in muestra:
        # Contar los valores menores o iguales a x
        count = muestra <= x
        #print(f"Valores menores e iguales a {x}: {count}")
        empirical_values.append( np.sum(count) / n)
    return empirical_values

F_n = empirical(x)

# Paso 4. Calcular los parámetros de la función dada.
n = len(x)
mu = np.mean(x) # np.sum(x) / n
sigma2 = np.var(x, ddof=1) 
sigma = np.sqrt(sigma2)

# Paso 5.Calcular la función de Distribución acumulada bajo la distribución propuesta
# P(X<x) = F(x)
F_x = stats.norm.cdf(x, loc=mu, scale=sigma)


# Paso 4.5 Construir el DataFrame 
lillie = pd.DataFrame({"x": x, "F(X)": F_x, "Fn(i)": F_n})


fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=F_n, line_shape='hv', name="Función empírica"))
fig.add_trace( go.Scatter( x=x, y=F_x, mode='lines', name="Función propuesta")) 
fig.show()

# Paso 5: Hacer la función desfasada
lillie["Fn(i-1)"] = lillie["Fn(i)"] - 1 / n

# Paso 6. Calcular columna D+ y columna D-
lillie["D+"] = abs(lillie["F(X)"] -lillie["Fn(i)"] )
lillie["D-"] = abs(lillie["F(X)"] -lillie["Fn(i-1)"] )


# Paso 7,
D_mas = lillie["D+"].max()
D_menos = lillie["D-"].max()

# Paso 8
Dn = max(D_mas, D_menos)
Dn

# Paso 9 
# Para un nivel de significancia del 5%
alpha = 0.05
W_alpha = 0.875897/np.sqrt(n)

# Regla de decisión
if Dn > W_alpha:
    print("Rechazar H0: Los datos no siguen una distribución normal")
else:
    print("No rechazar H0: Los datos siguen una distribución normal")
