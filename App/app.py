import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from pathlib import Path
from dateutil.parser import parse as dtparse

st.title ("Preddicion Mercado del trabajo en Colombia")

st.write("La Gran Encuesta Integrada de Hogares (GEIH) publicada por el DANE puede ser un gran insumo si se quiere un acercamiento a la realidad socioeconómica de las familias en Colombia. Este trabajo hace uso de este recurso con el fin de encontrar los determinantes con mayor incidencia a la hora de obtener un empleo formal en Colombia. En este apartado se muestran las principales características de los datos encontrados en la GEIH.")
st.markdown("""

## 📋 Objetivos
- Realizar un análisis exploratorio de los datos de la GEIH 2024, incluyendo estadísticas descriptivas y visualización de variables relevantes para el estudio del empleo.
- Aplicar y Evaluar el desempeño del modelo de regresión logística binaria para estimar la probabilidad de ocupación laboral en la formalidad.
- Identificar e interpretar los factores sociodemográficos y económicos más influyentes en la probabilidad de estar ocupado formalmente en Colombia.

> _"La excelencia es el resultado de la práctica constante."_  
""")

DATA_PATH = "C:/Users/ivana/OneDrive/Documentos/App/data_filtrada.csv"  
DATE_COLS_HINT = []                
PRIMARY_TABLE = "data"
APP_TITLE = "Analisis exploratorio de los datos"
st.set_page_config(page_title="Analisis exploratorio de datos", layout="wide")
st.title("🔍 Analisis exploratorio de los datos")

# -------- CARGA DEL DATASET FIJO --------
@st.cache_data
def cargar_datos():
    df = pd.read_csv(DATA_PATH)
    return df

df = cargar_datos()

##st.success(f"Dataset cargado: {DATA_PATH} - {df.shape[0]:,} filas y {df.shape[1]:,} columnas")

st.subheader("🧼 Valores faltantes en el dataset")

# Calcular faltantes
faltantes = df.isnull().sum()
faltantes = faltantes[faltantes > 0]
total_faltantes = faltantes.sum()

if total_faltantes == 0:
    st.success("✅ No hay valores faltantes en este dataset.")
else:
    # Mostrar resumen
    df_faltantes = pd.DataFrame({
        "Columna": faltantes.index,
        "Faltantes": faltantes.values,
        "% del total": (faltantes.values / len(df)) * 100
    }).sort_values(by="Faltantes", ascending=False)

    st.warning(f"⚠️ Se encontraron {total_faltantes:,} valores faltantes en {len(df_faltantes)} columnas.")
    st.dataframe(df_faltantes, use_container_width=True)

    chart = alt.Chart(df_faltantes).mark_bar().encode(
        y=alt.Y("Columna:N", sort="-x"),
        x=alt.X("Faltantes:Q"),
        tooltip=["Columna", "Faltantes", "% del total"]
    ).properties(height=400)
    st.altair_chart(chart, use_container_width=True)



DATA_PATH = "C:/Users/ivana/OneDrive/Documentos/App/data_filtrada_imputada.csv"  
DATE_COLS_HINT = []                
PRIMARY_TABLE = "data"
APP_TITLE = "Analisis exploratorio de los datos"
st.set_page_config(page_title="Analisis exploratorio de datos", layout="wide")

# -------- CARGA DEL DATASET FIJO --------
@st.cache_data
def cargar_datos():
    df2 = pd.read_csv(DATA_PATH)
    return df2

df2 = cargar_datos()

mapeo_formalidad = {0: 'Informal', 1: 'Formal'}
datos = df2
datos['formalidad'] = datos['formalidad'].map(mapeo_formalidad)

# -------- Agrupar valores --------
df_formalidad = datos['formalidad'].value_counts().reset_index()
df_formalidad.columns = ['formalidad', 'n']
colores = {'Informal': '#F8766D', 'Formal': '#00BFC4'}

# -------- Gráfico Altair --------
st.subheader("📊 Distribución de la Formalidad Laboral")

bar_chart = alt.Chart(df_formalidad).mark_bar(size=60).encode(
    x=alt.X('formalidad:N', title="Tipo de empleo"),
    y=alt.Y('n:Q', title="Número de personas"),
    color=alt.Color('formalidad:N', scale=alt.Scale(domain=list(colores.keys()),
                                                    range=list(colores.values())),
                    legend=None),
    tooltip=['formalidad', 'n']
).properties(
    width=500,
    height=400,
    title="Distribución de la Formalidad Laboral"
)

# Etiquetas sobre las barras
text = bar_chart.mark_text(
    align='center',
    baseline='bottom',
    dy=-5,
    fontSize=14
).encode(
    text='n:Q'
)

st.altair_chart(bar_chart + text, use_container_width=True)

st.sidebar.header("🎛️ Filtros de visualización")

# Filtro por región
regiones = sorted(df2['AREA'].dropna().unique())
region_seleccionada = st.sidebar.multiselect("AREA", regiones, default=regiones)

# Filtro por sexo
sexos = sorted(df2['Sexo'].dropna().unique())
sexo_seleccionado = st.sidebar.multiselect("Sexo", sexos, default=sexos)

# Filtro por edad
edad_min = int(df2['Edad'].min())
edad_max = int(df2['Edad'].max())
rango_edad = st.sidebar.slider("Edad", edad_min, edad_max, (edad_min, edad_max))

df_filtrado = df2[
    (df['AREA'].isin(region_seleccionada)) &
    (df['Sexo'].isin(sexo_seleccionado)) &
    (df['Edad'] >= rango_edad[0]) &
    (df['Edad'] <= rango_edad[1])
].copy()

if df_filtrado['formalidad'].dropna().dtype in ['int64', 'float64']:
    mapeo_formalidad = {0: 'Informal', 1: 'Formal'}
    df_filtrado['formalidad'] = df_filtrado['formalidad'].map(mapeo_formalidad)

# ✅ Conteo y gráfica
if 'formalidad' not in df_filtrado.columns or df_filtrado['formalidad'].dropna().empty:
    st.warning("⚠️ No hay datos válidos para la columna 'formalidad' después de aplicar los filtros.")
else:
    df_formalidad = df_filtrado['formalidad'].value_counts().reset_index()
    df_formalidad.columns = ['formalidad', 'n']
    colores = {'Informal': '#F8766D', 'Formal': '#00BFC4'}

    st.subheader("📊 Distribución de la Formalidad Laboral (filtrada)")

    bar_chart = alt.Chart(df_formalidad).mark_bar(size=60).encode(
        x=alt.X('formalidad:N', title="Tipo de empleo"),
        y=alt.Y('n:Q', title="Número de personas"),
        color=alt.Color('formalidad:N',
                        scale=alt.Scale(domain=list(colores.keys()), range=list(colores.values())),
                        legend=None),
        tooltip=['formalidad', 'n']
    ).properties(width=500, height=400)

    text = bar_chart.mark_text(
        align='center',
        baseline='bottom',
        dy=-5,
        fontSize=14
    ).encode(text='n:Q')

    st.altair_chart(bar_chart + text, use_container_width=True)

df_filtrado['grupo_edad'] = pd.cut(df_filtrado['Edad'], bins=[0, 17, 25, 35, 50, 65, 100],
                                    labels=['<18', '18-25', '26-35', '36-50', '51-65', '65+'],
                                    right=False)
st.subheader("📌 Indicadores Generales (KPIs)")

col1, col2 = st.columns(2)

with col1:
    if 'Ingresos laborales' in df_filtrado.columns:
        ingreso_prom = df_filtrado['Ingresos laborales'].mean()
        st.metric("💰 Ingreso promedio", f"${ingreso_prom:,.0f}")
    else:
        st.info("La columna 'Ingreso' no está disponible.")

with col2:
    if 'Mayor nivel educativo alcanzado' in df_filtrado.columns:
        edu_modal = df_filtrado['Mayor nivel educativo alcanzado'].mode().iloc[0]
        st.metric("🎓 Nivel educativo más frecuente", edu_modal)
    else:
        st.info("La columna 'Nivel_educativo' no está disponible.")
        
        # Agrupar por Sexo y Área
tabla_heatmap = df_filtrado.groupby(['Sexo', 'AREA']).agg(
    porcentaje_formalidad=('formalidad', lambda x: (x == 'Formal').mean() * 100)
).reset_index()

import altair as alt

st.subheader("🧭 Mapa de calor: Formalidad por Sexo y Región")

if tabla_heatmap.empty:
    st.warning("⚠️ No hay datos suficientes para generar el mapa de calor.")
else:
    heatmap = alt.Chart(tabla_heatmap).mark_rect().encode(
        x=alt.X('AREA:N', title="Región"),
        y=alt.Y('Sexo:N', title="Sexo"),
        color=alt.Color('porcentaje_formalidad:Q',
                        scale=alt.Scale(scheme='blues', domain=[0, 100]),
                        title="% Formalidad"),
        tooltip=['Sexo', 'AREA', alt.Tooltip('porcentaje_formalidad:Q', format='.1f')]
    ).properties(
        width=600,
        height=300
    )

    st.altair_chart(heatmap, use_container_width=True)

import altair as alt

st.subheader("🎂 Distribución de la Edad por Tipo de Empleo")

if 'Edad' in df_filtrado.columns and 'formalidad' in df_filtrado.columns:
    histograma_edad = alt.Chart(df_filtrado).mark_bar(opacity=0.6).encode(
        alt.X("Edad:Q", bin=alt.Bin(maxbins=30), title="Edad"),
        alt.Y("count():Q", stack=None, title="Número de personas"),
        alt.Color("formalidad:N",
                  scale=alt.Scale(domain=["Informal", "Formal"],
                                  range=["#F8766D", "#00BFC4"]),
                  title="Tipo de Empleo"),
        tooltip=["formalidad", "Edad"]
    ).properties(
        width=700,
        height=400
    )

    st.altair_chart(histograma_edad, use_container_width=True)
else:
    st.warning("❗ La columna 'Edad' o 'formalidad' no está disponible en el dataset.")

st.markdown("### 📦 Comparación de Edad: Formal vs. Informal")

boxplot_edad = alt.Chart(df_filtrado).mark_boxplot(extent='min-max').encode(
    x=alt.X('formalidad:N', title="Tipo de Empleo"),
    y=alt.Y('Edad:Q', title="Edad"),
    color=alt.Color('formalidad:N',
                    scale=alt.Scale(domain=["Informal", "Formal"],
                                    range=["#F8766D", "#00BFC4"]),
                    legend=None)
).properties(
    width=400,
    height=400
)

st.altair_chart(boxplot_edad, use_container_width=True)

st.subheader("🕓 Distribución del Tiempo de Desplazamiento por Tipo de Empleo")

if 'Tiempo de desplazamiento hasta su sitio de trabajo' in df_filtrado.columns and 'formalidad' in df_filtrado.columns:
    histo_desplazamiento = alt.Chart(df_filtrado).mark_bar(opacity=0.6).encode(
        alt.X("Tiempo de desplazamiento hasta su sitio de trabajo:Q", bin=alt.Bin(maxbins=40), title="Tiempo de desplazamiento (minutos)"),
        alt.Y("count():Q", stack=None, title="Número de personas"),
        alt.Color("formalidad:N",
                  scale=alt.Scale(domain=["Informal", "Formal"],
                                  range=["#F8766D", "#00BFC4"]),
                  title="Tipo de Empleo"),
        tooltip=["formalidad", "Tiempo de desplazamiento hasta su sitio de trabajo"]
    ).properties(
        width=700,
        height=400
    )

    st.altair_chart(histo_desplazamiento, use_container_width=True)
else:
    st.warning("⚠️ La columna 'tiempo_desplazamiento' o 'formalidad' no está disponible.")

st.markdown("### 📈 Tiempo promedio de desplazamiento por tipo de empleo")

tiempo_promedio = df_filtrado.groupby('formalidad')['Tiempo de desplazamiento hasta su sitio de trabajo'].mean().reset_index()
tiempo_promedio.columns = ['Tipo de Empleo', 'Tiempo Promedio (min)']
st.dataframe(tiempo_promedio.style.format({"Tiempo Promedio (min)": "{:.1f}"}))

import joblib
import pandas as pd
import streamlit as st

# 🧠 Cargar el modelo y las columnas usadas en el entrenamiento
modelo = joblib.load("modelo_formalidad.pkl")
columnas_modelo = joblib.load("columnas_modelo.pkl")

st.subheader("🔎 Predicción de formalidad laboral")

with st.form("formulario_prediccion"):
    col1, col2 = st.columns(2)

    with col1:
        edad = st.slider("Edad", 18, 65, 30)
        sexo = st.selectbox("Sexo", ["Hombre", "Mujer"])
        nivel_educativo = st.selectbox("Nivel educativo", df2["Mayor nivel educativo alcanzado"].unique())
        region = st.selectbox("Región", df2["AREA"].unique())

    with col2:
        ingreso = st.number_input("Ingreso mensual", min_value=0, step=100)
        meses_para_nuevo_trabajo = st.number_input("Meses desempleado", min_value=0, step=1)

    submit = st.form_submit_button("Predecir")

if submit:
    # 1️⃣ Armar el dataframe con los datos del usuario
    entrada = pd.DataFrame([{
        'Edad': edad,
        'Sexo': sexo,
        'Mayor nivel educativo alcanzado': nivel_educativo,
        'Ingreso': ingreso,
        'AREA': region,
        'Meses que estuvo sin empleo entre su trabajo actual y el anterior': meses_para_nuevo_trabajo
    }])

    # 2️⃣ Convertir variables categóricas en dummies
    entrada_dummies = pd.get_dummies(entrada)

    # 3️⃣ Reindexar para que coincida con columnas del modelo
    entrada_dummies = entrada_dummies.reindex(columns=columnas_modelo, fill_value=0)

    # 4️⃣ Predecir probabilidad de formalidad
    prob = modelo.predict_proba(entrada_dummies)[0][1]  # Clase 1 = formal

    st.success(f"🔮 Probabilidad estimada de tener empleo formal: **{prob*100:.2f}%**")
