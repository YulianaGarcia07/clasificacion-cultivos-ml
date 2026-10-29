"""
🌱 Recomendador de cultivos · App de Streamlit

Qué hace:   muestra el análisis del dataset de cultivos y recomienda un cultivo
            a partir de las condiciones de una parcela.
Necesita:   el archivo Crop_recommendation.csv en la misma carpeta que este script
            y las librerías de requirements.txt.
Cómo se ejecuta (en la terminal):   streamlit run app.py
"""

# ------------------------------------------------------------------
# 1. LIBRERÍAS
# ------------------------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st                      # convierte el script en una página web
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Configuración de la página: título de la pestaña del navegador y ancho completo
st.set_page_config(page_title="Recomendador de cultivos", page_icon="🌱", layout="wide")
sns.set_theme(style="whitegrid")


# ------------------------------------------------------------------
# 2. CARGA Y PREPARACIÓN DE LOS DATOS
# ------------------------------------------------------------------
# @st.cache_data guarda el resultado en memoria: el CSV se lee una sola vez,
# no cada vez que el usuario mueve un control (Streamlit vuelve a ejecutar
# todo el script con cada interacción)
@st.cache_data
def cargar_datos():
    df = pd.read_csv("Crop_recommendation.csv")

    # Diccionario para traducir columnas (nombre original → nombre nuevo)
    df = df.rename(columns={
        "N": "nitrogeno", "P": "fosforo", "K": "potasio",
        "temperature": "temperatura", "humidity": "humedad",
        "ph": "ph", "rainfall": "lluvia", "label": "cultivo",
    })

    # Diccionario para traducir los cultivos
    traduccion = {
        "apple": "Manzana", "banana": "Plátano", "blackgram": "Frijol negro",
        "chickpea": "Garbanzo", "coconut": "Coco", "coffee": "Café",
        "cotton": "Algodón", "grapes": "Uva", "jute": "Yute",
        "kidneybeans": "Judía roja", "lentil": "Lenteja", "maize": "Maíz",
        "mango": "Mango", "mothbeans": "Judía polilla", "mungbean": "Judía mungo",
        "muskmelon": "Melón", "orange": "Naranja", "papaya": "Papaya",
        "pigeonpeas": "Guandú", "pomegranate": "Granada", "rice": "Arroz",
        "watermelon": "Sandía",
    }
    df["cultivo"] = df["cultivo"].map(traduccion)
    return df


# @st.cache_resource hace lo mismo para objetos "pesados" como un modelo:
# se entrena una vez y se reutiliza
@st.cache_resource
def entrenar_modelo(df):
    X = df.drop(columns="cultivo")
    y = df["cultivo"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(X_train, y_train)

    # Guardamos las métricas en un diccionario para mostrarlas luego
    metricas = {
        "acierto_test": accuracy_score(y_test, modelo.predict(X_test)),
        "acierto_cv": cross_val_score(modelo, X, y, cv=5).mean(),
        "matriz": confusion_matrix(y_test, modelo.predict(X_test), labels=modelo.classes_),
    }
    return modelo, metricas


# Si el CSV no está, mostramos un aviso claro en lugar de un error técnico
try:
    df = cargar_datos()
except FileNotFoundError:
    st.error("No encuentro 'Crop_recommendation.csv'. Colócalo en la misma carpeta que app.py.")
    st.stop()   # detiene la app aquí

modelo, metricas = entrenar_modelo(df)
variables = ["nitrogeno", "fosforo", "potasio", "temperatura", "humedad", "ph", "lluvia"]
etiquetas = {
    "nitrogeno": "Nitrógeno", "fosforo": "Fósforo", "potasio": "Potasio",
    "temperatura": "Temperatura (°C)", "humedad": "Humedad (%)",
    "ph": "pH", "lluvia": "Lluvia (mm)",
}


# ------------------------------------------------------------------
# 3. CABECERA Y KPIs
# ------------------------------------------------------------------
st.title("🌱 Recomendador de cultivos")
st.caption("Clasificación multiclase con Random Forest · Dataset: Crop Recommendation (Kaggle)")

# st.columns divide la fila en bloques, como las tarjetas de KPI de Power BI
k1, k2, k3, k4 = st.columns(4)
k1.metric("Parcelas", f"{len(df):,}".replace(",", "."))
k2.metric("Cultivos", df["cultivo"].nunique())
k3.metric("Acierto (validación cruzada)", f"{metricas['acierto_cv']:.1%}")
k4.metric("Acierto por azar", f"{1 / df['cultivo'].nunique():.1%}")

# Pestañas: cada una es una "página" de la app
tab_recom, tab_perfil, tab_explorar, tab_modelo, tab_datos = st.tabs(
    ["🔮 Recomendador", "🗺️ Perfil de cultivos", "📊 Explorar variables", "🤖 Modelo", "📄 Datos"]
)


# ------------------------------------------------------------------
# 4. PESTAÑA: RECOMENDADOR
# ------------------------------------------------------------------
with tab_recom:
    st.subheader("¿Qué cultivo conviene en mi parcela?")
    st.write("Mueve los controles con las condiciones de la parcela.")

    col_izq, col_der = st.columns([1, 1])

    # Valores iniciales de ejemplo (una parcela húmeda y lluviosa)
    valores_iniciales = {"nitrogeno": 85.0, "fosforo": 45.0, "potasio": 40.0,
                         "temperatura": 23.0, "humedad": 82.0, "ph": 6.4, "lluvia": 210.0}

    # Un slider por variable. El rango sale de los datos reales
    valores = {}
    for i, var in enumerate(variables):
        columna = col_izq if i < 4 else col_der
        minimo, maximo = float(df[var].min()), float(df[var].max())
        valores[var] = columna.slider(
            etiquetas[var], min_value=minimo, max_value=maximo,
            value=valores_iniciales[var], step=0.1 if var == "ph" else 1.0,
        )

    # Convertimos los valores elegidos en una tabla de 1 fila para el modelo
    parcela = pd.DataFrame([valores])
    probabilidades = pd.Series(modelo.predict_proba(parcela)[0], index=modelo.classes_)
    top3 = probabilidades.sort_values(ascending=False).head(3)

    st.divider()
    st.success(f"### Cultivo recomendado: **{top3.index[0]}** ({top3.iloc[0]:.0%} de confianza)")

    # Gráfico con las 3 opciones más probables
    fig, ax = plt.subplots(figsize=(6, 2))
    top3.sort_values().plot(kind="barh", color="#4c956c", ax=ax)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probabilidad")
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    st.pyplot(fig)

    if top3.iloc[0] < 0.5:
        st.warning("Confianza baja: estas condiciones no se parecen mucho a ninguna parcela del dataset.")


# ------------------------------------------------------------------
# 5. PESTAÑA: PERFIL DE CULTIVOS (el heatmap del notebook)
# ------------------------------------------------------------------
with tab_perfil:
    st.subheader("Condiciones medias de cada cultivo")
    st.write(
        "Cada celda es la **media real** de esa variable en las parcelas de ese cultivo. "
        "El color se calcula por columna: **verde oscuro** = el valor más alto de esa variable, "
        "**amarillo** = el más bajo."
    )

    # Filtro: el usuario elige qué cultivos ver (por defecto, todos)
    seleccion = st.multiselect(
        "Cultivos a mostrar", sorted(df["cultivo"].unique()),
        default=sorted(df["cultivo"].unique()),
    )

    if len(seleccion) < 2:
        st.info("Elige al menos 2 cultivos para comparar.")
    else:
        medias = df[df["cultivo"].isin(seleccion)].groupby("cultivo").mean().round(1)
        medias = medias.rename(columns=etiquetas)
        # Normalización 0-1 por columna, solo para colorear
        rango = (medias.max() - medias.min()).replace(0, 1)   # evita dividir entre 0
        normalizado = (medias - medias.min()) / rango

        fig, ax = plt.subplots(figsize=(10, 0.4 * len(seleccion) + 1.5))
        sns.heatmap(normalizado, annot=medias, fmt="g", cmap="YlGn", cbar=False, ax=ax)
        ax.set_ylabel("")
        st.pyplot(fig)


# ------------------------------------------------------------------
# 6. PESTAÑA: EXPLORAR VARIABLES
# ------------------------------------------------------------------
with tab_explorar:
    st.subheader("Cómo se reparte cada variable entre cultivos")
    var_elegida = st.selectbox("Variable", variables, format_func=lambda v: etiquetas[v])

    orden = df.groupby("cultivo")[var_elegida].median().sort_values().index
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.boxplot(data=df, y="cultivo", x=var_elegida, order=orden, color="#8cc084", ax=ax)
    ax.set_xlabel(etiquetas[var_elegida])
    ax.set_ylabel("")
    st.pyplot(fig)
    st.caption("La caja contiene el 50 % central de las parcelas; la línea interior es la mediana.")


# ------------------------------------------------------------------
# 7. PESTAÑA: MODELO
# ------------------------------------------------------------------
with tab_modelo:
    col_a, col_b = st.columns([1, 1.4])

    with col_a:
        st.subheader("Importancia de las variables")
        importancias = pd.Series(modelo.feature_importances_, index=variables)
        importancias = importancias.rename(etiquetas).sort_values()
        fig, ax = plt.subplots(figsize=(5, 4))
        importancias.plot(kind="barh", color="#4c956c", ax=ax)
        ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
        st.pyplot(fig)
        st.write("La **lluvia** y la **humedad** son las variables que más deciden.")

    with col_b:
        st.subheader("Matriz de confusión (test)")
        fig, ax = plt.subplots(figsize=(8, 7))
        sns.heatmap(metricas["matriz"], annot=True, fmt="d", cmap="Greens", cbar=False,
                    xticklabels=modelo.classes_, yticklabels=modelo.classes_, ax=ax,
                    annot_kws={"size": 7})
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Real")
        st.pyplot(fig)
        st.write(f"Acierto en el conjunto de prueba: **{metricas['acierto_test']:.1%}**")

    st.info(
        "Limitación: un acierto tan alto indica que el dataset es muy limpio. "
        "Antes de usarlo en el campo habría que validarlo con parcelas reales."
    )


# ------------------------------------------------------------------
# 8. PESTAÑA: DATOS
# ------------------------------------------------------------------
with tab_datos:
    st.subheader("Datos originales")
    cultivo_filtro = st.selectbox("Filtrar por cultivo", ["Todos"] + sorted(df["cultivo"].unique()))
    tabla = df if cultivo_filtro == "Todos" else df[df["cultivo"] == cultivo_filtro]
    st.write(f"{len(tabla)} filas")
    st.dataframe(tabla, width="stretch")
