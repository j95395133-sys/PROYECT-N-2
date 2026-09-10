import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Teen Mental Health - EDA", layout="wide")
sns.set_style("whitegrid")


# Funcion personalizada para clasificar variables numericas o categoricas
def clasificar_variable(columna):
    if pd.api.types.is_numeric_dtype(columna):
        return "Numerica"
    else:
        return "Categorica"


# Clase que encapsula la logica del analisis (POO)
class DataAnalyzer:
    def __init__(self, df):
        self.df = df

    def info_general(self):
        filas, columnas = self.df.shape
        nulos = self.df.isnull().sum()
        duplicados = self.df.duplicated().sum()
        return filas, columnas, nulos, duplicados

    def clasificar_variables(self):
        tipos = {col: clasificar_variable(self.df[col]) for col in self.df.columns}
        tabla = pd.DataFrame({"Variable": tipos.keys(), "Tipo": tipos.values()})
        numericas = tabla[tabla["Tipo"] == "Numerica"]["Variable"].tolist()
        categoricas = tabla[tabla["Tipo"] == "Categorica"]["Variable"].tolist()
        return tabla, numericas, categoricas

    def estadisticas(self, columnas):
        return self.df[columnas].describe()

    def valores_faltantes(self):
        total = self.df.isnull().sum()
        porcentaje = (total / len(self.df) * 100).round(2)
        return pd.DataFrame({"Nulos": total, "Porcentaje": porcentaje})

    def filtrar(self, edad, generos, plataformas, interaccion):
        df2 = self.df.copy()
        df2 = df2[(df2["age"] >= edad[0]) & (df2["age"] <= edad[1])]
        if generos:
            df2 = df2[df2["gender"].isin(generos)]
        if plataformas:
            df2 = df2[df2["platform_usage"].isin(plataformas)]
        if interaccion:
            df2 = df2[df2["social_interaction_level"].isin(interaccion)]
        return df2

    def comparar_grupos(self, var_num, var_cat):
        return self.df.groupby(var_cat)[var_num].agg(["mean", "median", "count"]).round(2)


DICCIONARIO = {
    "age": "Edad del adolescente (13 a 19 anios)",
    "gender": "Genero registrado",
    "daily_social_media_hours": "Horas diarias de uso de redes sociales",
    "platform_usage": "Plataforma usada: Instagram, TikTok o ambas",
    "sleep_hours": "Horas de sueño por dia",
    "screen_time_before_sleep": "Tiempo de pantalla antes de dormir",
    "academic_performance": "Indicador de rendimiento academico",
    "physical_activity": "Horas de actividad fisica",
    "social_interaction_level": "Nivel de interaccion social",
    "stress_level": "Nivel de estres (escala 1 a 10)",
    "anxiety_level": "Nivel de ansiedad (escala 1 a 10)",
    "addiction_level": "Nivel de dependencia (escala 1 a 10)",
    "depression_label": "Etiqueta binaria: 0 = ausencia, 1 = presencia",
}
ESCALAS = ["stress_level", "anxiety_level", "addiction_level"]
HABITOS = ["daily_social_media_hours", "sleep_hours", "screen_time_before_sleep", "physical_activity"]

if "df" not in st.session_state:
    st.session_state.df = None

# Sidebar con el menu de navegacion
st.sidebar.title("Menu")
modulo = st.sidebar.radio(
    "Selecciona un modulo:",
    ["Home", "Carga del Dataset", "Analisis Exploratorio (EDA)", "Conclusiones"],
)
st.sidebar.markdown("---")
st.sidebar.info("Proyecto educativo y exploratorio. No es un diagnostico clinico.")


# ---------------- MODULO 1: HOME ----------------
if modulo == "Home":
    st.title("Teen Mental Health Dataset - Analisis Exploratorio de Datos")

    st.markdown("**Nombre completo:** Jesus Enrique Navio Ramirez")
    st.markdown("**Modulo:** Caso de Estudio N4 - Analisis Exploratorio de Datos (EDA)")
    st.markdown("**Curso:** Especializacion en Python for Analytics")
    st.markdown("**Institucion:** DMC Institute")
    st.markdown("**Año:** 2026")

    st.markdown("---")
    st.subheader("Descripcion del proyecto")
    st.write(
        "Esta aplicacion aplica lo aprendido en la especializacion: variables, "
        "funciones, f-strings, POO, NumPy, Pandas, visualizacion con Matplotlib y "
        "Seaborn, y estadistica descriptiva. El objetivo es analizar y visualizar "
        "el dataset de salud mental en adolescentes para encontrar patrones entre "
        "el uso de redes sociales, el descanso, la actividad fisica y el bienestar. "
        "No se construyen modelos predictivos."
    )

    st.subheader("Tecnologias utilizadas")
    st.write("Python, Streamlit, Pandas, NumPy, Matplotlib, Seaborn")

    with st.expander("Ver diccionario de variables"):
        st.dataframe(pd.DataFrame({"Variable": DICCIONARIO.keys(), "Descripcion": DICCIONARIO.values()}))


# ---------------- MODULO 2: CARGA DEL DATASET ----------------
elif modulo == "Carga del Dataset":
    st.title("Carga del Dataset")
    st.write("Sube el archivo Teen_Mental_Health_Dataset.csv para comenzar.")

    archivo = st.file_uploader("Selecciona el archivo CSV", type=["csv"])

    if archivo is not None:
        try:
            df = pd.read_csv(archivo)
            st.session_state.df = df
            st.success("Archivo cargado correctamente.")

            col1, col2 = st.columns(2)
            col1.metric("Filas", df.shape[0])
            col2.metric("Columnas", df.shape[1])

            st.subheader("Vista previa del dataset")
            st.dataframe(df.head(1201))

            if st.checkbox("Mostrar tipos de columnas"):
                st.write(df.dtypes)
        except Exception as e:
            st.error(f"Error al leer el archivo: {e}")
    else:
        st.warning("Aun no se ha cargado ningun archivo.")


# ---------------- MODULO 3: EDA ----------------
elif modulo == "Analisis Exploratorio (EDA)":
    st.title("Analisis Exploratorio de Datos (EDA)")

    if st.session_state.df is None:
        st.error("Primero carga el dataset en el modulo 'Carga del Dataset'.")
        st.stop()

    analyzer = DataAnalyzer(st.session_state.df)
    df = analyzer.df

    tabs = st.tabs([
        "1. Info general", "2. Clasif. variables", "3. Estadisticas", "4. Faltantes",
        "5. Distribuciones", "6. Categoricas", "7. Bivariado num-cat",
        "8. Bivariado cat-cat", "9. Filtros", "10. Hallazgos",
    ])

    # Item 1
    with tabs[0]:
        st.header("Informacion general del dataset")
        filas, columnas, nulos, duplicados = analyzer.info_general()
        col1, col2, col3 = st.columns(3)
        col1.metric("Filas", filas)
        col2.metric("Columnas", columnas)
        col3.metric("Duplicados", int(duplicados))
        st.write("Valores nulos por columna:")
        st.dataframe(nulos)

    # Item 2
    with tabs[1]:
        st.header("Clasificacion de variables")
        tabla, numericas, categoricas = analyzer.clasificar_variables()
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(tabla)
        with col2:
            st.bar_chart(tabla["Tipo"].value_counts())
            st.write(f"Numericas: {len(numericas)} | Categoricas: {len(categoricas)}")

    # Item 3
    with tabs[2]:
        st.header("Estadisticas descriptivas")
        _, numericas, _ = analyzer.clasificar_variables()
        st.dataframe(analyzer.estadisticas(numericas).round(2))

        var = st.selectbox("Variable para ver outliers:", numericas)
        fig, ax = plt.subplots(figsize=(6, 3))
        sns.boxplot(x=df[var], ax=ax)
        st.pyplot(fig)

    # Item 4
    with tabs[3]:
        st.header("Analisis de valores faltantes")
        faltantes = analyzer.valores_faltantes()
        st.dataframe(faltantes)
        if faltantes["Nulos"].sum() == 0:
            st.write("El dataset no tiene valores faltantes.")

    # Item 5
    with tabs[4]:
        st.header("Distribucion de variables numericas")
        _, numericas, _ = analyzer.clasificar_variables()
        var = st.selectbox("Variable:", numericas, key="var5")
        fig, ax = plt.subplots(figsize=(6, 3))
        sns.histplot(df[var], kde=True, ax=ax)
        st.pyplot(fig)

        st.subheader("Comparacion: estres, ansiedad y adiccion")
        st.caption("Analisis exploratorio, no es un diagnostico clinico.")
        fig2, axes = plt.subplots(1, 3, figsize=(12, 3))
        for ax, col in zip(axes, ESCALAS):
            sns.histplot(df[col], kde=True, ax=ax)
            ax.set_title(col)
        st.pyplot(fig2)

    # Item 6
    with tabs[5]:
        st.header("Analisis de variables categoricas")
        _, _, categoricas = analyzer.clasificar_variables()
        var = st.selectbox("Variable categorica:", categoricas)
        conteo = df[var].value_counts()
        col1, col2 = st.columns(2)
        col1.dataframe(conteo)
        col2.dataframe((df[var].value_counts(normalize=True) * 100).round(2))

        fig, ax = plt.subplots(figsize=(6, 3))
        sns.barplot(x=conteo.index, y=conteo.values, ax=ax)
        st.pyplot(fig)

    # Item 7
    with tabs[6]:
        st.header("Analisis bivariado (numerico vs categorico)")
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(5, 3))
            sns.boxplot(x="depression_label", y="daily_social_media_hours", data=df, ax=ax)
            st.pyplot(fig)
        with col2:
            fig2, ax2 = plt.subplots(figsize=(5, 3))
            sns.boxplot(x="depression_label", y="sleep_hours", data=df, ax=ax2)
            st.pyplot(fig2)

        var_extra = st.selectbox("Otra variable:", ["academic_performance", "physical_activity"])
        st.dataframe(analyzer.comparar_grupos(var_extra, "depression_label"))

    # Item 8
    with tabs[7]:
        st.header("Analisis bivariado (categorico vs categorico)")
        col1, col2 = st.columns(2)
        with col1:
            st.write("platform_usage vs depression_label")
            st.dataframe(pd.crosstab(df["platform_usage"], df["depression_label"], normalize="index").round(2))
        with col2:
            st.write("social_interaction_level vs depression_label")
            st.dataframe(pd.crosstab(df["social_interaction_level"], df["depression_label"], normalize="index").round(2))

        st.write("gender vs platform_usage")
        tabla3 = pd.crosstab(df["gender"], df["platform_usage"], normalize="index").round(2)
        fig, ax = plt.subplots(figsize=(6, 3))
        sns.heatmap(tabla3, annot=True, cmap="Blues", ax=ax)
        st.pyplot(fig)

    # Item 9
    with tabs[8]:
        st.header("Filtros dinamicos")
        col1, col2 = st.columns(2)
        with col1:
            edad = st.slider("Edad:", int(df["age"].min()), int(df["age"].max()),
                              (int(df["age"].min()), int(df["age"].max())))
            generos = st.multiselect("Genero:", df["gender"].unique().tolist(), default=df["gender"].unique().tolist())
        with col2:
            plataformas = st.multiselect("Plataforma:", df["platform_usage"].unique().tolist(),
                                          default=df["platform_usage"].unique().tolist())
            interaccion = st.multiselect("Interaccion social:", df["social_interaction_level"].unique().tolist(),
                                          default=df["social_interaction_level"].unique().tolist())

        df_filtrado = analyzer.filtrar(edad, generos, plataformas, interaccion)
        st.write(f"Registros filtrados: {len(df_filtrado)} de {len(df)}")

        col3, col4 = st.columns(2)
        with col3:
            var_bienestar = st.selectbox("Variable de bienestar:", ESCALAS + ["depression_label"])
        with col4:
            var_habito = st.selectbox("Variable de habito:", HABITOS)

        if len(df_filtrado) > 0:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.scatterplot(x=var_habito, y=var_bienestar, data=df_filtrado, ax=ax)
            st.pyplot(fig)

    # Item 10
    with tabs[9]:
        st.header("Hallazgos clave")
        _, numericas, _ = analyzer.clasificar_variables()
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(df[numericas].corr().round(2), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

        st.subheader("Insights")
        st.write(
            "- Mas horas de redes sociales se relacionan con menos horas de sueño.\n"
            "- La proporcion de depression_label cambia segun la plataforma usada.\n"
            "- Los niveles de estres y ansiedad se concentran en rangos medios-altos."
        )

        st.subheader("Recomendaciones")
        st.write(
            "- Fomentar el bienestar digital en los grupos con mayor uso de redes.\n"
            "- Monitorear los niveles de estres y ansiedad reportados.\n"
            "- Profundizar el analisis con apoyo de profesionales de la salud."
        )


# ---------------- MODULO 4: CONCLUSIONES ----------------
elif modulo == "Conclusiones":
    st.title("Conclusiones Finales")

    st.write(
        "1. Los adolescentes con mas horas diarias en redes sociales reportan menos "
        "horas de sueño (Item 7).\n\n"
        "2. La proporcion de depression_label varia segun la plataforma utilizada "
        "(Item 8).\n\n"
        "3. Una parte importante de los adolescentes se ubica en niveles medios-altos "
        "de estres y ansiedad (Item 5).\n\n"
        "4. El nivel de interaccion social se asocia con distintas proporciones de la "
        "etiqueta de depresion (Item 8).\n\n"
        "5. Los filtros dinamicos permiten identificar subgrupos especificos donde el "
        "bienestar requiere mas atencion, sin necesidad de un modelo predictivo (Item 9)."
    )

    st.info("Estas conclusiones son exploratorias y no reemplazan la evaluacion de un profesional de la salud.")
