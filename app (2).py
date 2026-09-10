"""
Caso de Estudio N°4 - Especialización en Python for Analytics
Análisis Exploratorio de Datos (EDA) - Teen Mental Health Dataset
Aplicación interactiva construida con Streamlit.

Autor: [Tu Nombre Completo]
Curso / Especialización: Especialización en Python for Analytics
Año: 2026
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# CONFIGURACIÓN GENERAL DE LA PÁGINA
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Teen Mental Health - EDA",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

sns.set_style("whitegrid")

# ---------------------------------------------------------------------------
# FUNCIÓN PERSONALIZADA: clasificación de tipo de variable
# ---------------------------------------------------------------------------
def clasificar_variable(serie: pd.Series) -> str:
    """
    Recibe una columna (pd.Series) y devuelve un string indicando
    si es 'Numérica' o 'Categórica', según su dtype.
    """
    if pd.api.types.is_numeric_dtype(serie):
        return "Numérica"
    else:
        return "Categórica"


# ---------------------------------------------------------------------------
# CLASE PRINCIPAL: DataAnalyzer (Programación Orientada a Objetos)
# ---------------------------------------------------------------------------
class DataAnalyzer:
    """
    Encapsula toda la lógica de carga, validación, clasificación,
    estadística descriptiva, visualización y filtrado del dataset
    de salud mental en adolescentes.
    """

    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe.copy()

    # ---------- Carga y validación ----------
    @staticmethod
    def cargar_csv(archivo) -> "DataAnalyzer | None":
        """
        Carga un archivo CSV subido por el usuario, detectando automáticamente
        el separador (coma, punto y coma, tabulador) y validando que no esté vacío.
        """
        try:
            # Contar líneas reales del archivo para poder comparar luego
            contenido = archivo.getvalue()
            n_lineas_archivo = contenido.count(b"\n")
            archivo.seek(0)

            # sep=None + engine="python" hace que pandas detecte el separador solo
            df = pd.read_csv(archivo, sep=None, engine="python")

            if df.empty:
                st.error("⚠️ El archivo cargado está vacío.")
                return None

            # Aviso si el número de filas leídas es mucho menor a las líneas del archivo
            if n_lineas_archivo > 0 and len(df) < n_lineas_archivo * 0.5:
                st.warning(
                    f"⚠️ El archivo tiene aproximadamente **{n_lineas_archivo}** líneas, "
                    f"pero solo se interpretaron **{len(df)}** filas como registros. "
                    "Esto suele deberse a comillas sin cerrar o saltos de línea dentro de "
                    "algún campo de texto del CSV. Revisa el archivo original en un editor "
                    "de texto plano."
                )

            return DataAnalyzer(df)
        except Exception as e:
            st.error(f"⚠️ Ocurrió un error al leer el archivo: {e}")
            return None

    # ---------- Información general ----------
    def info_general(self):
        n_filas, n_columnas = self.df.shape
        nulos = self.df.isnull().sum()
        duplicados = self.df.duplicated().sum()
        tipos = self.df.dtypes
        return n_filas, n_columnas, nulos, duplicados, tipos

    # ---------- Clasificación de variables ----------
    def clasificar_variables(self):
        clasificacion = {col: clasificar_variable(self.df[col]) for col in self.df.columns}
        clasif_df = pd.DataFrame(
            {"Variable": clasificacion.keys(), "Tipo": clasificacion.values()}
        )
        numericas = clasif_df[clasif_df["Tipo"] == "Numérica"]["Variable"].tolist()
        categoricas = clasif_df[clasif_df["Tipo"] == "Categórica"]["Variable"].tolist()
        return clasif_df, numericas, categoricas

    # ---------- Estadísticas descriptivas ----------
    def estadisticas_descriptivas(self, columnas=None):
        if columnas:
            return self.df[columnas].describe()
        return self.df.describe()

    # ---------- Valores faltantes ----------
    def analisis_faltantes(self):
        total = self.df.isnull().sum()
        porcentaje = (total / len(self.df)) * 100
        resumen = pd.DataFrame({"Nulos": total, "Porcentaje (%)": porcentaje.round(2)})
        return resumen[resumen["Nulos"] >= 0].sort_values("Nulos", ascending=False)

    # ---------- Filtros ----------
    def filtrar_datos(self, age_range=None, generos=None, plataformas=None, interacciones=None):
        df_filtrado = self.df.copy()
        if age_range and "age" in df_filtrado.columns:
            df_filtrado = df_filtrado[
                (df_filtrado["age"] >= age_range[0]) & (df_filtrado["age"] <= age_range[1])
            ]
        if generos and "gender" in df_filtrado.columns:
            df_filtrado = df_filtrado[df_filtrado["gender"].isin(generos)]
        if plataformas and "platform_usage" in df_filtrado.columns:
            df_filtrado = df_filtrado[df_filtrado["platform_usage"].isin(plataformas)]
        if interacciones and "social_interaction_level" in df_filtrado.columns:
            df_filtrado = df_filtrado[df_filtrado["social_interaction_level"].isin(interacciones)]
        return df_filtrado

    # ---------- Comparación entre grupos ----------
    def comparar_por_grupo(self, variable_numerica: str, variable_categorica: str):
        return self.df.groupby(variable_categorica)[variable_numerica].agg(
            ["mean", "median", "std", "count"]
        ).round(2)


# ---------------------------------------------------------------------------
# DICCIONARIO DE VARIABLES (para mostrar contexto al usuario)
# ---------------------------------------------------------------------------
DICCIONARIO_VARIABLES = {
    "age": "Edad del adolescente, entre 13 y 19 años",
    "gender": "Género registrado",
    "daily_social_media_hours": "Horas diarias de uso de redes sociales",
    "platform_usage": "Plataforma utilizada: Instagram, TikTok o ambas",
    "sleep_hours": "Horas de sueño por día",
    "screen_time_before_sleep": "Tiempo de pantalla antes de dormir, en horas",
    "academic_performance": "Indicador de rendimiento académico",
    "physical_activity": "Horas de actividad física",
    "social_interaction_level": "Nivel de interacción social: bajo, medio o alto",
    "stress_level": "Nivel de estrés en escala de 1 a 10",
    "anxiety_level": "Nivel de ansiedad en escala de 1 a 10",
    "addiction_level": "Nivel de dependencia o uso problemático en escala de 1 a 10",
    "depression_label": "Etiqueta binaria del dataset: 0 = ausencia y 1 = presencia de la condición etiquetada",
}

ESCALAS_BIENESTAR = ["stress_level", "anxiety_level", "addiction_level"]
VARIABLES_HABITOS = ["daily_social_media_hours", "sleep_hours", "screen_time_before_sleep", "physical_activity"]


# ---------------------------------------------------------------------------
# INICIALIZACIÓN DEL ESTADO DE SESIÓN
# ---------------------------------------------------------------------------
if "df" not in st.session_state:
    st.session_state.df = None


# ---------------------------------------------------------------------------
# SIDEBAR - NAVEGACIÓN PRINCIPAL
# ---------------------------------------------------------------------------
st.sidebar.title("📊 Menú de Navegación")
modulo = st.sidebar.radio(
    "Selecciona un módulo:",
    ["🏠 Home", "📁 Carga del Dataset", "🔍 Análisis Exploratorio (EDA)", "✅ Conclusiones"],
)
st.sidebar.markdown("---")
st.sidebar.info(
    "Este proyecto es de carácter educativo y exploratorio. "
    "Los resultados no constituyen un diagnóstico clínico."
)

# ---------------------------------------------------------------------------
# MÓDULO 1: HOME
# ---------------------------------------------------------------------------
if modulo == "🏠 Home":
    st.title("📊 Teen Mental Health Dataset - Análisis Exploratorio de Datos")
    st.markdown("### Especialización en Python for Analytics — Caso de Estudio N°4")

    st.markdown(
        """
        El objetivo de esta aplicación es **analizar, limpiar, transformar y visualizar**
        los datos del dataset *Teen Mental Health*, con el fin de identificar patrones
        exploratorios entre hábitos digitales, descanso, actividad física, interacción
        social y las variables de bienestar registradas.

        **Este proyecto NO busca construir modelos predictivos**, sino ofrecer una
        herramienta de análisis descriptivo orientada a la toma de decisiones informadas.
        """
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("👤 Datos del autor")
        st.markdown(
            """
            - **Nombre completo:** [Tu Nombre Completo]
            - **Curso / Especialización:** Especialización en Python for Analytics
            - **Año:** 2026
            """
        )
    with col2:
        st.subheader("🛠️ Tecnologías utilizadas")
        st.markdown(
            """
            - Python 🐍
            - Pandas & NumPy
            - Streamlit
            - Matplotlib & Seaborn
            - Programación Orientada a Objetos (POO)
            """
        )

    st.subheader("📄 Sobre el dataset")
    st.markdown(
        f"""
        El archivo **Teen_Mental_Health_Dataset.csv** contiene **1,200 registros** y
        **13 variables** sobre adolescentes de 13 a 19 años. Incluye información de uso
        diario de redes sociales, plataforma utilizada, horas de sueño, tiempo de pantalla
        antes de dormir, rendimiento académico, actividad física e interacción social, así
        como escalas de estrés, ansiedad y nivel de dependencia, además de la etiqueta
        binaria `depression_label`.

        ⚠️ El análisis se mantiene en un plano **educativo y exploratorio**: los resultados
        no constituyen un diagnóstico clínico ni sustituyen la valoración de profesionales
        de la salud.
        """
    )

    with st.expander("📚 Ver diccionario de variables"):
        st.dataframe(
            pd.DataFrame(
                {"Variable": DICCIONARIO_VARIABLES.keys(), "Descripción": DICCIONARIO_VARIABLES.values()}
            ),
            use_container_width=True,
        )

# ---------------------------------------------------------------------------
# MÓDULO 2: CARGA DEL DATASET
# ---------------------------------------------------------------------------
elif modulo == "📁 Carga del Dataset":
    st.title("📁 Carga del Dataset")
    st.markdown("Sube el archivo `Teen_Mental_Health_Dataset.csv` para comenzar el análisis.")

    archivo = st.file_uploader("Selecciona el archivo CSV", type=["csv"])

    if archivo is not None:
        analyzer = DataAnalyzer.cargar_csv(archivo)
        if analyzer is not None:
            st.session_state.df = analyzer.df
            st.success("✅ Archivo cargado correctamente.")

            n_filas, n_columnas = analyzer.df.shape
            col1, col2 = st.columns(2)
            col1.metric("Filas", f"{n_filas:,}")
            col2.metric("Columnas", n_columnas)

            st.subheader("👀 Vista previa del dataset")
            st.dataframe(analyzer.df.head(10), use_container_width=True)

            if st.checkbox("Mostrar nombres y tipos de columnas"):
                st.write(analyzer.df.dtypes)
    else:
        st.warning("⏳ Aún no se ha cargado ningún archivo. Por favor, sube el CSV para continuar.")


# ---------------------------------------------------------------------------
# MÓDULO 3: ANÁLISIS EXPLORATORIO DE DATOS (EDA)
# ---------------------------------------------------------------------------
elif modulo == "🔍 Análisis Exploratorio (EDA)":
    st.title("🔍 Análisis Exploratorio de Datos (EDA)")

    if st.session_state.df is None:
        st.error("🚫 Debes cargar el dataset en el módulo 'Carga del Dataset' antes de continuar.")
        st.stop()

    analyzer = DataAnalyzer(st.session_state.df)
    df = analyzer.df

    tabs = st.tabs([
        "1️⃣ Info general", "2️⃣ Clasif. variables", "3️⃣ Estadísticas",
        "4️⃣ Faltantes", "5️⃣ Distribuciones", "6️⃣ Categóricas",
        "7️⃣ Bivariado num-cat", "8️⃣ Bivariado cat-cat",
        "9️⃣ Filtros dinámicos", "🔟 Hallazgos clave",
    ])

    # ----- ÍTEM 1: Información general -----
    with tabs[0]:
        st.header("Ítem 1: Información general del dataset")
        n_filas, n_columnas, nulos, duplicados, tipos = analyzer.info_general()

        col1, col2, col3 = st.columns(3)
        col1.metric("Total de filas", f"{n_filas:,}")
        col2.metric("Total de columnas", n_columnas)
        col3.metric("Registros duplicados", int(duplicados))

        st.subheader("Tipos de datos por columna")
        st.dataframe(tipos.rename("Tipo de dato").astype(str), use_container_width=True)

        st.subheader("Conteo de valores nulos por columna")
        st.dataframe(nulos.rename("Valores nulos"), use_container_width=True)

        st.markdown(
            f"El dataset tiene **{n_filas}** registros y **{n_columnas}** variables. "
            f"Se detectaron **{int(duplicados)}** registros duplicados y "
            f"**{int(nulos.sum())}** valores nulos en total."
        )

    # ----- ÍTEM 2: Clasificación de variables -----
    with tabs[1]:
        st.header("Ítem 2: Clasificación de variables")
        clasif_df, numericas, categoricas = analyzer.clasificar_variables()

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Tabla de clasificación")
            st.dataframe(clasif_df, use_container_width=True)
        with col2:
            st.subheader("Conteo por tipo")
            conteo_tipo = clasif_df["Tipo"].value_counts()
            st.bar_chart(conteo_tipo)
            st.markdown(
                f"Se identificaron **{len(numericas)}** variables numéricas y "
                f"**{len(categoricas)}** variables categóricas."
            )

    # ----- ÍTEM 3: Estadísticas descriptivas -----
    with tabs[2]:
        st.header("Ítem 3: Estadísticas descriptivas")
        _, numericas, _ = analyzer.clasificar_variables()
        desc = analyzer.estadisticas_descriptivas(numericas)
        st.dataframe(desc.round(2), use_container_width=True)

        st.subheader("Detección preliminar de valores extremos (boxplots)")
        var_box = st.selectbox("Selecciona una variable numérica:", numericas, key="box_item3")
        fig, ax = plt.subplots(figsize=(6, 3))
        sns.boxplot(x=df[var_box], ax=ax, color="#4C72B0")
        ax.set_title(f"Distribución y valores atípicos de {var_box}")
        st.pyplot(fig)

        media = df[var_box].mean()
        mediana = df[var_box].median()
        q1, q3 = df[var_box].quantile([0.25, 0.75])
        st.markdown(
            f"Para **{var_box}**: media = **{media:.2f}**, mediana = **{mediana:.2f}**, "
            f"rango intercuartílico (IQR) = **{q1:.2f} - {q3:.2f}**. "
            "Una diferencia notable entre media y mediana puede indicar asimetría en la distribución."
        )

    # ----- ÍTEM 4: Valores faltantes -----
    with tabs[3]:
        st.header("Ítem 4: Análisis de valores faltantes")
        resumen_faltantes = analyzer.analisis_faltantes()
        st.dataframe(resumen_faltantes, use_container_width=True)

        if resumen_faltantes["Nulos"].sum() > 0:
            fig, ax = plt.subplots(figsize=(8, 3))
            resumen_faltantes[resumen_faltantes["Nulos"] > 0]["Nulos"].plot(kind="bar", ax=ax, color="tomato")
            ax.set_ylabel("Cantidad de nulos")
            st.pyplot(fig)
            st.markdown(
                "Se recomienda evaluar imputación o eliminación de registros según el "
                "porcentaje de valores faltantes por variable."
            )
        else:
            st.success(
                "✅ El dataset no presenta valores faltantes. Por ello, el análisis puede "
                "enfocarse en la validación de tipos, distribuciones y comparación entre grupos."
            )

    # ----- ÍTEM 5: Distribución de variables numéricas -----
    with tabs[4]:
        st.header("Ítem 5: Distribución de variables numéricas")
        _, numericas, _ = analyzer.clasificar_variables()

        col1, col2 = st.columns([1, 2])
        with col1:
            var_hist = st.selectbox("Selecciona una variable:", numericas, key="hist_item5")
        with col2:
            fig, ax = plt.subplots(figsize=(6, 3))
            sns.histplot(df[var_hist], kde=True, ax=ax, color="#55A868")
            ax.set_title(f"Distribución de {var_hist}")
            st.pyplot(fig)

        st.subheader("Comparación de escalas: estrés, ansiedad y adicción")
        st.caption(
            "⚠️ Esta comparación es exploratoria y NO debe interpretarse como un diagnóstico clínico."
        )
        fig2, axes = plt.subplots(1, 3, figsize=(14, 3.5))
        for ax, col in zip(axes, ESCALAS_BIENESTAR):
            sns.histplot(df[col], kde=True, ax=ax, color="#C44E52")
            ax.set_title(col)
        st.pyplot(fig2)

        st.markdown(
            "La forma de cada distribución permite observar si la mayoría de adolescentes "
            "se concentra en niveles bajos, medios o altos de estrés, ansiedad y dependencia, "
            "así como la presencia de posibles valores extremos."
        )

    # ----- ÍTEM 6: Análisis de variables categóricas -----
    with tabs[5]:
        st.header("Ítem 6: Análisis de variables categóricas")
        _, _, categoricas = analyzer.clasificar_variables()

        var_cat = st.selectbox("Selecciona una variable categórica:", categoricas, key="cat_item6")
        conteo = df[var_cat].value_counts()
        proporcion = (df[var_cat].value_counts(normalize=True) * 100).round(2)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Conteo")
            st.dataframe(conteo.rename("Frecuencia"), use_container_width=True)
        with col2:
            st.subheader("Proporción (%)")
            st.dataframe(proporcion.rename("Porcentaje"), use_container_width=True)

        fig, ax = plt.subplots(figsize=(6, 3))
        sns.barplot(x=conteo.index, y=conteo.values, ax=ax, palette="viridis")
        ax.set_ylabel("Frecuencia")
        ax.set_title(f"Distribución de {var_cat}")
        st.pyplot(fig)

        categoria_top = conteo.idxmax()
        st.markdown(
            f"La categoría más frecuente en **{var_cat}** es **{categoria_top}**, "
            f"con una proporción de **{proporcion.max()}%** del total de registros."
        )

    # ----- ÍTEM 7: Análisis bivariado (numérico vs categórico) -----
    with tabs[6]:
        st.header("Ítem 7: Análisis bivariado (numérico vs categórico)")

        if "depression_label" in df.columns:
            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots(figsize=(5, 3.5))
                sns.boxplot(x="depression_label", y="daily_social_media_hours", data=df, ax=ax, palette="Set2")
                ax.set_title("Horas de redes sociales según depression_label")
                st.pyplot(fig)
            with col2:
                fig2, ax2 = plt.subplots(figsize=(5, 3.5))
                sns.boxplot(x="depression_label", y="sleep_hours", data=df, ax=ax2, palette="Set2")
                ax2.set_title("Horas de sueño según depression_label")
                st.pyplot(fig2)

            var_extra = st.selectbox(
                "Analiza también:", ["academic_performance", "physical_activity"], key="item7_extra"
            )
            fig3, ax3 = plt.subplots(figsize=(6, 3.5))
            sns.boxplot(x="depression_label", y=var_extra, data=df, ax=ax3, palette="Set2")
            ax3.set_title(f"{var_extra} según depression_label")
            st.pyplot(fig3)

            comparacion = analyzer.comparar_por_grupo(var_extra, "depression_label")
            st.dataframe(comparacion, use_container_width=True)
            st.markdown(
                f"Al comparar **{var_extra}** entre los grupos de `depression_label`, se observan "
                "diferencias en las medias y medianas que pueden orientar hipótesis exploratorias, "
                "sin implicar relaciones causales."
            )
        else:
            st.warning("La columna 'depression_label' no está presente en el dataset cargado.")

    # ----- ÍTEM 8: Análisis bivariado (categórico vs categórico) -----
    with tabs[7]:
        st.header("Ítem 8: Análisis bivariado (categórico vs categórico)")

        if "depression_label" in df.columns:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("platform_usage vs depression_label")
                tabla1 = pd.crosstab(df["platform_usage"], df["depression_label"], normalize="index").round(2)
                st.dataframe(tabla1, use_container_width=True)
                fig, ax = plt.subplots(figsize=(5, 3))
                tabla1.plot(kind="bar", stacked=True, ax=ax, colormap="coolwarm")
                st.pyplot(fig)

            with col2:
                st.subheader("social_interaction_level vs depression_label")
                tabla2 = pd.crosstab(df["social_interaction_level"], df["depression_label"], normalize="index").round(2)
                st.dataframe(tabla2, use_container_width=True)
                fig2, ax2 = plt.subplots(figsize=(5, 3))
                tabla2.plot(kind="bar", stacked=True, ax=ax2, colormap="coolwarm")
                st.pyplot(fig2)

            st.subheader("gender vs platform_usage")
            tabla3 = pd.crosstab(df["gender"], df["platform_usage"], normalize="index").round(2)
            st.dataframe(tabla3, use_container_width=True)
            fig3, ax3 = plt.subplots(figsize=(6, 3))
            sns.heatmap(tabla3, annot=True, cmap="Blues", ax=ax3)
            st.pyplot(fig3)

            st.markdown(
                "Las tablas cruzadas muestran las proporciones relativas entre categorías, "
                "permitiendo identificar combinaciones donde la presencia de la etiqueta "
                "`depression_label` es proporcionalmente mayor."
            )
        else:
            st.warning("La columna 'depression_label' no está presente en el dataset cargado.")

    # ----- ÍTEM 9: Análisis basado en parámetros seleccionados -----
    with tabs[8]:
        st.header("Ítem 9: Análisis basado en parámetros seleccionados por el usuario")

        col1, col2 = st.columns(2)
        with col1:
            edad_min, edad_max = int(df["age"].min()), int(df["age"].max())
            rango_edad = st.slider("Rango de edad:", edad_min, edad_max, (edad_min, edad_max))
            generos_sel = st.multiselect("Género:", df["gender"].unique().tolist(), default=df["gender"].unique().tolist())
        with col2:
            plataformas_sel = st.multiselect(
                "Plataforma:", df["platform_usage"].unique().tolist(), default=df["platform_usage"].unique().tolist()
            )
            interacciones_sel = st.multiselect(
                "Nivel de interacción social:",
                df["social_interaction_level"].unique().tolist(),
                default=df["social_interaction_level"].unique().tolist(),
            )

        df_filtrado = analyzer.filtrar_datos(rango_edad, generos_sel, plataformas_sel, interacciones_sel)
        st.info(f"🔎 Registros después de aplicar filtros: **{len(df_filtrado)}** de {len(df)}")

        col3, col4 = st.columns(2)
        with col3:
            var_bienestar = st.selectbox("Variable de bienestar:", ESCALAS_BIENESTAR + ["depression_label"])
        with col4:
            var_habito = st.selectbox("Variable de hábitos digitales:", VARIABLES_HABITOS)

        if len(df_filtrado) > 0:
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.scatterplot(x=var_habito, y=var_bienestar, data=df_filtrado, alpha=0.6, ax=ax)
            ax.set_title(f"{var_bienestar} vs {var_habito} (datos filtrados)")
            st.pyplot(fig)

            correlacion = df_filtrado[[var_habito, var_bienestar]].corr().iloc[0, 1]
            st.markdown(
                f"Con los filtros aplicados, la correlación entre **{var_habito}** y "
                f"**{var_bienestar}** es de **{correlacion:.2f}**."
            )
        else:
            st.warning("No hay registros que cumplan con los filtros seleccionados.")

    # ----- ÍTEM 10: Hallazgos clave -----
    with tabs[9]:
        st.header("Ítem 10: Hallazgos clave")

        _, numericas, _ = analyzer.clasificar_variables()
        st.subheader("Mapa de correlación entre variables numéricas")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(df[numericas].corr().round(2), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

        st.subheader("💡 Insights principales")
        st.markdown(
            """
            - Las variables de bienestar (estrés, ansiedad, adicción) muestran relaciones
              observables con el tiempo de uso de redes sociales y las horas de sueño.
            - Los grupos con mayor uso diario de redes sociales tienden a presentar
              menores horas de sueño reportadas.
            - La proporción de `depression_label` varía según la plataforma utilizada y el
              nivel de interacción social, lo que sugiere focos de atención diferenciados.
            """
        )

        st.subheader("📌 Recomendaciones orientadas a la toma de decisiones")
        st.markdown(
            """
            - Priorizar programas de bienestar digital en los segmentos con mayor uso de
              redes sociales y menor cantidad de horas de sueño.
            - Monitorear de forma continua los niveles de estrés y ansiedad reportados,
              sin utilizar estos resultados como diagnóstico clínico.
            - Profundizar el análisis cualitativo en los grupos con mayor proporción de
              `depression_label`, en conjunto con profesionales de la salud.
            """
        )


# ---------------------------------------------------------------------------
# MÓDULO 4: CONCLUSIONES FINALES
# ---------------------------------------------------------------------------
elif modulo == "✅ Conclusiones":
    st.title("✅ Conclusiones Finales")

    if st.session_state.df is None:
        st.warning("Carga el dataset y explora el módulo EDA para contextualizar mejor las conclusiones.")

    st.markdown(
        """
        **1. Relación entre uso digital y descanso.**
        Los adolescentes con mayor número de horas diarias en redes sociales tienden a
        reportar menos horas de sueño, evidenciado en el análisis bivariado del Ítem 7.

        **2. Diferencias por plataforma.**
        La proporción de `depression_label` varía según la plataforma utilizada
        (Instagram, TikTok o ambas), como se observa en las tablas cruzadas del Ítem 8.

        **3. Concentración de niveles de estrés y ansiedad.**
        Las distribuciones del Ítem 5 muestran que una parte importante de los
        adolescentes se ubica en niveles medios-altos de estrés y ansiedad, lo que
        amerita atención preventiva, sin constituir un diagnóstico.

        **4. Interacción social como factor relevante.**
        El nivel de interacción social (bajo, medio, alto) se asocia con distintas
        proporciones de la etiqueta de depresión, según el Ítem 8, sugiriendo que el
        aislamiento social podría ser un factor a monitorear.

        **5. Valor del análisis exploratorio para la toma de decisiones.**
        El uso de filtros dinámicos (Ítem 9) permite identificar subgrupos específicos
        (por edad, género, plataforma) donde los indicadores de bienestar requieren mayor
        atención, apoyando decisiones informadas sin recurrir a modelos predictivos.
        """
    )

    st.info(
        "Estas conclusiones son de carácter exploratorio y educativo. No deben "
        "utilizarse como sustituto de la evaluación de profesionales de la salud mental."
    )
