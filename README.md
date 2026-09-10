# 📊 Teen Mental Health – Análisis Exploratorio de Datos (EDA)

**Caso de Estudio N°4 – Especialización en Python for Analytics**

## 📝 Descripción del proyecto

Aplicación interactiva construida con **Streamlit** para realizar un Análisis
Exploratorio de Datos (EDA) sobre el dataset `Teen_Mental_Health_Dataset.csv`.
El objetivo **no** es construir modelos predictivos, sino analizar, limpiar,
transformar y visualizar los datos para identificar patrones exploratorios
entre hábitos digitales, descanso, actividad física, interacción social y las
variables de bienestar registradas en adolescentes de 13 a 19 años.

El proyecto aplica de forma integrada:
- Variables, tipos de datos y f-strings
- Funciones personalizadas
- Programación Orientada a Objetos (clase `DataAnalyzer`)
- NumPy y Pandas
- Visualización con Matplotlib y Seaborn
- Estadística descriptiva (media, mediana, cuartiles, dispersión, distribución)

## 🖼️ Capturas de la aplicación

> _Agrega aquí capturas de pantalla del Home, la carga del dataset y los
> ítems de análisis más relevantes del EDA._

## ⚙️ Instrucciones de ejecución

1. Clona este repositorio:
   ```bash
   git clone <URL_DEL_REPOSITORIO>
   cd <NOMBRE_DEL_REPOSITORIO>
   ```
2. Crea un entorno virtual (opcional pero recomendado):
   ```bash
   python -m venv venv
   source venv/bin/activate    # En Windows: venv\Scripts\activate
   ```
3. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```
4. Ejecuta la aplicación:
   ```bash
   streamlit run app.py
   ```
5. En el navegador, ve al módulo **"Carga del Dataset"** y sube el archivo
   `Teen_Mental_Health_Dataset.csv`.

## 📚 Descripción de las variables principales

| Variable | Descripción |
|---|---|
| `age` | Edad del adolescente, entre 13 y 19 años |
| `gender` | Género registrado |
| `daily_social_media_hours` | Horas diarias de uso de redes sociales |
| `platform_usage` | Plataforma utilizada: Instagram, TikTok o ambas |
| `sleep_hours` | Horas de sueño por día |
| `screen_time_before_sleep` | Tiempo de pantalla antes de dormir, en horas |
| `academic_performance` | Indicador de rendimiento académico |
| `physical_activity` | Horas de actividad física |
| `social_interaction_level` | Nivel de interacción social: bajo, medio o alto |
| `stress_level` | Nivel de estrés en escala de 1 a 10 |
| `anxiety_level` | Nivel de ansiedad en escala de 1 a 10 |
| `addiction_level` | Nivel de dependencia o uso problemático en escala de 1 a 10 |
| `depression_label` | Etiqueta binaria: 0 = ausencia, 1 = presencia de la condición etiquetada |

⚠️ El análisis se mantiene en un plano educativo y exploratorio: los
resultados no constituyen un diagnóstico clínico ni sustituyen la valoración
de profesionales de la salud.

## 🔗 Links relevantes

- Repositorio GitHub: `<agregar link>`
- Aplicación desplegada (Streamlit Community Cloud): `<agregar link>`

## 👤 Autor

- **Nombre completo:** [Tu Nombre Completo]
- **Curso / Especialización:** Especialización en Python for Analytics
- **Año:** 2026
