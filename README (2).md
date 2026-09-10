# Teen Mental Health Dataset - EDA

Caso de Estudio N4 - Especializacion en Python for Analytics

## Descripcion del proyecto

Aplicacion en Streamlit para hacer un Analisis Exploratorio de Datos (EDA) del
dataset Teen_Mental_Health_Dataset.csv. Se analizan habitos digitales,
descanso, actividad fisica, interaccion social y variables de bienestar en
adolescentes de 13 a 19 anios. No se construyen modelos predictivos.

## Capturas de la aplicacion

![Home](<img width="1880" height="753" alt="home" src="https://github.com/user-attachments/assets/47138868-1f17-4b00-9558-b5fb73220af0" />
)
![Carga del Dataset](<img width="1890" height="950" alt="carga_dataset" src="https://github.com/user-attachments/assets/3e344848-c99a-445e-8eb2-85340483858b" />
)
![Info general](<img width="1856" height="897" alt="eda_info_general" src="https://github.com/user-attachments/assets/f41ba6e6-d52d-4a7e-8854-7ced15ea416d" />
)
![Distribuciones](<img width="1862" height="872" alt="eda_distribuciones" src="https://github.com/user-attachments/assets/faffa0aa-7621-47ed-aab2-19a0b2c3121e" />
)
![Conclusiones](<img width="1862" height="846" alt="conclusiones" src="https://github.com/user-attachments/assets/d44df1d6-e960-4cf5-88a6-089d5ef72a03" />
)

## Instrucciones de ejecucion

1. Clonar el repositorio:
   ```
   git clone https://github.com/j95395133-sys/PROYECT-N-2.git
   ```
2. Instalar dependencias:
   ```
   pip install -r requirements.txt
   ```
3. Ejecutar la app:
   ```
   streamlit run app.py
   ```
4. Subir el archivo Teen_Mental_Health_Dataset.csv en el modulo "Carga del Dataset".

## Variables principales

| Variable | Descripcion |
|---|---|
| age | Edad del adolescente (13 a 19 anios) |
| gender | Genero registrado |
| daily_social_media_hours | Horas diarias de uso de redes sociales |
| platform_usage | Plataforma usada: Instagram, TikTok o ambas |
| sleep_hours | Horas de sueño por dia |
| screen_time_before_sleep | Tiempo de pantalla antes de dormir |
| academic_performance | Indicador de rendimiento academico |
| physical_activity | Horas de actividad fisica |
| social_interaction_level | Nivel de interaccion social |
| stress_level | Nivel de estres (escala 1 a 10) |
| anxiety_level | Nivel de ansiedad (escala 1 a 10) |
| addiction_level | Nivel de dependencia (escala 1 a 10) |
| depression_label | Etiqueta binaria: 0 = ausencia, 1 = presencia |

## Links relevantes

- Repositorio GitHub: https://github.com/j95395133-sys/PROYECT-N-2
- Aplicacion desplegada: https://proyect-n-2-kkqj9pcm79knl4mk37wogb.streamlit.app/
