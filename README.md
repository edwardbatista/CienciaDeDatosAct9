# Ciencia de Datos Actividad 9: Recomendador de Productos (KNN) — Streamlit
## Archivos
- `app.py`: interfaz de usuario (Streamlit)
- `recommender.py`: clase con el pipeline del recomendador
- `requirements.txt`: dependencias

# Informe Técnico — Sistema de Recomendación de Productos (Filtrado Colaborativo KNN)
1. Introducción

Este proyecto forma parte de la Práctica Integradora de Ciencia de Datos, cuyo objetivo es aplicar los conceptos de minería de datos, análisis exploratorio y aprendizaje automático en un caso real de negocio.
El contexto es una empresa de comercio electrónico que busca mejorar la experiencia del cliente mediante un sistema de recomendación de productos, capaz de sugerir artículos que otros usuarios con comportamientos similares ya han comprado.

2. Descripción del Dataset

El dataset contiene información de transacciones históricas realizadas por distintos clientes.
Cada registro representa una compra completa (carrito de compra) e incluye:

Customer_Name: identificador del cliente.

Product: lista de productos adquiridos en esa compra.

Total_Items y Total_Cost: tamaño y valor del pedido.

Payment_Method, City, Store_Type, Promotion, entre otras variables contextuales.

El dataset fue procesado para crear una matriz binaria Cliente–Producto, donde cada celda indica si un cliente ha comprado o no un determinado producto.

3. Metodología
a) Análisis Exploratorio (EDA)

Se realizó un análisis inicial para comprender la distribución de variables, los productos más vendidos y las diferencias de comportamiento entre tipos de clientes y ciudades.
Se utilizaron gráficos de barras, histogramas y estadísticas descriptivas con pandas y matplotlib.

b) Preprocesamiento

Conversión de la columna Product (string → lista).

Agrupación de productos por cliente (conjunto único por usuario).

Creación de matriz binaria mediante MultiLabelBinarizer.

c) Modelado — Filtrado Colaborativo

Se implementó un modelo KNN (K-Nearest Neighbors) con métrica coseno, entrenado sobre la matriz Cliente–Producto.
El sistema identifica clientes similares en base a sus compras y recomienda productos que esos vecinos adquirieron, pero el cliente objetivo aún no.

d) Evaluación

Se aplicó una métrica simple de Precision@K utilizando un holdout temporal, midiendo qué proporción de los productos recomendados coinciden con los realmente comprados en futuras transacciones.

4. Despliegue Automatizado

El modelo se integró en una interfaz interactiva usando Streamlit, permitiendo:

Subir el archivo CSV de transacciones.

Entrenar el modelo directamente desde la interfaz.

Seleccionar un cliente y visualizar las recomendaciones personalizadas.

El sistema fue desplegado en Streamlit Cloud, configurando Python 3.10 y dependencias optimizadas (pandas, numpy, scikit-learn, streamlit).
La aplicación puede ejecutarse localmente con:
## Cómo ejecutar localmente
```bash
pip install -r requirements.txt
streamlit run app.py
```
5. Resultados y Conclusiones

El modelo demostró ser capaz de identificar relaciones entre clientes con comportamientos de compra similares, generando recomendaciones relevantes y personalizadas.
Este enfoque de filtrado colaborativo puede integrarse fácilmente en entornos de e-commerce para incrementar el ticket promedio, fomentar el cross-selling y mejorar la satisfacción del cliente.

En futuros desarrollos se propone:

Incorporar ponderación por similitud (1 − distancia).

Combinar reglas de asociación (Apriori) para manejar cold start.

Personalizar recomendaciones según temporada, ciudad o promociones activas.

6. Tecnologías Utilizadas

Lenguaje: Python 3.10

Librerías principales: pandas, scikit-learn, numpy, streamlit

Entorno: Google Colab (entrenamiento), Streamlit Cloud (despliegue)

Control de versiones: Git y GitHub
