
import streamlit as st
import pandas as pd
from recommender import KNNCollabRecommender

st.set_page_config(page_title="Recomendador KNN", page_icon="🛒", layout="wide")
st.title("🛒 Recomendador de Productos — Filtrado Colaborativo (KNN)")

st.markdown(
    "Sube el archivo CSV de transacciones (debe contener las columnas **Customer_Name** y **Product**). "
    "La columna **Product** debe ser una lista por transacción, por ejemplo: `['Bread','Milk']`."
)

uploaded = st.file_uploader("Carga tu CSV", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    st.success(f"Cargado: {uploaded.name} — {len(df):,} filas")
    with st.expander("Ver muestra del dataset"):
        st.dataframe(df.head(20))

    model = KNNCollabRecommender()
    with st.spinner("Entrenando modelo KNN..."):
        model.fit_from_transactions(df)
    st.success("Modelo entrenado ✅")

    cliente = st.selectbox("Selecciona un cliente", options=list(model.pivot.index))
    col1, col2 = st.columns(2)
    with col1:
        top_k = st.slider("Top recomendaciones", 1, 20, 5)
    with col2:
        n_neighbors = st.slider("Vecinos a considerar (incluye el propio)", 2, 30, 6)

    if st.button("Generar recomendaciones"):
        recs = model.recommend(cliente, top_k=top_k, n_neighbors=n_neighbors)
        if len(recs) == 0:
            st.info("No hay recomendaciones nuevas para este cliente.")
        else:
            st.subheader("Resultados")
            st.table(recs.rename("frecuencia_en_vecinos"))
else:
    st.warning("Esperando que subas el CSV…")
