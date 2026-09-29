"""
Referencia salarial de mercado para mujeres asalariadas en Argentina
TFM - Máster en Data Science, Big Data & Business Analytics (UCM) - Patricia Olguín

Uso:  streamlit run app/app.py      (desde la carpeta raíz del repositorio)
El modelo se genera con el notebook del TFM y se guarda en modelos/modelo_AR.joblib.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

CARPETA = Path(__file__).resolve().parent.parent / "modelos"
st.set_page_config(page_title="Referencia salarial de mercado", layout="wide")


@st.cache_resource
def cargar_modelo():
    return joblib.load(CARPETA / "modelo_AR.joblib")


if not (CARPETA / "modelo_AR.joblib").exists():
    st.error("No se encontró modelos/modelo_AR.joblib. Ejecute primero el notebook del TFM.")
    st.stop()
pk = cargar_modelo()
VARIABLES = pk["variables"]


def preparar(datos):
    """Ordena las columnas, aplica las categorías del entrenamiento y completa lo que falte con valores típicos."""
    X = pd.DataFrame(datos).copy()
    for v in VARIABLES:
        if v not in X:
            X[v] = pk["valores_tipicos"][v]
        if pk["tipos"][v] == "categórica":
            X[v] = pd.Categorical(X[v].astype(str), categories=pk["categorias"][v])
        else:
            X[v] = pd.to_numeric(X[v], errors="coerce")
    return X[VARIABLES]


def predecir(X):
    m = pk["mediana_salario_hora"]
    centro = pk["modelo"].predict(X)
    inf = pk["cuantil_inf"].predict(X) - pk["ajuste_banda"]
    sup = pk["cuantil_sup"].predict(X) + pk["ajuste_banda"]
    centro = np.clip(centro, inf, sup)
    return m * np.exp(centro), m * np.exp(inf), m * np.exp(sup)


def pesos(valor):
    return f"$ {valor:,.0f}".replace(",", ".")


st.sidebar.title("Referencia salarial de mercado")
st.sidebar.markdown(f"**Período de referencia:** {pk['periodo_referencia']}  \n"
                    f"**Salario horario mediano del período:** {pesos(pk['mediana_salario_hora'])}  \n"
                    f"**Banda:** {int(pk['nivel'] * 100)}% de cobertura (predicción conformal)")
st.sidebar.caption("Modelo entrenado con los microdatos de la Encuesta Permanente de Hogares (INDEC). Es una referencia "
                   "de mercado, no un veredicto sobre casos individuales.")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Simulador de perfil", "Auditoría de nómina", "Desempeño del modelo",
                                        "Hogar y cuidados", "Territorio"])

with tab1:
    st.subheader("¿Cuánto paga el mercado por este perfil?")
    valores = {}
    bloques = sorted(set(pk["bloques"].values()))
    columnas = st.columns(2)
    for i, bloque in enumerate(bloques):
        with columnas[i % 2].expander(bloque, expanded=i < 2):
            for v in [x for x in VARIABLES if pk["bloques"][x] == bloque and not x.endswith("_faltante")]:
                etiqueta, tipo, tipico = pk["descripciones"][v], pk["tipos"][v], pk["valores_tipicos"][v]
                if v == "exp_pot2":
                    continue
                if tipo == "categórica":
                    opciones = pk["categorias"][v]
                    valores[v] = st.selectbox(etiqueta, opciones, index=opciones.index(tipico) if tipico in opciones else 0)
                elif tipo == "binaria":
                    valores[v] = float(st.checkbox(etiqueta, value=bool(round(tipico))))
                else:
                    lo, hi = pk["rangos"][v]
                    valores[v] = st.slider(etiqueta, float(lo), float(hi), float(np.clip(tipico, lo, hi)))
    if "exp_pot2" in VARIABLES:
        valores["exp_pot2"] = valores.get("exp_pot", pk["valores_tipicos"].get("exp_pot", 0)) ** 2
    X = preparar([valores])
    centro, inf, sup = predecir(X)
    c1, c2, c3 = st.columns(3)
    c1.metric("Salario horario esperado", pesos(centro[0]))
    c2.metric("Mínimo de la banda", pesos(inf[0]))
    c3.metric("Máximo de la banda", pesos(sup[0]))
    st.caption(f"Equivale a un salario mensual a tiempo completo (40 horas semanales) de {pesos(centro[0] * 40 * 4.3)} "
               f"(banda: {pesos(inf[0] * 40 * 4.3)} a {pesos(sup[0] * 40 * 4.3)}).")
    contrib = pk["modelo"].predict(X, pred_contrib=True)[0][:-1]
    explicacion = pd.DataFrame({"variable": [pk["descripciones"][v] for v in VARIABLES],
                                "efecto (%)": 100 * (np.exp(contrib) - 1)})
    explicacion = explicacion.reindex(explicacion["efecto (%)"].abs().sort_values().index).tail(15)
    fig = px.bar(explicacion, x="efecto (%)", y="variable", orientation="h", color=explicacion["efecto (%)"] > 0,
                 color_discrete_map={True: "#2a9d8f", False: "#e76f51"},
                 title="¿Qué explica este resultado? Aporte de cada característica respecto del salario típico (SHAP)")
    fig.update_layout(showlegend=False, height=520, yaxis_title="")
    st.plotly_chart(fig, width="stretch")

with tab2:
    st.subheader("Auditoría de equidad: compare su nómina con la referencia de mercado")
    st.markdown("Suba un CSV **anonimizado** con una fila por empleada: `id_empleada`, `salario_horario` (en pesos) y las "
                "columnas del perfil. Las columnas que falten se completan con valores típicos.")
    plantilla = pd.DataFrame([{"id_empleada": "E001", "salario_horario": round(pk["mediana_salario_hora"], 2),
                               **{v: pk["valores_tipicos"][v] for v in VARIABLES}}])
    st.download_button("Descargar plantilla", plantilla.to_csv(index=False).encode(), "plantilla_nomina.csv")
    archivo = st.file_uploader("Nómina (CSV)", type="csv")
    ruta_demo = CARPETA / "nomina_demo_AR.csv"
    usar_demo = ruta_demo.exists() and st.checkbox(
        "Usar la nómina de demostración (simulada con registros de la EPH del 1T2026; no corresponde a ninguna empresa)")
    nomina = pd.read_csv(archivo) if archivo is not None else (pd.read_csv(ruta_demo) if usar_demo else None)
    if nomina is not None:
        centro, inf, sup = predecir(preparar(nomina))
        res = nomina[["id_empleada", "salario_horario"]].assign(esperado=centro, banda_min=inf, banda_max=sup)
        res["diferencia (%)"] = 100 * (res["salario_horario"] / res["esperado"] - 1)
        res["situación"] = np.select([res.salario_horario < res.banda_min, res.salario_horario > res.banda_max],
                                     ["Por debajo del mercado", "Por encima del mercado"], "Dentro de la banda")
        c1, c2, c3 = st.columns(3)
        c1.metric("Empleadas analizadas", len(res))
        c2.metric("Por debajo de la banda", f"{(res['situación'] == 'Por debajo del mercado').mean():.0%}")
        c3.metric("Diferencia mediana con el mercado", f"{res['diferencia (%)'].median():+.1f}%")
        costo = ((res["banda_min"] - res["salario_horario"]).clip(lower=0) * 40 * 4.3).sum()
        st.info(f"Costo mensual estimado de llevar a todas al mínimo de la banda (jornada completa): {pesos(costo)}")
        st.plotly_chart(px.histogram(res, x="diferencia (%)", color="situación", nbins=30,
                                     title="Diferencia de cada salario con la referencia de mercado"), width="stretch")
        st.dataframe(res.round(1), width="stretch")
        st.download_button("Descargar resultados", res.to_csv(index=False).encode(), "auditoria_resultados.csv")

with tab3:
    st.subheader("¿Qué tan confiable es el modelo?")
    st.markdown("Desempeño en el 1T2026, un trimestre que el modelo no vio al entrenarse, y cobertura real de las bandas.")
    st.dataframe(pd.DataFrame(pk["evaluacion"]).round(3), width="stretch")
    st.dataframe(pd.DataFrame(pk["cobertura"]).round(3), width="stretch")
    for archivo, titulo in [("origen_movil.csv", "R² al predecir cada trimestre con los anteriores"),
                            ("importancia_shap.csv", "Determinantes con mayor peso (SHAP)")]:
        ruta = CARPETA / archivo
        if not ruta.exists():
            continue
        datos = pd.read_csv(ruta)
        if archivo == "origen_movil.csv":
            datos["trimestre predicho"] = datos["trimestre predicho"].astype(str)
            st.plotly_chart(px.line(datos, x="trimestre predicho", y="R2", color="modelo", markers=True, title=titulo),
                            width="stretch")
        else:
            datos = datos.head(15).iloc[::-1]
            st.plotly_chart(px.bar(datos, x="% de la importancia", y="descripción", color="bloque", orientation="h",
                                   title=titulo, height=520), width="stretch")

with tab4:
    st.subheader("¿Por qué los hijos pesan poco en el salario por hora?")
    st.markdown("El costo de los cuidados **no aparece en el precio de la hora** sino en la decisión de trabajar y en la "
                "cantidad de horas. Esto importa para interpretar la referencia salarial: dos mujeres con el mismo perfil "
                "pueden tener el mismo salario por hora y salarios mensuales muy distintos.")
    ruta_part, ruta_can = CARPETA / "participacion_hijos.csv", CARPETA / "canales_hogar.csv"
    if ruta_part.exists():
        part = pd.read_csv(ruta_part)
        c1, c2 = st.columns(2)
        c1.plotly_chart(px.bar(part, x="hijos menores de 6", y="% ocupadas", text_auto=".1f", range_y=[0, 100],
                               title="Mujeres de 18 a 60 años ocupadas"), width="stretch")
        c2.plotly_chart(px.bar(part, x="hijos menores de 6", y="horas semanales (ocupadas)", text_auto=".1f",
                               title="Horas semanales de las ocupadas"), width="stretch")
    if ruta_can.exists():
        can = pd.read_csv(ruta_can)
        elegidos = {"Salario por hora (2) + educación, experiencia y aglomerado": "Salario por hora",
                    "Horas semanales (con 2)": "Horas semanales", "Salario mensual (con 2)": "Salario mensual"}
        can = can[can["modelo"].isin(elegidos)].assign(resultado=lambda d: d["modelo"].map(elegidos))
        st.plotly_chart(px.bar(can, x="variable", y="efecto (%)", color="resultado", barmode="group", text_auto=".1f",
                               title="Efecto de cada característica del hogar (asalariadas, a igual educación y aglomerado)"),
                        width="stretch")
        st.caption("Hijos: por cada hijo; hacinamiento: por cada persona más por cuarto; clima educativo y pareja: por cada "
                   "año de educación. Fuente: EPH-INDEC, 1T2024-1T2026.")
    ruta_pot = CARPETA / "salario_potencial.csv"
    if ruta_pot.exists():
        pot = pd.read_csv(ruta_pot).melt(id_vars="grupo", var_name="hijos menores de 6", value_name="diferencia (%)")
        st.plotly_chart(px.bar(pot, x="grupo", y="diferencia (%)", color="hijos menores de 6", barmode="group",
                               text_auto=".1f", title="Salario potencial por hora respecto de las asalariadas "
                                                      "(modelo de Heckman)"), width="stretch")
        st.caption("Salario que el mercado ofrecería a cada grupo según su educación, experiencia y aglomerado, "
                   "corregido por el sesgo de selección.")

with tab5:
    st.subheader("¿Cuánto paga cada aglomerado a igual perfil?")
    st.markdown("La **prima ajustada** es la diferencia salarial del aglomerado respecto del promedio del país que queda "
                "a igual educación, experiencia, ocupación, actividad, registro y hogar. La **prima bruta** incluye además "
                "las diferencias en la composición del empleo.")
    ruta_terr = CARPETA / "prima_aglomerado.csv"
    if ruta_terr.exists():
        terr = pd.read_csv(ruta_terr)
        fig = px.scatter_geo(terr, lat="lat", lon="lon", color="prima ajustada", size="asalariadas encuestadas",
                             hover_name="aglomerado", hover_data={"prima bruta": True, "lat": False, "lon": False},
                             color_continuous_scale="RdBu", color_continuous_midpoint=0, size_max=28,
                             projection="mercator", height=700)
        fig.update_geos(fitbounds="locations", showcountries=True, showsubunits=True)
        st.plotly_chart(fig, width="stretch")
        st.dataframe(terr[["aglomerado", "región", "prima bruta", "prima ajustada", "% empleo registrado",
                           "% empleo estatal", "años de educación"]].sort_values("prima ajustada", ascending=False),
                     width="stretch", hide_index=True)
