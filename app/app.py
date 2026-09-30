"""
Referencia salarial para mujeres asalariadas en Argentina
TFM - Máster en Data Science, Big Data & Business Analytics (UCM) - Patricia Olguín

Uso:  streamlit run app/app.py      (desde la carpeta raíz del repositorio)
El modelo y las tablas se generan con el notebook del TFM y se guardan en la carpeta modelos/.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

CARPETA = Path(__file__).resolve().parent.parent / "modelos"
st.set_page_config(page_title="Referencia salarial para mujeres asalariadas", layout="wide")


@st.cache_resource
def cargar_modelo():
    return joblib.load(CARPETA / "modelo_AR.joblib")


def tabla(nombre, **kwargs):
    """Lee una tabla de resultados del notebook (None si todavía no existe)."""
    ruta = CARPETA / nombre
    return pd.read_csv(ruta, **kwargs) if ruta.exists() else None


if not (CARPETA / "modelo_AR.joblib").exists():
    st.error("No se encontró modelos/modelo_AR.joblib. Ejecute primero el notebook del TFM.")
    st.stop()
pk = cargar_modelo()
VARIABLES = pk["variables"]
MEDIANA = pk["mediana_salario_hora"]


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
    """Salario horario esperado y banda del 90%, en pesos del período de referencia."""
    centro = pk["modelo"].predict(X)
    inf = pk["cuantil_inf"].predict(X) - pk["ajuste_banda"]
    sup = pk["cuantil_sup"].predict(X) + pk["ajuste_banda"]
    centro = np.clip(centro, inf, sup)
    return MEDIANA * np.exp(centro), MEDIANA * np.exp(inf), MEDIANA * np.exp(sup)


def pesos(valor):
    return f"$ {valor:,.0f}".replace(",", ".")


# ------------------------------------------------------------------------------------------------------------------
st.sidebar.title("Referencia salarial para mujeres asalariadas")
st.sidebar.markdown(
    f"**Período de referencia:** {pk['periodo_referencia']}  \n"
    f"**Salario horario mediano de los asalariados:** {pesos(MEDIANA)}")
st.sidebar.markdown(
    "**Cómo leer los resultados.** El salario esperado es lo que el mercado paga en promedio a mujeres con ese perfil. "
    "La **banda** contiene el salario real de 9 de cada 10 trabajadoras con ese perfil. Probado con datos que el modelo "
    "no había visto, anticipa alrededor de la mitad de las diferencias salariales entre trabajadoras; el resto depende "
    "de factores que la encuesta no mide (empresa concreta, desempeño).")
st.sidebar.caption("Fuente: microdatos de la Encuesta Permanente de Hogares (INDEC), 1T2024-1T2026, y Canasta Básica "
                   "Total regional (INDEC). Es una referencia de mercado, no un veredicto sobre casos individuales.")

tabs = st.tabs(["¿Cuánto debería cobrar?", "Auditoría de nómina", "Territorio y recursos naturales",
                "Hogar y cuidados", "Argentina, Brasil y España"])

# ------------------------------------------------------------------------------------------------------------------
with tabs[0]:
    st.subheader("¿Cuánto paga el mercado por este perfil? ¿Alcanza para vivir?")
    valores = {}
    columnas = st.columns(2)
    for i, bloque in enumerate(sorted(set(pk["bloques"].values()))):
        with columnas[i % 2].expander(bloque, expanded=i < 2):
            for v in [x for x in VARIABLES if pk["bloques"][x] == bloque and not x.endswith("_faltante") and x != "exp_pot2"]:
                etiqueta, tipo, tipico = pk["descripciones"][v], pk["tipos"][v], pk["valores_tipicos"][v]
                if tipo == "categórica":
                    opciones = pk["categorias"][v]
                    nombres = pk.get("nombres_aglomerado", {}) if v == "AGLOMERADO" else {}
                    valores[v] = st.selectbox(etiqueta, opciones, index=opciones.index(tipico) if tipico in opciones else 0,
                                              format_func=lambda o, n=nombres: n.get(o, o))
                elif tipo == "binaria":
                    valores[v] = float(st.checkbox(etiqueta, value=bool(round(tipico))))
                else:
                    lo, hi = pk["rangos"][v]
                    valores[v] = st.slider(etiqueta, float(lo), float(hi), float(np.clip(tipico, lo, hi)))
    if "exp_pot2" in VARIABLES:
        valores["exp_pot2"] = valores.get("exp_pot", pk["valores_tipicos"].get("exp_pot", 0)) ** 2
    X = preparar([valores])
    centro, inf, sup = predecir(X)
    mensual = centro[0] * pk["horas_mes"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Salario horario esperado", pesos(centro[0]))
    c2.metric("Banda (mínimo – máximo)", f"{pesos(inf[0])} – {pesos(sup[0])}")
    c3.metric("Salario mensual a jornada completa", pesos(mensual))
    region = pk["aglomerado_region"].get(str(valores.get("AGLOMERADO")))
    cbt = pk["cbt_region"].get(region)
    if cbt:
        wr = mensual / (cbt * pk["adultos_equivalentes"])
        c4.metric("Welfare ratio", f"{wr:.2f}")
        st.info(f"**¿Alcanza para vivir?** Con este salario a jornada completa, una trabajadora de la región **{region}** "
                f"cubriría **{wr:.2f} veces** la canasta básica total de una familia tipo (pareja con dos hijos). "
                + ("No alcanza para que esa familia supere la línea de pobreza." if wr < 1 else
                   "Alcanza para que esa familia supere la línea de pobreza."))
    contrib = pk["modelo"].predict(X, pred_contrib=True)[0][:-1]
    explicacion = pd.DataFrame({"característica": [pk["descripciones"][v] for v in VARIABLES],
                                "efecto (%)": 100 * (np.exp(contrib) - 1)})
    explicacion = explicacion.reindex(explicacion["efecto (%)"].abs().sort_values().index).tail(12)
    fig = px.bar(explicacion, x="efecto (%)", y="característica", orientation="h",
                 color=explicacion["efecto (%)"] > 0, color_discrete_map={True: "#2a9d8f", False: "#e76f51"},
                 title="¿Qué explica este resultado? Cuánto sube (verde) o baja (rojo) el salario cada característica")
    fig.update_layout(showlegend=False, height=480, yaxis_title="")
    st.plotly_chart(fig, width="stretch")

# ------------------------------------------------------------------------------------------------------------------
with tabs[1]:
    st.subheader("Auditoría de nómina: ¿pagamos según el mercado? ¿Hay brecha de género a igual perfil?")
    st.markdown("Suba un CSV **anonimizado** con una fila por persona: `id_persona`, `sexo` (Mujer / Varón), "
                "`salario_horario` (en pesos) y las columnas del perfil. Las columnas que falten se completan con valores "
                "típicos. La referencia de mercado se calcula con el modelo de las mujeres asalariadas.")
    plantilla = pd.DataFrame([{"id_persona": "E001", "sexo": "Mujer", "salario_horario": round(MEDIANA, 2),
                               **{v: pk["valores_tipicos"][v] for v in VARIABLES}}])
    st.download_button("Descargar plantilla", plantilla.to_csv(index=False).encode(), "plantilla_nomina.csv")
    archivo = st.file_uploader("Nómina (CSV)", type="csv")
    ruta_demo = CARPETA / "nomina_demo_AR.csv"
    usar_demo = ruta_demo.exists() and st.checkbox(
        "Usar la nómina de demostración (simulada con registros de la EPH del 1T2026 del sector salud; no corresponde a "
        "ninguna empresa)")
    nomina = pd.read_csv(archivo) if archivo is not None else (pd.read_csv(ruta_demo) if usar_demo else None)
    if nomina is not None:
        if "id_persona" not in nomina and "id_empleada" in nomina:
            nomina = nomina.rename(columns={"id_empleada": "id_persona"})
        centro, inf, sup = predecir(preparar(nomina))
        res = nomina[["id_persona", "salario_horario"] + (["sexo"] if "sexo" in nomina else [])].assign(
            esperado=centro, banda_min=inf, banda_max=sup)
        res["diferencia (%)"] = 100 * (res["salario_horario"] / res["esperado"] - 1)
        res["situación"] = np.select([res.salario_horario < res.banda_min, res.salario_horario > res.banda_max],
                                     ["Por debajo del mercado", "Por encima del mercado"], "Dentro de la banda")
        c1, c2, c3 = st.columns(3)
        c1.metric("Personas analizadas", len(res))
        c2.metric("Por debajo de la banda de mercado", f"{(res['situación'] == 'Por debajo del mercado').mean():.0%}")
        costo = ((res["banda_min"] - res["salario_horario"]).clip(lower=0) * pk["horas_mes"]).sum()
        c3.metric("Costo mensual de llevarlas al mínimo de la banda", pesos(costo))

        if "sexo" in res and set(res["sexo"]) >= {"Mujer", "Varón"}:
            ratio = np.log(res["salario_horario"] / res["esperado"])
            observada = 100 * (res.loc[res.sexo == "Varón", "salario_horario"].mean()
                               / res.loc[res.sexo == "Mujer", "salario_horario"].mean() - 1)
            igual_perfil = 100 * (np.exp(ratio[res.sexo == "Varón"].mean() - ratio[res.sexo == "Mujer"].mean()) - 1)
            g1, g2 = st.columns(2)
            g1.metric("Brecha de género observada (salario por hora)", f"{observada:.1f}%")
            g2.metric("Brecha de género a igual perfil", f"{igual_perfil:.1f}%")
            if igual_perfil >= 5:
                st.warning("La brecha a igual perfil supera el **5%**: con la Directiva (UE) 2023/970 de transparencia "
                           "retributiva, una empresa europea debería justificarla o hacer una evaluación conjunta con los "
                           "representantes de los trabajadores.")
            else:
                st.success("La brecha a igual perfil está por debajo del umbral del 5% de la Directiva (UE) 2023/970.")
        st.plotly_chart(px.histogram(res, x="diferencia (%)", color="situación", nbins=30,
                                     title="Diferencia de cada salario con la referencia de mercado"), width="stretch")
        st.dataframe(res.round(1), width="stretch", hide_index=True)
        st.download_button("Descargar resultados", res.to_csv(index=False).encode(), "auditoria_resultados.csv")
    brechas = tabla("brecha_grupos.csv")
    if brechas is not None:
        with st.expander("Referencia nacional: brecha de género no explicada por grupo de trabajadores (EPH)"):
            fig = px.bar(brechas, x="no explicada (%)", y="grupo", orientation="h", height=560,
                         color=brechas["no explicada (%)"] >= 5, color_discrete_map={True: "#e76f51", False: "#2a9d8f"})
            fig.add_vline(x=5, line_dash="dash")
            fig.update_layout(showlegend=False, yaxis_title="")
            st.plotly_chart(fig, width="stretch")

# ------------------------------------------------------------------------------------------------------------------
with tabs[2]:
    st.subheader("¿Cuánto paga cada ciudad a igual perfil? ¿Y descontando el costo de vida?")
    terr = tabla("prima_aglomerado.csv")
    if terr is not None:
        medida = st.radio("Medida", ["prima real", "prima ajustada", "prima bruta"], horizontal=True,
                          format_func={"prima real": "A igual perfil y costo de vida (real)",
                                       "prima ajustada": "A igual perfil (en pesos)",
                                       "prima bruta": "Sin ajustar"}.get)
        fig = px.scatter_geo(terr, lat="lat", lon="lon", color=medida, size="asalariadas encuestadas", hover_name="aglomerado",
                             hover_data={"prima bruta": True, "prima ajustada": True, "prima real": True, "lat": False,
                                         "lon": False}, color_continuous_scale="RdBu", color_continuous_midpoint=0,
                             size_max=26, projection="mercator", height=650)
        fig.update_geos(fitbounds="locations", showcountries=True, showsubunits=True)
        st.plotly_chart(fig, width="stretch")
        st.caption("Prima: diferencia salarial respecto del promedio del país (%). Real: descontando la Canasta Básica "
                   "Total de la región (INDEC).")
    polos = tabla("polos_recursos.csv", index_col=0)
    if polos is not None:
        st.markdown("**Polos energéticos y mineros.** Casi no emplean mujeres de forma directa: si la riqueza de los "
                    "recursos naturales mejora su salario, lo hace a través del mercado de trabajo de la ciudad.")
        st.dataframe(polos, width="stretch")
    if terr is not None:
        st.dataframe(terr[["aglomerado", "región", "polo", "prima bruta", "prima ajustada", "prima real",
                           "% empleo registrado"]].sort_values("prima real", ascending=False), width="stretch", hide_index=True)

# ------------------------------------------------------------------------------------------------------------------
with tabs[3]:
    st.subheader("Los cuidados se pagan en empleo y en horas, no en el precio de la hora")
    st.markdown("Dos mujeres con el mismo perfil pueden tener el mismo salario por hora y salarios mensuales muy distintos: "
                "los hijos y las tareas del hogar reducen la probabilidad de trabajar y las horas trabajadas.")
    part, can, pot = tabla("participacion_hijos.csv"), tabla("canales_hogar.csv"), tabla("salario_potencial.csv")
    if part is not None:
        c1, c2 = st.columns(2)
        c1.plotly_chart(px.bar(part, x="hijos menores de 6", y="% ocupadas", text_auto=".1f", range_y=[0, 100],
                               title="Mujeres de 18 a 60 años ocupadas"), width="stretch")
        c2.plotly_chart(px.bar(part, x="hijos menores de 6", y="horas semanales (ocupadas)", text_auto=".1f",
                               title="Horas semanales de las ocupadas"), width="stretch")
    if can is not None:
        st.plotly_chart(px.bar(can, x="variable", y="efecto (%)", color="resultado", barmode="group", text_auto=".1f",
                               title="Efecto de cada característica del hogar (asalariadas, a igual educación y ciudad)"),
                        width="stretch")
    if pot is not None:
        pot = pot.melt(id_vars="grupo", var_name="hijos menores de 6", value_name="diferencia (%)")
        st.plotly_chart(px.bar(pot, x="grupo", y="diferencia (%)", color="hijos menores de 6", barmode="group",
                               text_auto=".1f", title="Salario por hora que ofrecería el mercado a cada grupo de mujeres, "
                                                      "respecto de las asalariadas"), width="stretch")

# ------------------------------------------------------------------------------------------------------------------
with tabs[4]:
    st.subheader("¿Qué premia cada mercado y cuánto vale el salario en cada país?")
    paises, efectos, bloques = tabla("comparacion_paises.csv", index_col=0), tabla("comparacion_efectos.csv", index_col=0), \
        tabla("comparacion_bloques.csv", index_col=0)
    if paises is None:
        st.info("La comparación internacional todavía no está disponible.")
    else:
        c1, c2 = st.columns(2)
        c1.plotly_chart(px.bar(paises.reset_index(), x=paises.index.name or "index", y="salario horario mediano (USD PPA 2024)",
                               text_auto=".1f", title="Salario horario mediano de las asalariadas (dólares PPA 2024)"),
                        width="stretch")
        c2.plotly_chart(px.bar(paises.reset_index(), x=paises.index.name or "index", y="no explicada (%)", text_auto=".1f",
                               title="Brecha de género no explicada (%)"), width="stretch")
        if efectos is not None:
            st.markdown("**Cuánto suma cada factor al salario por hora de las mujeres (%)**")
            st.dataframe(efectos.round(1), width="stretch")
        if bloques is not None:
            st.plotly_chart(px.bar(bloques.reset_index().melt(id_vars=bloques.index.name or "index", var_name="bloque",
                                                              value_name="% de la importancia"),
                                   x=bloques.index.name or "index", y="% de la importancia", color="bloque",
                                   title="Peso de cada bloque de determinantes"), width="stretch")
