"""
Referencia salarial para mujeres asalariadas en Argentina
TFM - Máster en Data Science, Big Data & Business Analytics (UCM) - Patricia Olguín

Uso:  streamlit run app/app.py      (desde la carpeta raíz del repositorio)
El modelo y las tablas se generan con el notebook del TFM y se guardan en la carpeta modelos/.
"""
import io
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

CARPETA = Path(__file__).resolve().parent.parent / "modelos"
# Paleta común del notebook, la aplicación y el resumen
px.defaults.color_discrete_sequence = ["#2a9d8f", "#e9c46a", "#e76f51", "#264653", "#f4a261", "#8d99ae"]
px.defaults.color_continuous_scale = ["#e76f51", "#f7f3e8", "#2a9d8f"]
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


# Nombres de las categorías según el diseño de registro de la EPH, el CAES-Mercosur 1.0 y el CNO-2001 (INDEC)
CAES = {"01": "Agricultura, ganadería y caza", "10": "Elaboración de alimentos", "13": "Fabricación de textiles",
        "14": "Confección de prendas de vestir", "40": "Construcción", "45": "Comercio y reparación de vehículos",
        "48": "Comercio (excepto vehículos)", "49": "Transporte terrestre", "52": "Almacenamiento y servicios al transporte",
        "53": "Correo y mensajería", "55": "Alojamiento (hoteles y otros)", "56": "Servicios de comidas y bebidas",
        "61": "Telecomunicaciones", "62": "Programación y consultoría informática", "64": "Servicios financieros",
        "65": "Seguros y fondos de pensiones", "68": "Actividades inmobiliarias", "69": "Actividades jurídicas y contables",
        "72": "Investigación y desarrollo", "74": "Otras actividades profesionales y técnicas",
        "79": "Agencias de viajes y turismo", "80": "Investigación y seguridad", "81": "Limpieza y mantenimiento de edificios",
        "82": "Servicios administrativos a empresas", "84": "Administración pública y defensa", "85": "Enseñanza",
        "86": "Atención de la salud", "87": "Residencias con atención de la salud", "88": "Servicios sociales sin alojamiento",
        "92": "Juegos de azar y apuestas", "93": "Deporte y entretenimiento", "94": "Asociaciones",
        "96": "Otros servicios personales", "97": "Hogares que emplean personal doméstico",
        "99": "Organismos extraterritoriales o actividad no especificada"}
CNO = {"03": "Directivas de organismos y empresas estatales", "06": "Directivas de medianas empresas privadas",
       "10": "Gestión administrativa, planificación y control", "11": "Gestión jurídico-legal",
       "20": "Gestión presupuestaria, contable y financiera", "30": "Comercialización directa",
       "31": "Corretaje, venta domiciliaria y promoción", "32": "Comercialización indirecta (repositoras, cadetas)",
       "34": "Transporte", "35": "Telecomunicaciones", "36": "Almacenaje", "40": "Salud y sanidad", "41": "Educación",
       "42": "Investigación científica y tecnológica", "44": "Prevención de siniestros y medio ambiente",
       "45": "Comunicación de masas", "46": "Servicios sociales, comunales, políticos y religiosos",
       "47": "Vigilancia y seguridad civil", "48": "Servicios policiales", "51": "Deporte", "53": "Servicios gastronómicos",
       "55": "Servicio doméstico", "56": "Limpieza (no doméstica)", "57": "Cuidado y atención de personas",
       "58": "Otros servicios sociales", "72": "Construcción e infraestructura", "80": "Producción industrial y artesanal",
       "81": "Producción de software"}
ETIQUETAS = {
    "caes_division": CAES, "cno_caracter": CNO,
    "INTENSI": {"código 1": "Subocupada (trabaja menos horas de las que quiere)", "código 2": "Ocupada plena",
                "código 3": "Sobreocupada (más de 45 horas)"},
    "PP07K": {"código 1": "Recibo con sello, membrete o firma del empleador", "código 2": "Papel o recibo sin membrete",
              "código 3": "Entrega una factura", "código 4": "No le dan ni entrega nada"},
    "PP04A1": {"código 0": "No corresponde (empresa privada)", "código 1": "Estado nacional", "código 2": "Estado provincial",
               "código 3": "Estado municipal"},
    "IV4": {"código 1": "Membrana o cubierta asfáltica", "código 2": "Baldosa o losa sin cubierta", "código 3": "Pizarra o teja",
            "código 4": "Chapa de metal sin cubierta", "código 5": "Chapa de fibrocemento o plástico",
            "código 7": "Caña, tabla o paja", "código 9": "Departamento en propiedad horizontal"},
    "AGLOMERADO": pk.get("nombres_aglomerado", {}),
}
GENERICAS = {"Otros": "Otra categoría", "Sin dato": "Sin dato"}


def etiqueta(v, codigo):
    """Nombre legible de una categoría (el código se conserva internamente para el modelo)."""
    return ETIQUETAS.get(v, {}).get(str(codigo), GENERICAS.get(str(codigo), str(codigo)))


def valor_legible(v, valor):
    """Valor típico en el formato de la plantilla: nombres para las categorías y Sí/No para las binarias."""
    if pk["tipos"][v] == "categórica":
        return etiqueta(v, valor)
    if pk["tipos"][v] == "binaria":
        return "Sí" if round(valor) == 1 else "No"
    return round(float(valor), 1)


def a_codigo(v, valor):
    """Acepta el nombre legible o el código original y devuelve el código que usa el modelo."""
    inverso = {n.lower(): c for c, n in ETIQUETAS.get(v, {}).items()}
    inverso.update({n.lower(): c for c, n in GENERICAS.items()})
    texto = str(valor).strip()
    if texto.endswith(".0"):
        texto = texto[:-2]
    if v in ("caes_division", "cno_caracter") and texto.isdigit():
        texto = texto.zfill(2)
    return inverso.get(texto.lower(), texto)


def preparar(datos):
    """Ordena las columnas, aplica las categorías del entrenamiento y completa lo que falte con valores típicos."""
    X = pd.DataFrame(datos).copy()
    X = X.rename(columns={d: v for v, d in pk["descripciones"].items()})     # acepta los encabezados legibles
    if "exp_pot2" in VARIABLES and "exp_pot2" not in X and "exp_pot" in X:
        X["exp_pot2"] = pd.to_numeric(X["exp_pot"], errors="coerce") ** 2
    for v in VARIABLES:
        if v not in X:
            X[v] = pk["valores_tipicos"][v]
        if pk["tipos"][v] == "categórica":
            X[v] = pd.Categorical(X[v].map(lambda x, v=v: a_codigo(v, x)), categories=pk["categorias"][v])
        else:
            X[v] = pd.to_numeric(X[v].replace({"Sí": 1, "Si": 1, "sí": 1, "si": 1, "No": 0, "no": 0}), errors="coerce")
    return X[VARIABLES]


def predecir(X):
    """Salario horario esperado y banda del 90%, en pesos del período de referencia."""
    centro = pk["modelo"].predict(X)
    inf = pk["cuantil_inf"].predict(X) - pk["ajuste_banda"]
    sup = pk["cuantil_sup"].predict(X) + pk["ajuste_banda"]
    centro = np.clip(centro, inf, sup)
    return MEDIANA * np.exp(centro), MEDIANA * np.exp(inf), MEDIANA * np.exp(sup)


def coma(valor, decimales=1):
    """Número con coma decimal, como se escribe en español."""
    return f"{valor:.{decimales}f}".replace(".", ",")


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
                texto, tipo, tipico = pk["descripciones"][v], pk["tipos"][v], pk["valores_tipicos"][v]
                if tipo == "categórica":
                    opciones = pk["categorias"][v]
                    valores[v] = st.selectbox(texto, opciones, index=opciones.index(tipico) if tipico in opciones else 0,
                                              format_func=lambda o, v=v: etiqueta(v, o))
                elif tipo == "binaria":
                    valores[v] = float(st.checkbox(texto, value=bool(round(tipico))))
                else:
                    lo, hi = pk["rangos"][v]
                    valores[v] = st.slider(texto, float(lo), float(hi), float(np.clip(tipico, lo, hi)))
    if "exp_pot2" in VARIABLES:
        valores["exp_pot2"] = valores.get("exp_pot", pk["valores_tipicos"].get("exp_pot", 0)) ** 2
    X = preparar([valores])
    centro, inf, sup = predecir(X)
    mensual = centro[0] * pk["horas_mes"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Salario horario esperado", pesos(centro[0]))
    c2.metric("Banda del 90% (en pesos)", f"{pesos(inf[0])[2:]} a {pesos(sup[0])[2:]}")
    c3.metric("Salario mensual a jornada completa", pesos(mensual))
    region = pk["aglomerado_region"].get(str(valores.get("AGLOMERADO")))
    cbt = pk["cbt_region"].get(region)
    if cbt:
        wr = mensual / (cbt * pk["adultos_equivalentes"])
        c4.metric("Welfare ratio", coma(wr, 2))
        st.info(f"**¿Alcanza para vivir?** Con este salario a jornada completa, una trabajadora de la región **{region}** "
                f"cubriría **{coma(wr, 2)} veces** la canasta básica total de una familia tipo (pareja con dos hijos). "
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

def a_csv(df):
    """CSV para Excel en español: separador ';', coma decimal y UTF-8 con BOM (conserva tildes y eñes)."""
    return df.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig")


def leer_csv(archivo):
    """Lee el CSV subido con cualquier separador (',' o ';') y codificación (UTF-8 o la de Excel en Windows)."""
    datos = archivo.getvalue()
    for codificacion in ("utf-8-sig", "cp1252"):
        try:
            texto = datos.decode(codificacion)
            break
        except UnicodeDecodeError:
            continue
    primera = texto.splitlines()[0] if texto else ""
    sep = ";" if primera.count(";") > primera.count(",") else ","
    return pd.read_csv(io.StringIO(texto), sep=sep, decimal="," if sep == ";" else ".")


# ------------------------------------------------------------------------------------------------------------------
with tabs[1]:
    st.subheader("Auditoría de nómina: ¿pagamos según el mercado? ¿Hay brecha de género a igual perfil?")
    st.markdown("Suba un CSV **anonimizado** con una fila por persona: **id_persona**, **sexo** (Mujer o Varón), "
                "**salario_horario** (en pesos) y las columnas del perfil, con los mismos nombres que la plantilla. "
                "Las columnas que falten se completan con valores típicos. La referencia de mercado se calcula con el "
                "modelo de las mujeres asalariadas.")
    plantilla = pd.DataFrame([{"id_persona": "E001", "sexo": "Mujer", "salario_horario": round(MEDIANA, 2),
                               **{pk["descripciones"][v]: valor_legible(v, pk["valores_tipicos"][v])
                                  for v in VARIABLES if v != "exp_pot2" and not v.endswith("_faltante")}}])
    st.download_button("Descargar plantilla", a_csv(plantilla), "plantilla_nomina.csv", mime="text/csv")
    archivo = st.file_uploader("Nómina (CSV)", type="csv")
    ruta_demo = CARPETA / "nomina_demo_AR.csv"
    usar_demo = ruta_demo.exists() and st.checkbox(
        "Usar la nómina de demostración (simulada con registros de la EPH del 1T2026 del sector salud; no corresponde a "
        "ninguna empresa)")
    nomina = leer_csv(archivo) if archivo is not None else (pd.read_csv(ruta_demo) if usar_demo else None)
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
            g1.metric("Brecha de género observada (salario por hora)", f"{coma(observada)}%")
            g2.metric("Brecha de género a igual perfil", f"{coma(igual_perfil)}%")
            if igual_perfil >= 5:
                st.warning("La brecha a igual perfil supera el **5%**: con la Directiva (UE) 2023/970 de transparencia "
                           "retributiva, una empresa europea debería justificarla o hacer una evaluación conjunta con los "
                           "representantes de los trabajadores.")
            else:
                st.success("La brecha a igual perfil está por debajo del umbral del 5% de la Directiva (UE) 2023/970.")
        st.plotly_chart(px.histogram(res, x="diferencia (%)", color="situación", nbins=30,
                                     color_discrete_map={"Dentro de la banda": "#2a9d8f", "Por debajo del mercado": "#e76f51",
                                                         "Por encima del mercado": "#e9c46a"},
                                     title="Diferencia de cada salario con la referencia de mercado"), width="stretch")
        st.dataframe(res.round(1), width="stretch", hide_index=True)
        st.download_button("Descargar resultados", a_csv(res), "auditoria_resultados.csv", mime="text/csv")
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
                                         "lon": False}, color_continuous_scale=["#e76f51", "#f7f3e8", "#2a9d8f"], color_continuous_midpoint=0,
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
