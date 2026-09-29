# -*- coding: utf-8 -*-
"""
Funciones del TFM "Determinantes del salario de las mujeres asalariadas en Argentina"

@author: Patricia Olguín

Siguiendo la organización del módulo de Minería de datos (archivo NuestrasFunciones.py), las funciones auxiliares se
reúnen en este archivo y el notebook las importa con:  from funciones_tfm import *
Varias funciones son adaptaciones de las vistas en el máster (se indica en cada caso).
"""

import io
import os
import re
import zipfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import requests
import scipy.stats as stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
import patsy

SEMILLA = 42


# ==================================================================================================================
# 1. LECTURA DE LA EPH
# ==================================================================================================================

## Función para descargar una onda de la EPH desde el INDEC y guardarla en Drive
def descargar_onda_indec(ano, trimestre, ruta_archivo):
    url = f"https://www.indec.gob.ar/ftp/cuadros/menusuperior/eph/EPH_usu_{trimestre}_Trim_{ano}_txt.zip"
    r = requests.get(url, timeout=300)
    # Si la onda no está publicada, el INDEC devuelve una página HTML en lugar del zip
    if r.status_code != 200 or "html" in r.headers.get("content-type", ""):
        raise FileNotFoundError(f"El INDEC todavía no publicó la onda {trimestre}T{ano}")
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        for nombre in z.namelist():
            base = os.path.basename(nombre).lower()
            if base.startswith("usu_individual"):
                open(ruta_archivo(ano, trimestre, "pers"), "wb").write(z.read(nombre))
            elif base.startswith("usu_hogar"):
                open(ruta_archivo(ano, trimestre, "hog"), "wb").write(z.read(nombre))
    print(f"Descargada la onda {trimestre}T{ano}")


## Función para leer un archivo de la EPH (separador ';' y coma decimal)
def leer_eph(ruta):
    return pd.read_csv(ruta, sep=";", decimal=",", low_memory=False)


# ==================================================================================================================
# 2. CONSTRUCCIÓN DE VARIABLES CON LA BASE COMPLETA DE PERSONAS
# ==================================================================================================================

## Función para calcular los años de educación a partir del nivel (CH12), si lo finalizó (CH13) y el último año (CH14)
def anios_educacion(d):
    inicio = {1: 0, 2: 0, 3: 0, 4: 7, 5: 9, 6: 12, 7: 12, 8: 17}      # años previos al nivel
    duracion = {1: 0, 2: 7, 3: 9, 4: 5, 5: 3, 6: 3, 7: 5, 8: 2}       # duración teórica del nivel
    ini, dur = d["CH12"].map(inicio), d["CH12"].map(duracion)
    aprobados = d["CH14"].where(d["CH14"].between(0, 9)).fillna(0)
    anios = pd.Series(np.where(d["CH13"] == 1, ini + dur, ini + np.minimum(aprobados, dur)), index=d.index, dtype=float)
    anios[d["NIVEL_ED"] == 7] = 0          # sin instrucción
    return anios


## Función para construir las variables del hogar con TODOS sus miembros (antes de filtrar a las asalariadas)
def variables_del_hogar(p, clave_hogar):
    edad = p["CH06"]
    hogar = [p[c] for c in clave_hogar]
    educ = anios_educacion(p)

    # Composición del hogar
    for nombre, condicion in {"menores_6": edad < 6, "menores_18": edad < 18, "mayores_65": edad >= 65,
                              "ocupados_hogar": p["ESTADO"] == 1,
                              "mujeres_adultas_hogar": (p["CH04"] == 2) & (edad >= 18)}.items():
        p[nombre] = condicion.groupby(hogar).transform("sum")

    # Pareja: el cónyuge si ella es jefa (CH03 = 1); el jefe si ella es cónyuge (CH03 = 2)
    jefe = p.loc[p["CH03"] == 1, clave_hogar].assign(educ_j=educ, ocup_j=(p["ESTADO"] == 1) * 1.0, edad_j=edad)
    conyuge = p.loc[p["CH03"] == 2, clave_hogar].assign(educ_c=educ, ocup_c=(p["ESTADO"] == 1) * 1.0, edad_c=edad)
    pareja = (p[clave_hogar].merge(jefe.drop_duplicates(clave_hogar), on=clave_hogar, how="left")
                            .merge(conyuge.drop_duplicates(clave_hogar), on=clave_hogar, how="left"))
    es_jefa, es_conyuge = (p["CH03"] == 1).values, (p["CH03"] == 2).values
    p["educ_pareja"] = np.select([es_jefa, es_conyuge], [pareja["educ_c"], pareja["educ_j"]], np.nan)
    p["pareja_ocupada"] = np.select([es_jefa, es_conyuge], [pareja["ocup_c"], pareja["ocup_j"]], np.nan)
    edad_pareja = np.select([es_jefa, es_conyuge], [pareja["edad_c"], pareja["edad_j"]], np.nan)

    # Clima educativo: años de educación promedio de los OTROS adultos de 25 años o más (sin ella ni su pareja)
    educ_25 = educ.where(edad >= 25)
    pareja_25 = pd.Series(np.where(edad_pareja >= 25, p["educ_pareja"], np.nan), index=p.index)
    suma = educ_25.fillna(0).groupby(hogar).transform("sum") - educ_25.fillna(0) - pareja_25.fillna(0)
    cuenta = educ_25.notna().groupby(hogar).transform("sum") - educ_25.notna() - pareja_25.notna()
    p["clima_educativo"] = (suma / cuenta.where(cuenta > 0)).round(2)

    # Hijos propios: hijos del jefe si ella es jefa o cónyuge; nietos del jefe si ella es hija o nuera (aproximación)
    for tope in (6, 18):
        hijos = ((p["CH03"] == 3) & (edad < tope)).groupby(hogar).transform("sum")
        nietos = ((p["CH03"] == 5) & (edad < tope)).groupby(hogar).transform("sum")
        p[f"hijos_propios_menores_{tope}"] = np.select([p["CH03"].isin([1, 2]), p["CH03"].isin([3, 4])],
                                                       [hijos, nietos], 0)
    p["anios_educ_hogar"] = educ           # auxiliar: se usa para el análisis de participación
    return p


## Función para procesar una onda: variables del hogar, población de estudio y unión con la base de hogares
def procesar_onda(personas, hogares, ano, trimestre, clave_hogar):
    p = variables_del_hogar(personas.copy(), clave_hogar)

    # Todas las mujeres de 18 a 60 años, trabajen o no (para estudiar la participación laboral)
    mujeres_18_60 = p.loc[(p["CH04"] == 2) & p["CH06"].between(18, 60),
                          clave_hogar + ["COMPONENTE", "AGLOMERADO", "hijos_propios_menores_6", "hijos_propios_menores_18",
                                         "pareja_ocupada", "ESTADO", "CAT_OCUP", "PP3E_TOT", "PONDERA", "CH06",
                                         "anios_educ_hogar"]].assign(periodo=ano * 10 + trimestre)

    # Población de estudio: asalariados ocupados de 18 a 60 años con salario, ponderador y horas válidos
    filtro = ((p["ESTADO"] == 1) & (p["CAT_OCUP"] == 3) & p["CH06"].between(18, 60) & (p["P21"] > 0)
              & (p["PONDIIO"] > 0) & p["PP3E_TOT"].between(1, 98))
    p = p[filtro].drop(columns="anios_educ_hogar")

    # Unión con la base de hogares (solo las columnas que no están en la de personas)
    cols_hogar = clave_hogar + [c for c in hogares.columns if c not in p.columns]
    base_onda = p.merge(hogares[cols_hogar], on=clave_hogar, how="inner", validate="m:1")
    base_onda["ANO4"], base_onda["TRIMESTRE"] = ano, trimestre
    return base_onda, mujeres_18_60


## Aglomerados de la EPH: nombre y ubicación aproximada de su ciudad principal (latitud, longitud)
AGLOMERADOS = {
    2: ("Gran La Plata", -34.92, -57.95), 3: ("Bahía Blanca-Cerri", -38.72, -62.27), 4: ("Gran Rosario", -32.95, -60.65),
    5: ("Gran Santa Fe", -31.63, -60.70), 6: ("Gran Paraná", -31.73, -60.53), 7: ("Posadas", -27.37, -55.90),
    8: ("Gran Resistencia", -27.45, -58.99), 9: ("Comodoro Rivadavia-Rada Tilly", -45.86, -67.48),
    10: ("Gran Mendoza", -32.89, -68.83), 12: ("Corrientes", -27.47, -58.83), 13: ("Gran Córdoba", -31.42, -64.19),
    14: ("Concordia", -31.39, -58.02), 15: ("Formosa", -26.18, -58.18), 17: ("Neuquén-Plottier", -38.95, -68.06),
    18: ("Santiago del Estero-La Banda", -27.79, -64.26), 19: ("Jujuy-Palpalá", -24.19, -65.30),
    20: ("Río Gallegos", -51.62, -69.22), 22: ("Gran Catamarca", -28.47, -65.78), 23: ("Gran Salta", -24.79, -65.41),
    25: ("La Rioja", -29.41, -66.86), 26: ("Gran San Luis", -33.30, -66.34), 27: ("Gran San Juan", -31.54, -68.54),
    29: ("Gran Tucumán-Tafí Viejo", -26.82, -65.22), 30: ("Santa Rosa-Toay", -36.62, -64.29),
    31: ("Ushuaia-Río Grande", -54.80, -68.30), 32: ("Ciudad Autónoma de Buenos Aires", -34.60, -58.38),
    33: ("Partidos del Gran Buenos Aires", -34.75, -58.60), 34: ("Mar del Plata", -38.00, -57.56),
    36: ("Río Cuarto", -33.13, -64.35), 38: ("San Nicolás-Villa Constitución", -33.33, -60.22),
    91: ("Rawson-Trelew", -43.25, -65.31), 93: ("Viedma-Carmen de Patagones", -40.81, -63.00),
}
REGIONES = {1: "Gran Buenos Aires", 40: "Noroeste", 41: "Noreste", 42: "Cuyo", 43: "Pampeana", 44: "Patagonia"}


## Función para calcular la mediana ponderada
def mediana_ponderada(valores, pesos):
    valores, pesos = np.asarray(valores, float), np.asarray(pesos, float)
    orden = np.argsort(valores)
    acumulado = np.cumsum(pesos[orden])
    return valores[orden][np.searchsorted(acumulado, 0.5 * acumulado[-1])]


# ==================================================================================================================
# 3. DEPURACIÓN
# ==================================================================================================================

## Función para obtener el perfil de cada variable (tipo, nulos, ceros, valores distintos y moda)
def perfil(df):
    filas = []
    for col in df.columns:
        s = df[col]
        moda = s.mode(dropna=True)
        filas.append({"variable": col, "tipo": str(s.dtype), "% nulos": 100 * s.isna().mean(),
                      "% ceros (no corresponde)": 100 * (s == 0).mean() if pd.api.types.is_numeric_dtype(s) else 0.0,
                      "valores distintos": s.nunique(),
                      "valor más frecuente": moda.iloc[0] if len(moda) else np.nan})
    return pd.DataFrame(filas).set_index("variable").round(1)


## Funciones para recodificar la actividad (CAES-Mercosur) en divisiones de 2 dígitos y ramas
def division_caes(codigo):
    if pd.isna(codigo) or int(codigo) <= 0:
        return np.nan
    s = str(int(codigo))
    return int(s) if len(s) <= 2 else int(s.zfill(4)[:2])


def rama_desde_division(div):
    if pd.isna(div):
        return np.nan
    cortes = [(3, "Agro y pesca"), (9, "Minería"), (33, "Industria"), (39, "Electricidad, gas y agua"),
              (43, "Construcción"), (48, "Comercio"), (53, "Transporte"), (56, "Hoteles y restaurantes"),
              (63, "Información y comunicación"), (68, "Finanzas e inmobiliarias"),
              (82, "Servicios profesionales y empresariales"), (84, "Administración pública"), (85, "Enseñanza"),
              (88, "Salud y servicios sociales"), (96, "Otros servicios"), (98, "Servicio doméstico")]
    for tope, nombre in cortes:
        if div <= tope:
            return nombre
    return "Organismos extraterritoriales"


## Función de ingeniería de variables: construye variables más informativas a partir de los códigos de la EPH
def crear_variables(df):
    cno = df["PP04D_COD"].map(lambda c: str(int(c)).zfill(5) if pd.notna(c) and c > 0 else np.nan)
    # Educación y experiencia potencial de Mincer
    df["edad"] = df["CH06"]
    df["anios_educ"] = anios_educacion(df)
    df["exp_pot"] = (df["edad"] - df["anios_educ"] - 6).clip(lower=0)
    df["exp_pot2"] = df["exp_pot"] ** 2
    df["nivel_educativo"] = df["NIVEL_ED"].map({7: "Sin instrucción", 1: "Primario incompleto", 2: "Primario completo",
                                                3: "Secundario incompleto", 4: "Secundario completo",
                                                5: "Superior incompleto", 6: "Superior completo"})
    # Características personales
    df["parentesco"] = df["CH03"].map({1: "Jefa", 2: "Cónyuge", 3: "Hija"}).fillna("Otro familiar o no familiar")
    df["estado_civil"] = df["CH07"].map({1: "Unida", 2: "Casada", 3: "Separada o divorciada", 4: "Viuda", 5: "Soltera"})
    df["cobertura_salud"] = df["CH08"].map({1: "Obra social", 12: "Obra social", 13: "Obra social", 123: "Obra social",
                                            2: "Prepaga o mutual", 23: "Prepaga o mutual", 3: "Plan público", 4: "Ninguna"})
    lugares = {1: "Esta localidad", 2: "Otra localidad de la provincia", 3: "Otra provincia", 4: "País limítrofe",
               5: "Otro país"}
    df["lugar_nacimiento"] = df["CH15"].map(lugares)
    df["residencia_5_anios"] = df["CH16"].map(lugares)
    # Ocupación (CNO-2001) y actividad (CAES-Mercosur)
    df["cno_caracter"] = cno.str[:2]
    df["cno_jerarquia"] = cno.str[2].map({"0": "Dirección", "1": "Cuenta propia", "2": "Jefatura", "3": "Ejecución"})
    df["cno_tecnologia"] = cno.str[3].map({"1": "Sin maquinaria", "2": "Maquinaria y equipos", "3": "Sistemas informatizados"})
    df["calificacion"] = cno.str[4].map({"1": "Profesional", "2": "Técnica", "3": "Operativa", "4": "No calificada"})
    division = df["PP04B_COD"].map(division_caes)
    df["rama"] = division.map(rama_desde_division)
    df["caes_division"] = division.map(lambda x: f"{int(x):02d}" if pd.notna(x) else np.nan)
    df["tamano_establecimiento"] = np.select(
        [df["PP04C"].between(1, 5) | (df["PP04C99"] == 1), df["PP04C"].between(6, 8) | (df["PP04C99"] == 2),
         df["PP04C"].between(9, 12) | (df["PP04C99"] == 3)], ["Hasta 5", "6 a 40", "Más de 40"], default="Sin dato")
    df["sector"] = df["PP04A"].map({1: "Estatal", 2: "Privado", 3: "Otro"})
    df["lugar_trabajo"] = df["PP04G"].map({11: "Local u oficina", 12: "Su vivienda (lugar exclusivo)", 13: "Chacra o finca",
                                           2: "Puesto callejero", 3: "Vehículo", 4: "Vehículo de transporte", 5: "Obra",
                                           6: "Su vivienda (sin lugar exclusivo)", 7: "Vivienda del patrón",
                                           8: "Domicilio de clientes", 9: "Calle o ambulante", 10: "Otro"})
    # Antigüedad en tramos 1 a 6 (PP07A); el servicio doméstico la informa en meses y años (PP04B3)
    meses = df["PP04B3_ANO"].fillna(0) * 12 + df["PP04B3_MES"].fillna(0) + (df["PP04B3_DIA"].fillna(0) > 0) / 30
    tramo = pd.cut(meses, [0, 1, 3, 6, 12, 60, np.inf], labels=[1, 2, 3, 4, 5, 6], right=False).astype(float)
    df["antiguedad"] = df["PP07A"].where(df["PP07A"].between(1, 6)).fillna(tramo.where(meses > 0))
    df["jornada_parcial"] = (df["PP3E_TOT"] < 35).astype(float)
    # Hogar y tareas domésticas
    df["hacinamiento"] = df["IX_TOT"] / df["II2"].where(df["II2"] > 0)
    df["otros_ocupados_hogar"] = df["ocupados_hogar"] - 1
    df["otras_mujeres_adultas"] = df["mujeres_adultas_hogar"] - 1
    tareas, ayuda = df[["VII1_1", "VII1_2"]], df[["VII2_1", "VII2_2", "VII2_3", "VII2_4"]]
    df["realiza_tareas_casa"] = tareas.eq(df["COMPONENTE"], axis=0).any(axis=1).astype(float)
    df["servicio_domestico_hogar"] = pd.concat([tareas, ayuda], axis=1).eq(96).any(axis=1).astype(float)
    df["recibe_ayuda_tareas"] = ayuda.isin([0, 98]).all(axis=1).map({True: 0.0, False: 1.0})
    return df


## Función para calcular la V de Cramér (adaptada de NuestrasFunciones.py: las continuas se agrupan en 5 tramos)
def cramers_v(var1, varObj):
    if not var1.dtypes.name in ("category", "object"):
        var1 = pd.cut(var1, bins=5) if var1.nunique() > 5 else var1
    if not varObj.dtypes.name in ("category", "object"):
        varObj = pd.cut(varObj, bins=5)
    data = pd.crosstab(var1.astype(str), varObj).values
    if min(data.shape) < 2:
        return 0.0
    return stats.contingency.association(data, method="cramer")


## Función para medir la incidencia de outliers (adaptada de gestiona_outliers de NuestrasFunciones.py)
def incidencia_outliers(col):
    col = col.dropna()
    # Criterio 1 según la asimetría: desvíos a la media o a la mediana (MAD)
    if abs(col.skew()) < 1:
        criterio1 = abs((col - col.mean()) / col.std()) > 3
    else:
        criterio1 = abs((col - col.median()) / stats.median_abs_deviation(col)) > 6
    # Criterio 2: fuera de 3 rangos intercuartílicos
    q1, q3 = col.quantile([0.25, 0.75])
    criterio2 = (col < q1 - 3 * (q3 - q1)) | (col > q3 + 3 * (q3 - q1))
    return pd.Series({"% atípicos inferiores": 100 * (criterio1 & criterio2 & (col < q1)).mean(),
                      "% atípicos superiores": 100 * (criterio1 & criterio2 & (col > q3)).mean()})


## Función para histograma con boxplot (de NuestrasFunciones.py, simplificada)
def histogram_boxplot(data, xlabel=None, title=None, figsize=(9, 5), bins=60):
    f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)}, figsize=figsize)
    sns.boxplot(x=data, ax=ax_box)
    sns.histplot(x=data, ax=ax_hist, bins=bins)
    ax_hist.axvline(np.mean(data), color="g", linestyle="-")       # media
    ax_hist.axvline(np.median(data), color="y", linestyle="--")    # mediana
    if xlabel:
        ax_hist.set(xlabel=xlabel)
    if title:
        ax_box.set(title=title, xlabel="")
    plt.show()


# ==================================================================================================================
# 4. MODELADO Y EVALUACIÓN
# ==================================================================================================================

## Función para calcular las métricas ponderadas y graficar real vs. predicho (como en la tarea de Machine Learning)
def saca_metricas(y_real, y_pred, pesos, titulo="", graficar=True):
    rmse = np.sqrt(mean_squared_error(y_real, y_pred, sample_weight=pesos))
    mae = mean_absolute_error(y_real, y_pred, sample_weight=pesos)
    r2 = r2_score(y_real, y_pred, sample_weight=pesos)
    print(f"{titulo}  RMSE = {rmse:.4f} | MAE = {mae:.4f} | R² = {r2:.4f}")
    if graficar:
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        muestra = np.random.default_rng(SEMILLA).choice(len(y_real), size=min(4000, len(y_real)), replace=False)
        axs[0].scatter(np.asarray(y_real)[muestra], np.asarray(y_pred)[muestra], s=4, alpha=0.4)
        axs[0].plot([-2.5, 2], [-2.5, 2], c="red")
        axs[0].set(xlabel="Real", ylabel="Predicho", title=f"{titulo}: real vs. predicho")
        sns.histplot(np.asarray(y_real) - np.asarray(y_pred), bins=60, ax=axs[1])
        axs[1].set_title("Distribución de los errores")
        plt.tight_layout()
        plt.show()
    return {"RMSE": rmse, "MAE": mae, "R2": r2}


## Función One-Hot Encoding (limpia los nombres de columna de signos que LightGBM no admite)
def one_hot(datos, categoricas):
    X = pd.get_dummies(datos, columns=categoricas, drop_first=True, dtype=float)
    X.columns = [re.sub(r"[,:\[\]{}\"]+", "", c).replace(" ", "_") for c in X.columns]
    return X


## Función para generar la fórmula de un modelo (de NuestrasFunciones.py; las categóricas se marcan con C())
def ols_formula(variables, dependent_var, categoricas=()):
    return dependent_var + " ~ " + " + ".join(f"C({v})" if v in categoricas else v for v in variables)


## Función para calcular el VIF (de NuestrasFunciones.py)
def vif_modelo(formula, data):
    y, X = patsy.dmatrices(formula, data, return_type="dataframe")
    vif = pd.DataFrame({"feature": X.columns,
                        "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]})
    return vif[vif["feature"] != "Intercept"].sort_values(by="VIF", ascending=False)


## Función para convertir un coeficiente de la regresión en un nombre legible
def nombre_legible(termino, descripcion):
    coincide = re.match(r"C\((\w+)\)\[T\.(.+)\]", termino)
    if coincide:
        return f"{descripcion.get(coincide.group(1), coincide.group(1))}: {coincide.group(2)}"
    return descripcion.get(termino, termino)


## Función para calcular el índice de estabilidad poblacional (PSI) entre una referencia y un período nuevo
def psi(referencia, actual, cortes=10):
    referencia, actual = pd.Series(referencia).dropna(), pd.Series(actual).dropna()
    if referencia.dtype == object or isinstance(referencia.dtype, pd.CategoricalDtype) or referencia.nunique() <= 12:
        r = referencia.astype(str).value_counts(normalize=True)
        a = actual.astype(str).value_counts(normalize=True).reindex(r.index).fillna(0)
    else:
        bordes = np.unique(np.quantile(referencia, np.linspace(0, 1, cortes + 1)))
        bordes[0], bordes[-1] = -np.inf, np.inf
        r = pd.cut(referencia, bordes).value_counts(normalize=True, sort=False)
        a = pd.cut(actual, bordes).value_counts(normalize=True, sort=False)
    r, a = r.clip(lower=1e-4), a.clip(lower=1e-4)
    return float(((a - r) * np.log(a / r)).sum())


## Función para el modelo de Heckman en dos etapas (corrección del sesgo de selección)
def heckman_dos_etapas(datos, formula_seleccion, formula_salario, pesos, grupos):
    """1) Probit de la probabilidad de ser asalariada (con salario observado) sobre TODAS las mujeres.
       2) Ecuación de salarios de las asalariadas con el inverso del ratio de Mills (imr) como variable adicional.
       Devuelve el probit, la regresión sin corregir, la corregida y los datos con el índice lineal y el imr."""
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from scipy.stats import norm
    probit = smf.glm(formula_seleccion, data=datos, family=sm.families.Binomial(link=sm.families.links.Probit()),
                     var_weights=datos[pesos] / datos[pesos].mean()).fit()
    datos = datos.assign(xb=probit.predict(datos, which="linear"))
    datos["imr"] = norm.pdf(datos["xb"]) / norm.cdf(datos["xb"])
    s = datos[datos["asalariada"] == 1]
    ajuste = dict(weights=s[pesos] / s[pesos].mean())
    robustos = dict(cov_type="cluster", cov_kwds={"groups": pd.factorize(s[grupos])[0]})
    sin_corregir = smf.wls(formula_salario, data=s, **ajuste).fit(**robustos)
    corregido = smf.wls(formula_salario + " + imr", data=s, **ajuste).fit(**robustos)
    # Correlación implícita entre los errores de las dos ecuaciones (rho): debe estar entre -1 y 1
    b = corregido.params["imr"]
    sigma = np.sqrt((corregido.resid ** 2).mean() + b ** 2 * (s["imr"] * (s["imr"] + s["xb"])).mean())
    return probit, sin_corregir, corregido, datos, b / sigma
