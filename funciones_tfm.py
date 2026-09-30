# -*- coding: utf-8 -*-
"""
Funciones del TFM "Determinantes del salario de las mujeres asalariadas en Argentina"

@author: Patricia Olguín

Siguiendo la organización del módulo de Minería de datos (archivo NuestrasFunciones.py), las funciones auxiliares se
reúnen en este archivo y el notebook las importa con:  from funciones_tfm import *
Varias funciones son adaptaciones de las vistas en el máster (se indica en cada caso).
"""

import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import scipy.stats as stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
import patsy

SEMILLA = 42


# ==================================================================================================================
# 1. LECTURA DE LA EPH
# ==================================================================================================================

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


## Diccionario: descripción legible, bloque temático y carácter (accionable o estructural) de cada variable
DESCRIPCION = {
    "AGLOMERADO": "Aglomerado urbano", "CH10": "Asiste o asistió a un establecimiento educativo",
    "PP03C": "Tiene más de un empleo", "PP3F_TOT": "Horas en otras ocupaciones", "PP03G": "Quería trabajar más horas",
    "PP03H": "Disponibilidad para trabajar más horas", "PP03I": "Buscó trabajar más horas",
    "PP03J": "Buscó otro empleo", "PP03K": "Motivo de búsqueda de otro empleo", "INTENSI": "Intensidad de la ocupación",
    "PP04A1": "Nivel del Estado (nacional, provincial, municipal)", "PP04B1": "Servicio doméstico en casa de familia",
    "PP04B2": "Cantidad de casas en que trabaja (servicio doméstico)", "PP07C": "Empleo con fecha de finalización",
    "PP07E": "Período de prueba, beca o pasantía", "PP07F1": "Le dan de comer en el trabajo",
    "PP07F3": "Le dan productos o mercadería", "PP07F4": "Otros beneficios (celular, auto, pasajes)",
    "PP07F5": "No recibe beneficios en especie", "PP07G1": "Vacaciones pagas", "PP07G2": "Aguinaldo",
    "PP07G3": "Días pagos por enfermedad", "PP07G4": "Obra social por el empleo",
    "PP07H": "Descuento jubilatorio (empleo registrado)", "PP07I": "Aporta por su cuenta a la jubilación",
    "PP07J": "Turno de trabajo", "PP07K": "Comprobante de pago", "PP07L": "El recibo abarca todo el sueldo",
    "PP09A": "Trabaja en CABA, GBA o ambos", "PP07B1_01": "Cobra un plan social por este trabajo",
    "SECTOR": "Sector formal, informal u hogares (INDEC)", "PP07F1_1": "Usa sus propias maquinarias o equipos",
    "IV1": "Tipo de vivienda", "IV2": "Ambientes de la vivienda", "IV3": "Material de los pisos",
    "IV4": "Cubierta del techo", "IV5": "Techo con cielorraso", "IV7": "Fuente del agua", "IV9": "Ubicación del baño",
    "IV10": "Tipo de inodoro", "IV11": "Desagüe del baño", "IV12_1": "Vivienda cerca de basural",
    "IV12_2": "Vivienda en zona inundable", "II2": "Cuartos para dormir", "II3": "Usa un ambiente para trabajar",
    "II4_1": "Tiene cuarto de cocina", "II4_2": "Tiene lavadero", "II4_3": "Tiene garage", "II5": "Duermen en cocina, lavadero o garage",
    "II6": "Usa cocina, lavadero o garage para trabajar", "II7": "Tenencia de la vivienda", "II8": "Combustible para cocinar",
    "V2": "Hogar con jubilación o pensión", "V5_01": "Hogar con AUH o Asignación por Embarazo",
    "V5_02": "Hogar con otro plan social", "V6": "Recibe mercadería del gobierno, iglesias, etc.",
    "V7": "Recibe mercadería de familiares o vecinos", "V11_01": "Hogar con beca de estudio del gobierno",
    "V12": "Recibe cuota alimentaria o ayuda de personas fuera del hogar", "V13": "Gastó ahorros",
    "V14": "Pidió préstamos a familiares o amigos", "V15": "Pidió préstamos a bancos o financieras",
    "V16": "Compra en cuotas o al fiado", "V17": "Vendió pertenencias", "V22_01": "Retroactivo de jubilación",
    "V22_02": "Retroactivo de jubilación de ama de casa", "V22_03": "Retroactivo de otras pensiones",
    "IX_TOT": "Miembros del hogar", "IX_MEN10": "Miembros menores de 10 años", "menores_6": "Menores de 6 años en el hogar",
    "menores_18": "Menores de 18 años en el hogar", "mayores_65": "Mayores de 65 años en el hogar",
    "hijos_propios_menores_6": "Hijos propios menores de 6 años", "hijos_propios_menores_18": "Hijos propios menores de 18 años",
    "educ_pareja": "Años de educación de la pareja", "pareja_ocupada": "La pareja está ocupada",
    "clima_educativo": "Clima educativo de otros adultos del hogar (años)",
    "edad": "Edad", "anios_educ": "Años de educación", "exp_pot": "Experiencia potencial (años)",
    "exp_pot2": "Experiencia potencial al cuadrado", "parentesco": "Parentesco con el jefe/a de hogar",
    "estado_civil": "Estado civil", "cobertura_salud": "Cobertura de salud", "lugar_nacimiento": "Lugar de nacimiento",
    "residencia_5_anios": "Dónde vivía hace 5 años", "cno_caracter": "Carácter de la ocupación (CNO, 2 dígitos)",
    "cno_jerarquia": "Jerarquía de la ocupación", "cno_tecnologia": "Tecnología que utiliza",
    "calificacion": "Calificación de la ocupación", "caes_division": "Actividad del establecimiento (CAES, 2 dígitos)",
    "rama": "Rama de actividad", "tamano_establecimiento": "Personas que trabajan en el establecimiento",
    "sector": "Sector estatal o privado", "lugar_trabajo": "Lugar donde realiza sus tareas",
    "antiguedad": "Antigüedad en el empleo (tramos de 1 a 6)", "jornada_parcial": "Jornada parcial (< 35 horas)",
    "hacinamiento": "Personas por cuarto para dormir", "otros_ocupados_hogar": "Otros ocupados en el hogar",
    "otras_mujeres_adultas": "Otras mujeres adultas en el hogar", "realiza_tareas_casa": "Realiza las tareas de la casa",
    "servicio_domestico_hogar": "El hogar tiene servicio doméstico", "recibe_ayuda_tareas": "Otras personas ayudan con las tareas",
}


def bloque_de(v):
    if v in ("anios_educ", "exp_pot", "exp_pot2", "edad", "CH10", "antiguedad"):
        return "Capital humano"
    if v in ("cno_caracter", "cno_jerarquia", "cno_tecnologia", "calificacion", "caes_division", "rama",
             "tamano_establecimiento", "sector", "PP04A1", "lugar_trabajo", "PP04B1", "PP04B2", "SECTOR"):
        return "Ocupación y empresa"
    if v.startswith("PP0") or v in ("INTENSI", "jornada_parcial", "PP3F_TOT"):
        return "Condiciones de trabajo"
    if v in ("parentesco", "estado_civil", "menores_6", "menores_18", "mayores_65", "otros_ocupados_hogar",
             "otras_mujeres_adultas", "realiza_tareas_casa", "servicio_domestico_hogar", "recibe_ayuda_tareas", "IX_TOT",
             "IX_MEN10", "cobertura_salud", "hijos_propios_menores_6", "hijos_propios_menores_18", "educ_pareja",
             "pareja_ocupada", "clima_educativo"):
        return "Hogar y cuidados"
    if v.startswith(("IV", "II")) or v == "hacinamiento":
        return "Vivienda y hábitat"
    if re.match(r"V\d", v):
        return "Estrategias del hogar"
    return "Territorio y migración"


CARACTER = {"Capital humano": "Accionable (persona)", "Ocupación y empresa": "Accionable (empresa)",
            "Condiciones de trabajo": "Accionable (empresa y política laboral)",
            "Hogar y cuidados": "Estructural (política pública)", "Vivienda y hábitat": "Estructural (política pública)",
            "Estrategias del hogar": "Estructural (política pública)", "Territorio y migración": "Estructural (política pública)"}


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


DESCRIPCION.update({"hijos_0_5": "Hijos propios de 0 a 5 años", "hijos_6_17": "Hijos propios de 6 a 17 años",
                    "nivel_educativo": "Nivel educativo"})


# ==================================================================================================================
# 5. TERRITORIO, RECURSOS NATURALES Y PODER DE COMPRA
# ==================================================================================================================

## Polos de recursos naturales: aglomerados de la EPH cuya economía gira en torno a los hidrocarburos o la minería
POLOS = {17: "Hidrocarburos", 9: "Hidrocarburos",                                   # Vaca Muerta y Golfo San Jorge
         22: "Minería", 23: "Minería", 19: "Minería", 27: "Minería", 20: "Minería"}  # litio, cobre y oro
## Divisiones CAES de empleo directo en el sector: extracción de minerales e hidrocarburos (05-09) y refinación (19)
DIVISIONES_RECURSOS = [5, 6, 7, 8, 9, 19]

## Tamaño de la familia de referencia: hogar tipo 2 del INDEC (pareja de 35 y 31 años con dos hijos de 6 y 8) = 3,09
## adultos equivalentes, casi igual a la familia de 3 adultos equivalentes de Allen (2001)
ADULTOS_EQUIVALENTES = 3.09
HORAS_MES_COMPLETO = 40 * 4.3           # jornada completa de 40 horas semanales


## Función para calcular la Canasta Básica Total (CBT) por adulto equivalente de cada región y trimestre de la EPH
def cbt_por_trimestre(canasta):
    """La CBT es mensual; el salario de la EPH (P21) corresponde al mes anterior a la entrevista, así que para cada
       trimestre se promedia la CBT de los tres meses anteriores (por ejemplo, para el 1T2024: dic-2023 a feb-2024)."""
    canasta = canasta.assign(mes=pd.PeriodIndex(canasta["mes"], freq="M"))
    filas = []
    for ano in range(2024, 2027):
        for tri in range(1, 5):
            meses = pd.period_range(pd.Period(f"{ano}-{3 * tri - 2:02d}", "M") - 1, periods=3, freq="M")
            sub = canasta[canasta["mes"].isin(meses)]
            if sub["mes"].nunique() == 3:
                filas += [{"periodo": ano * 10 + tri, "region": r, "cbt": v}
                          for r, v in sub.groupby("region")["cbt_adulto_equivalente"].mean().items()]
    return pd.DataFrame(filas)


## Función para calcular la paridad de poder adquisitivo (PPA) de 2024 de Argentina, que el Banco Mundial no publica
def ppa_argentina_2024(ppa_2021, ipc):
    """Extrapola la PPA de 2021 con la inflación relativa de Argentina frente a Estados Unidos (método habitual del
       Banco Mundial para los años sin relevamiento de precios). IPC de EE.UU. (BLS, CPI-U): 270,970 (2021) y
       313,689 (2024)."""
    inflacion_ar = ipc[ipc.index.year == 2024].mean() / ipc[ipc.index.year == 2021].mean()
    return ppa_2021 * inflacion_ar / (313.689 / 270.970)


# ==================================================================================================================
# 6. COMPARACIÓN CON BRASIL Y ESPAÑA
# ==================================================================================================================

## Variables comunes a las tres encuestas (EPH, PNAD Contínua y ECV)
COMUNES = ["anios_educ", "exp_pot", "exp_pot2", "calificacion", "rama", "formal", "temporal", "jornada_parcial",
           "en_pareja", "menores_6", "menores_18", "otros_ocupados", "gran_ciudad"]

## Ramas comunes a partir de la sección de la CIIU/CNAE (letra)
SECCION_A_RAMA = {"A": "Agro y minería", "B": "Agro y minería", "C": "Industria", "D": "Construcción y servicios básicos",
                  "E": "Construcción y servicios básicos", "F": "Construcción y servicios básicos", "G": "Comercio",
                  "H": "Transporte e información", "J": "Transporte e información", "I": "Hoteles y restaurantes",
                  "K": "Finanzas y servicios a empresas", "L": "Finanzas y servicios a empresas",
                  "M": "Finanzas y servicios a empresas", "N": "Finanzas y servicios a empresas",
                  "O": "Administración pública", "P": "Enseñanza", "Q": "Salud y servicios sociales",
                  "R": "Otros servicios", "S": "Otros servicios", "U": "Otros servicios", "T": "Servicio doméstico"}
## Divisiones de 2 dígitos -> sección (letra): (última división de la sección, letra)
DIVISION_A_SECCION = [(3, "A"), (9, "B"), (33, "C"), (35, "D"), (39, "E"), (43, "F"), (48, "G"), (53, "H"), (56, "I"),
                      (63, "J"), (66, "K"), (68, "L"), (75, "M"), (82, "N"), (84, "O"), (85, "P"), (88, "Q"), (93, "R"),
                      (96, "S"), (98, "T"), (99, "U")]
## Ramas de la EPH -> ramas comunes
RAMA_EPH_A_COMUN = {"Agro y pesca": "Agro y minería", "Minería": "Agro y minería",
                    "Electricidad, gas y agua": "Construcción y servicios básicos",
                    "Construcción": "Construcción y servicios básicos", "Transporte": "Transporte e información",
                    "Información y comunicación": "Transporte e información",
                    "Finanzas e inmobiliarias": "Finanzas y servicios a empresas",
                    "Servicios profesionales y empresariales": "Finanzas y servicios a empresas",
                    "Organismos extraterritoriales": "Otros servicios"}


def rama_comun(division):
    """Rama común a partir de la división de actividad (2 dígitos)."""
    if pd.isna(division):
        return np.nan
    return next((SECCION_A_RAMA[s] for tope, s in DIVISION_A_SECCION if division <= tope), np.nan)


def calificacion_ciuo(gran_grupo):
    """Calificación a partir del gran grupo de la CIUO-08 (1 dígito): alta = directivos, profesionales y técnicos;
       baja = ocupaciones elementales; media = el resto (0 = fuerzas armadas)."""
    if pd.isna(gran_grupo):
        return np.nan
    return {1: "alta", 2: "alta", 3: "alta", 9: "baja"}.get(int(gran_grupo), "media")


## Función para armar la base común de Argentina a partir de la base de estudio del notebook
def base_comun_argentina(df, mas_500):
    return pd.DataFrame({
        "pais": "Argentina", "periodo": df["periodo"].astype(str), "t": df["t"], "id_hogar": df["id_hogar"],
        "peso": df["PONDIIO"], "mujer": (df["CH04"] == 2) * 1, "salario_horario": df["salario_horario"],
        "anios_educ": df["anios_educ"], "exp_pot": df["exp_pot"],
        "calificacion": df["calificacion"].map({"Profesional": "alta", "Técnica": "alta", "Operativa": "media",
                                                "No calificada": "baja"}),
        "rama": df["rama"].replace(RAMA_EPH_A_COMUN), "formal": df["PP07H"], "temporal": df["PP07C"].fillna(0),
        "jornada_parcial": df["jornada_parcial"], "en_pareja": df["estado_civil"].isin(["Unida", "Casada"]) * 1.0,
        "menores_6": df["menores_6"], "menores_18": df["menores_18"], "otros_ocupados": df["otros_ocupados_hogar"],
        "gran_ciudad": (mas_500.astype(str).str.upper() == "S") * 1.0})


## Funciones de preparación de las bases de Brasil y España (se usan una sola vez: el resultado se guarda en Drive)
def preparar_brasil(carpeta, ondas):
    """PNAD Contínua (IBGE): archivos de ancho fijo leídos con DuckDB, con las posiciones del programa de lectura del IBGE.
       Devuelve los asalariados de 18 a 60 años con las variables comunes."""
    import io, zipfile
    from pathlib import Path
    import duckdb
    cols = ["Ano", "Trimestre", "UPA", "V1008", "V1014", "V1023", "V1028", "V2005", "V2007", "V2009", "VD3005",
            "VD4002", "VD4009", "VD4012", "VD4016", "V4039", "V4010", "V4013", "V4025"]
    texto = {"UPA", "V1008", "V1014", "V4010", "V4013"}
    with zipfile.ZipFile(Path(carpeta) / "Dicionario_e_input.zip") as z:
        programa = z.read(next(n for n in z.namelist() if re.search(r"input.*\.txt$", n, re.I))).decode("latin-1")
    pos = {nombre: (int(ini), int(ancho)) for ini, nombre, ancho in re.findall(r"@(\d+)\s+(\w+)\s+\$?(\d+)\.", programa)}
    partes = []
    for a, q in ondas:
        with zipfile.ZipFile(Path(carpeta) / f"PNADC_{q:02d}{a}.zip") as z:
            txt = Path(z.extract(next(n for n in z.namelist() if n.lower().endswith(".txt")), "/tmp"))
        columnas = ", ".join(f"trim(substr(line, {pos[c][0]}, {pos[c][1]})) AS {c}" if c in texto else
                             f"TRY_CAST(NULLIF(trim(substr(line, {pos[c][0]}, {pos[c][1]})), '') AS DOUBLE) AS {c}" for c in cols)
        partes.append(duckdb.sql(f"""
            WITH p AS (SELECT {columnas} FROM read_csv('{txt}', columns={{'line': 'VARCHAR'}}, header=false, delim='\\t',
                                                       quote='', escape='', auto_detect=false)),
            p2 AS (SELECT *, UPA || '_' || V1008 || '_' || V1014 AS id_hogar FROM p),
            hogar AS (SELECT id_hogar, SUM(CASE WHEN V2009 < 6 THEN 1 ELSE 0 END) AS menores_6,
                             SUM(CASE WHEN V2009 < 18 THEN 1 ELSE 0 END) AS menores_18,
                             SUM(CASE WHEN VD4002 = 1 THEN 1 ELSE 0 END) AS ocupados,
                             MAX(CASE WHEN V2005 IN (2, 3) THEN 1 ELSE 0 END) AS hay_conyuge FROM p2 GROUP BY id_hogar)
            SELECT p2.*, hogar.menores_6, hogar.menores_18, hogar.ocupados, hogar.hay_conyuge FROM p2 JOIN hogar USING (id_hogar)
            WHERE VD4002 = 1 AND VD4009 BETWEEN 1 AND 7 AND V2009 BETWEEN 18 AND 60 AND VD4016 > 0 AND V4039 BETWEEN 1 AND 98
        """).df())
        txt.unlink()
    br = pd.concat(partes, ignore_index=True)
    gran_grupo = pd.to_numeric(br["V4010"].str[:1], errors="coerce")
    division = pd.to_numeric(br["V4013"].str.zfill(5).str[:2], errors="coerce")
    return pd.DataFrame({
        "pais": "Brasil", "periodo": (br["Ano"] * 10 + br["Trimestre"]).astype(int).astype(str),
        "t": ((br["Ano"] - 2024) * 4 + br["Trimestre"] - 1).astype(int), "id_hogar": "BR_" + br["id_hogar"],
        "peso": br["V1028"], "mujer": (br["V2007"] == 2) * 1, "salario_horario": br["VD4016"] / (br["V4039"] * 4.3),
        "anios_educ": br["VD3005"], "exp_pot": (br["V2009"] - br["VD3005"] - 6).clip(lower=0),
        "calificacion": gran_grupo.map(calificacion_ciuo), "rama": division.map(rama_comun),
        "formal": (br["VD4012"] == 1) * 1.0, "temporal": (br["V4025"] == 1) * 1.0, "jornada_parcial": (br["V4039"] < 35) * 1.0,
        "en_pareja": (br["V2005"].isin([2, 3]) | ((br["V2005"] == 1) & (br["hay_conyuge"] == 1))) * 1.0,
        "menores_6": br["menores_6"], "menores_18": br["menores_18"], "otros_ocupados": br["ocupados"] - 1,
        "gran_ciudad": br["V1023"].isin([1, 2]) * 1.0})


def leer_ecv(carpeta, anio, letra):
    """Lee el fichero P, R o D de la ECV (zip del INE con un zip por fichero)."""
    import io, zipfile
    from pathlib import Path
    with zipfile.ZipFile(Path(carpeta) / f"datos_{anio}.zip") as z:
        interno = next(n for n in z.namelist() if re.search(rf"ECV_T{letra}_{anio}\.zip$", n, re.I))
        with zipfile.ZipFile(io.BytesIO(z.read(interno))) as z2:
            archivo = next(n for n in z2.namelist() if re.search(rf"ECV_T{letra}_{anio}\.(tab|csv)$", n, re.I))
            return pd.read_csv(io.BytesIO(z2.read(archivo)), sep="\t", low_memory=False,
                               dtype={"PE041": str, "PL051A": str, "PL111AA": str})


def preparar_espana(carpeta, anios):
    """Encuesta de Condiciones de Vida (INE): asalariados hoy (PL040A = 3) que lo fueron los 12 meses del año anterior
       (el ingreso PY010N es del año anterior). PB205 = 1: convive con su pareja. PL141: 11/12 temporal escrito/verbal,
       21/22 indefinido escrito/verbal; formal = contrato escrito."""
    educ_es = {"0": 0, "1": 6, "2": 10, "3": 12, "4": 13, "5": 15, "6": 16, "7": 17, "8": 20}   # CINE-2011 -> años
    partes = []
    for anio in anios:
        p, r, d = (leer_ecv(carpeta, anio, letra) for letra in ("p", "r", "d"))
        r["hogar"] = r["RB030"] // 100
        hogar = (r.assign(menores_6=r.RB082 < 6, menores_18=r.RB082 < 18, ocupados=r.RB211 == 1)
                  .groupby("hogar")[["menores_6", "menores_18", "ocupados"]].sum())
        p = (p.merge(r[["RB030", "RB082", "hogar"]], left_on="PB030", right_on="RB030")
              .join(hogar, on="hogar").merge(d[["DB030", "DB100"]], left_on="hogar", right_on="DB030"))
        p = p[(p.PL040A == 3) & (p.PL073.fillna(0) + p.PL074.fillna(0) == 12) & p.RB082.between(18, 60)
              & (p.PY010N > 0) & p.PL060.between(1, 98)]
        print(f"ECV {anio}: {len(p):,} asalariados | en pareja {(p.PB205 == 1).mean():.0%} | "
              f"temporales {p.PL141.isin([11, 12]).mean():.0%} (EPA: ~16%)")
        educ = p["PE041"].str.strip().str[:1].map(educ_es)
        partes.append(pd.DataFrame({
            "pais": "España", "periodo": f"ECV {anio}", "t": anio - 2024, "id_hogar": "ES_" + p["hogar"].astype(str),
            "peso": p["PB040"], "mujer": (p["PB150"] == 2) * 1, "salario_horario": p["PY010N"] / 12 / (p["PL060"] * 4.3),
            "anios_educ": educ, "exp_pot": (p["RB082"] - educ - 6).clip(lower=0),
            "calificacion": pd.to_numeric(p["PL051A"].str[:1], errors="coerce").map(calificacion_ciuo),
            "rama": p["PL111AA"].str.strip().str.upper().map(SECCION_A_RAMA),
            "formal": p["PL141"].isin([11, 21]).where(p["PL141"].notna()) * 1.0, "temporal": p["PL141"].isin([11, 12]) * 1.0,
            "jornada_parcial": (p["PL060"] < 35) * 1.0, "en_pareja": (p["PB205"] == 1) * 1.0, "menores_6": p["menores_6"],
            "menores_18": p["menores_18"], "otros_ocupados": p["ocupados"] - 1, "gran_ciudad": (p["DB100"] == 1) * 1.0}))
    return pd.concat(partes, ignore_index=True)


def validar_con_ees(ruta_zip, ecv):
    """Estima el mismo modelo laboral en la Encuesta de Estructura Salarial 2022 (nóminas de empresas) y en la ECV de
       mujeres: si coinciden el signo y el orden de los efectos, la ECV es válida para la comparación."""
    import io, zipfile
    import statsmodels.formula.api as smf
    with zipfile.ZipFile(ruta_zip) as z:
        nombre = next(n for n in z.namelist() if n.lower().endswith((".tab", ".csv")))
        ees = pd.read_csv(io.BytesIO(z.read(nombre)), sep="\t", encoding="latin-1", low_memory=False,
                          dtype={"CNACE": str, "CNO1": str, "ANOS2": str})
    ees.columns = ees.columns.str.strip().str.upper()
    ees = ees[(ees.SEXO == 6) & (ees.DRELABM == 31) & (ees.SIESPM1 == 6)]      # mujeres con el mes completo trabajado
    horas = (ees.JSP1 + ees.JSP2 / 60) * 4.35 + ees.HEXTRA
    edad = ees.ANOS2.str.zfill(2).map({"01": 18, "02": 25, "03": 35, "04": 45, "05": 55, "06": 62})
    educ = ees.ESTU.map({1: 3, 2: 6, 3: 10, 4: 12, 5: 14, 6: 15, 7: 17})
    cno = ees.CNO1.str[0]
    ees = pd.DataFrame({"peso": ees.FACTOTAL, "salario_horario": (ees.SALBASE + ees.COMSAL + ees.PHEXTRA) / horas,
                        "anios_educ": educ, "exp_pot": (edad - educ - 6).clip(lower=0),
                        "calificacion": np.select([cno.isin(list("ABCD")), cno.isin(list("OP"))], ["alta", "baja"], "media"),
                        "rama": ees.CNACE.str[0].map(SECCION_A_RAMA), "temporal": (ees.TIPOCON == 2) * 1.0,
                        "jornada_parcial": (ees.TIPOJOR == 2) * 1.0})[(edad.between(18, 60))].dropna()
    ees = ees[ees.salario_horario > 0]
    formula = ("y ~ anios_educ + exp_pot + I(exp_pot ** 2) + C(calificacion, Treatment('media')) + C(rama) + temporal"
               " + jornada_parcial")
    terminos = {"anios_educ": "Año adicional de educación",
                "C(calificacion, Treatment('media'))[T.alta]": "Ocupación calificada (vs. media)",
                "C(calificacion, Treatment('media'))[T.baja]": "Ocupación no calificada (vs. media)",
                "temporal": "Contrato temporal", "jornada_parcial": "Jornada parcial"}
    efectos = {}
    for fuente, d in {"ECV (hogares)": ecv, "EES 2022 (empresas)": ees}.items():
        d = d.dropna(subset=["anios_educ", "calificacion", "rama"]).assign(
            y=lambda x: np.log(x.salario_horario / mediana_ponderada(x.salario_horario, x.peso)))
        modelo = smf.wls(formula, data=d, weights=d.peso / d.peso.mean()).fit()
        efectos[fuente] = {etiqueta: 100 * (np.exp(modelo.params[t]) - 1) for t, etiqueta in terminos.items()}
    return pd.DataFrame(efectos).round(1)
