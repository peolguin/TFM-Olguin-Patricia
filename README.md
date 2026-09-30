# Determinantes y predicción del salario de las mujeres ocupadas en Argentina (2024-2026)

### Una comparación con Brasil y España y una herramienta de referencia salarial

**Trabajo Fin de Máster** · Máster en Data Science, Big Data & Business Analytics · Universidad Complutense de Madrid
**Autora:** Patricia Olguín · 2026

**Aplicación en línea:** [tfm-olguin-patricia.streamlit.app](https://tfm-olguin-patricia.streamlit.app/)
**Notebook en Colab:** [formato ipynb](https://colab.research.google.com/github/peolguin/TFM-Olguin-Patricia/blob/main/notebook/TFM_salarios_mujeres_Argentina.ipynb) y [html](https://peolguin.github.io/TFM-Olguin-Patricia/notebook/TFM_salarios_mujeres_Argentina.html)

## De qué trata

El trabajo intenta determinar los factores que influyen en el salario de las mujeres ocupadas en Argentina y predecir sus valores en el corto plazo. Tiene en cuenta una perspectiva comparativa con otros países y presenta una herramienta de referencia salarial Aprovecha los microdatos de la Encuesta Permanente de Hogares (EPH) calculada por el Instituto Nacional de Estadísticas y Censos, a fin de obtener información de manera indirecta porque los cuadros publicados no responden las siguientes preguntas:

* ¿cuánto paga el mercado por un perfil concreto, y con qué margen?
* ¿cuánto de la diferencia salarial entre ciudades desaparece al descontar el costo de vida?
* ¿llega la riqueza de Vaca Muerta y de la minería al salario de las mujeres?
* ¿alcanza el salario de una asalariada para sostener a una familia (*welfare ratio* de Allen)?
* ¿cuánto de la brecha de género no se explica por el perfil, y supera el umbral del 5% de la Directiva europea de
  transparencia retributiva?

En la **parte I** identifica los determinantes del salario de las mujeres asalariadas en Argentina, comprueba si sirven para predecir el salario a un año y convierte el modelo en una **herramienta de referencia salarial** que puede usar una empresa (para auditar su nómina) o un organismo público (para monitorear el mercado de trabajo femenino). En la **parte II** repite el análisis con las mismas variables en Brasil y España, para distinguir lo que es propio del mercado argentino de lo que es general.

* **Datos:**
  * EPH, bases de personas y hogares, 1T2024 a 1T2026;
  * Canasta Básica Total regional (INDEC) e IPC (INDEC);
  * PPA del Banco Mundial;
  * para la comparación, la PNAD Contínua (IBGE, Brasil) y la Encuesta de Condiciones de Vida (INE, España).
* **Variable objetivo:** logaritmo del salario horario relativo a la mediana del trimestre. Neutraliza la inflación y
  la volatilidad macroeconómica sin necesidad de deflactar.
* **Modelos:** regresión lineal (ecuación de Mincer), Ridge, Lasso, árbol de decisión, Random Forest y LightGBM.
  Los experimentos se registran en MLflow.
* **Validación fuera de tiempo:** entrenamiento 1T2024 a 2T2025, validación 3T a 4T2025, prueba 1T2026.
* **Interpretación:**
  * valores SHAP;
  * canales del hogar y corrección de Heckman;
  * prima territorial nominal y real;
  * polos energéticos y mineros;
  * *welfare ratio* de Robert Allen con la canasta del INDEC;
  * brecha de género frente al umbral del 5% de la Directiva (UE) 2023/970;
  * comparación con Brasil y España.

## Estructura del repositorio

| Carpeta / archivo | Contenido |
|---|---|
| notebook/TFM\_salarios\_mujeres\_Argentina.ipynb | El trabajo completo en un solo notebook: **parte I**, Argentina (secciones 1-10); **parte II**, comparación con Brasil y España (sección 11); conclusiones generales (sección 12). Unos 15 minutos de ejecución, más 15 la primera vez para preparar las bases de Brasil y España |
| funciones\_tfm.py | Funciones auxiliares que importa el notebook |
| app/app.py | Aplicación Streamlit |
| modelos/ | Modelo entrenado y tablas de resultados que usa la aplicación (los genera el notebook) |
| requirements.txt | Librerías de la aplicación, con las versiones con que se entrenó el modelo |
| resumen/ | Memoria del trabajo (20 páginas + anexo) |
| video/ | Presentación de 5 minutos |
| readme/ | Copia de este documento |

## Cómo reproducir el trabajo en Google Colab

1. **Datos.**
   * Los microdatos no se suben a GitHub: están en la carpeta de Google Drive TFM, compartida con los tutores.
   * Abra el enlace de la carpeta y elija **Organizar**, **Agregar acceso directo** y **Mi unidad**. El notebook la encuentra
     en MyDrive/TFM.
2. **Notebook.** Abra el [notebook en Colab](https://colab.research.google.com/github/peolguin/TFM-Olguin-Patricia/blob/main/notebook/TFM_salarios_mujeres_Argentina.ipynb)
   y elija el menú **Entorno de ejecución**, opción **Ejecutar todas**.
   * Pedirá permiso para acceder a Drive.
   * No hace falta ninguna clave de GitHub: solo la autora sube los resultados al repositorio.

| Carpeta de Drive | Archivos | Fuente |
|---|---|---|
| TFM/datos\_originales/ | eph\_{año}\_trim{trimestre}\_{pers\|hog}.csv (18) | INDEC, EPH continua, bases de usuarios |
| TFM/datos\_originales/ | canasta\_basica\_regional.csv | INDEC, informes semestrales de pobreza (CBT por región y mes) |
| TFM/datos\_originales/ | ipc\_indec.csv | INDEC, IPC nivel general nacional |
| TFM/datos\_originales/ | ppa\_banco\_mundial.csv | Banco Mundial, PA.NUS.PRVT.PP |
| TFM/datos\_comparacion/pnadc/ | PNADC\_{trimestre}{año}.zip (1T2024-2T2026) y Dicionario\_e\_input.zip | IBGE, PNAD Contínua trimestral |
| TFM/datos\_comparacion/ecv/, ees/ | datos\_2024.zip, datos\_2025.zip; datos\_2022.zip | INE, Encuesta de Condiciones de Vida y Encuesta de Estructura Salarial |

La primera ejecución de la parte II reduce los microdatos de Brasil y España a una base común (TFM/datos\_procesados/base\_comun\_BR\_ES.parquet); las siguientes la leen en segundos.

## La aplicación

La aplicación está publicada en Streamlit Community Cloud: [tfm-olguin-patricia.streamlit.app](https://tfm-olguin-patricia.streamlit.app/). Tiene cinco pestañas:

| Pestaña | Pregunta que responde |
|---|---|
| ¿Cuánto debería cobrar? | Salario de mercado y banda del 90% para un perfil, y si ese salario alcanza para sostener a una familia (*welfare ratio*) |
| Auditoría de nómina | Quiénes cobran por debajo del mercado, cuánto costaría corregirlo y cuál es la brecha de género a igual perfil frente al umbral del 5% |
| Territorio y recursos naturales | Prima salarial nominal y real de cada ciudad y de los polos energéticos y mineros |
| Hogar y cuidados | Cómo afectan los hijos el empleo, las horas y el salario de las mujeres |
| Argentina, Brasil y España | Qué factores premia cada mercado y cuánto vale el salario en cada país |

Para ejecutarla en una computadora:

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

## Resultados principales

### Parte I: Argentina

| Indicador | Resultado |
|---|---|
| Variables | 319 variables de personas y hogares (incluidas 5 construidas con todo el hogar), 97 candidatas y **26 seleccionadas** |
| Predicción a un año (1T2026) | LightGBM R² = 0,44 · regresión lineal 0,40 · origen móvil entre 0,44 y 0,55 |
| Principales determinantes (SHAP) | aglomerado, empleo registrado, jornada, educación, comprobante de pago, actividad y ocupación |
| Reparto de la importancia | 63% empresa y política laboral · 14% persona · 23% factores estructurales |
| Hijos y cuidados | no cambian el salario *por hora*; cada hijo de 0 a 5 años reduce 6 puntos la probabilidad de estar ocupada |
| Territorio y costo de vida | a igual perfil, 39 puntos de diferencia entre regiones en pesos y 9 puntos descontando la canasta regional |
| Polos de recursos naturales | hidrocarburos: +26% a igual perfil y +5,5% en términos reales · minería: −12% y 0% real |
| *Welfare ratio* (Allen) | el salario mediano de una asalariada a jornada completa cubre 0,75 canastas básicas de una familia tipo (1T2026); 0,51 en el 1T2024 |
| Brecha de género | observada 4,8% · no explicada 6,4% (umbral de la Directiva europea: 5%) · 13-15% en comercio, industria y servicios profesionales |
| Estabilidad | composición estable (PSI < 0,05); la prima del empleo registrado cayó del 34% al 19% entre 2024 y 2025, por lo que conviene reentrenar cada trimestre |
| Bandas del 90% | cobertura real en el 1T2026: 85% |

### Parte II: Argentina, Brasil y España (mujeres asalariadas)

| Indicador | Argentina | Brasil | España |
|---|---|---|---|
| Año adicional de educación | +5,1% | +5,8% | +3,9% |
| Ocupación calificada (vs. media) | +21% | +65% | +34% |
| Empleo registrado / contrato escrito | +44% | +34% | +16% |
| Jornada parcial (por hora) | +27% | +29% | +8% |
| Bloque con más peso (SHAP) | condiciones de trabajo (44%) | capital humano (39%) | ocupación y empresa (43%) |
| Brecha de género observada / no explicada | 4% / 6% | 6% / 15% | 7% / 11% |
| Salario horario mediano (USD PPA 2024) | 6,3 | 4,2 | 16,8 |

Los salarios son declarados en una encuesta y los efectos son asociaciones, no efectos causales.

## Fuentes

* INDEC: Encuesta Permanente de Hogares continua; Canasta Básica Total por región; Índice de Precios al Consumidor.
* IBGE: PNAD Contínua. INE: Encuesta de Condiciones de Vida y Encuesta de Estructura Salarial.
* Banco Mundial: World Development Indicators (PPA) y líneas de pobreza internacionales.
* Allen, R. C. (2001). The Great Divergence in European Wages and Prices. *Explorations in Economic History*.
* Heckman, J. (1979). Sample Selection Bias as a Specification Error. *Econometrica*.
* Mincer, J. (1974). *Schooling, Experience, and Earnings*. NBER.
* Romano, Y., Patterson, E. y Candès, E. (2019). Conformalized Quantile Regression. *NeurIPS*.
* Lundberg, S. y Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS*.
* Directiva (UE) 2023/970 sobre transparencia retributiva.

### Trabajos previos de la autora sobre salarios reales y bienestar

* Olguín, P. y Bragoni, B. (2020). Salarios reales y subsistencia de los trabajadores de Mendoza durante la gran expansión (Argentina, 1890-1914). *Revista de Historia Económica - Journal of Iberian and Latin American Economic History*, 1-28. [doi:10.1017/S0212610920000051](https://doi.org/10.1017/S0212610920000051).
* Olguín, P. y Bragoni, B. (2023). Salarios de mujeres e ingreso familiar de subsistencia en Mendoza (Argentina) a principios del siglo XX. *Investigaciones de Historia Económica*, Asociación Española de Historia Económica. [doi:10.33231/j.ihe.2023.04.001](https://doi.org/10.33231/j.ihe.2023.04.001).
* Olguín, P. y Bragoni, B. (2025). Convivir para subsistir: salarios reales de varones y mujeres en Mendoza (primera mitad del siglo XX). En D. Santilli (comp.), *La Argentina entre tres siglos. Precios, salarios y desigualdad en la larga duración (1776-1945)*. Buenos Aires: Prometeo.
* Olguín, P. (2025). Entre el bienestar y la subsistencia. Salarios de varones y mujeres en Mendoza en el siglo XX. XXIX Jornadas de Historia Económica, AAHE y Universidad Nacional de Jujuy, San Salvador de Jujuy, septiembre de 2025.
* Olguín, P., Mahnic, P., Norton, A. y Gómez, F. (2026). Lo que muestran y lo que ocultan las estadísticas. Desempeño económico y desigualdad en Mendoza en el largo plazo, 1895-2023. XXIX Jornadas de Investigación, SIIP, Universidad Nacional de Cuyo, Mendoza, abril de 2026.
