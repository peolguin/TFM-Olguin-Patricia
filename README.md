# ¿Cuánto *debería* cobrar? Determinantes del salario de las mujeres asalariadas en Argentina, en perspectiva comparada con Brasil y España (2024-2026)

**Trabajo Fin de Máster** · Máster en Data Science, Big Data & Business Analytics · Universidad Complutense de Madrid
**Autora:** Patricia Olguín · 2026

**Aplicación en línea:** *(pegar aquí la dirección .streamlit.app)*
**Notebook en Colab:** [abrir](https://colab.research.google.com/github/peolguin/TFM-Olguin-Patricia/blob/main/notebook/TFM_salarios_mujeres_Argentina.ipynb)

## De qué trata

La Encuesta Permanente de Hogares (EPH) del INDEC es la fuente más usada para estudiar el mercado de trabajo argentino,
pero sus cuadros publicados no responden preguntas como:

* ¿cuánto paga el mercado por un perfil concreto?
* ¿cuánto de la ventaja salarial de una ciudad desaparece al descontar su costo de vida?
* ¿alcanza el salario de una asalariada para sostener a una familia?

Este trabajo aprovecha los microdatos para obtener esa información indirecta. Identifica los determinantes del salario
de las mujeres asalariadas y comprueba si sirven para predecir el salario a un año. Además, convierte el modelo en una
**herramienta de referencia salarial** que puede usar una empresa (para auditar su nómina) o un organismo público (para
monitorear el mercado de trabajo femenino).

* **Datos:**
  * EPH, bases de personas y hogares, 1T2024–1T2026;
  * Canasta Básica Total regional (INDEC) e IPC (INDEC);
  * PPA del Banco Mundial;
  * para la comparación, la PNAD Contínua (IBGE, Brasil) y la Encuesta de Condiciones de Vida (INE, España).
* **Variable objetivo:** logaritmo del salario horario relativo a la mediana del trimestre. Neutraliza la inflación y
  la volatilidad macroeconómica sin necesidad de deflactar.
* **Modelos:** regresión lineal (ecuación de Mincer), Ridge, Lasso, árbol de decisión, Random Forest y LightGBM.
  Los experimentos se registran en MLflow.
* **Validación fuera de tiempo:** entrenamiento 1T2024–2T2025, validación 3T–4T2025, prueba 1T2026.
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
| `notebook/TFM_salarios_mujeres_Argentina.ipynb` | El trabajo completo en un solo notebook: **parte I**, Argentina (secciones 1-10); **parte II**, comparación con Brasil y España (sección 11); conclusiones generales (sección 12). Unos 15 minutos de ejecución, más 15 la primera vez para preparar las bases de Brasil y España |
| `funciones_tfm.py` | Funciones auxiliares que importa el notebook (organizadas como `NuestrasFunciones.py` del módulo de Minería de datos) |
| `app/app.py` | Aplicación Streamlit |
| `modelos/` | Modelo entrenado y tablas de resultados que usa la aplicación (los genera el notebook) |
| `requirements.txt` | Librerías de la aplicación, con las versiones con que se entrenó el modelo |
| `resumen/` | Memoria del trabajo (20 páginas + anexo) |
| `video/` | Presentación de 5 minutos |
| `readme/` | Copia de este documento |

## Cómo reproducir el trabajo en Google Colab

1. **Datos.**
   * Los microdatos no se suben a GitHub: están en la carpeta de Google Drive `TFM`, compartida con los tutores.
   * Abra el enlace de la carpeta y elija **Organizar → Agregar acceso directo → Mi unidad**. El notebook la encuentra
     en `MyDrive/TFM`.
2. **Notebook.** Abra el [notebook en Colab](https://colab.research.google.com/github/peolguin/TFM-Olguin-Patricia/blob/main/notebook/TFM_salarios_mujeres_Argentina.ipynb)
   y elija **Entorno de ejecución → Ejecutar todas**.
   * Pedirá permiso para acceder a Drive.
   * No hace falta ninguna clave de GitHub: solo la autora sube los resultados al repositorio.

| Carpeta de Drive | Archivos | Fuente |
|---|---|---|
| `TFM/datos_originales/` | `eph_{año}_trim{trimestre}_{pers\|hog}.csv` (18) | INDEC, EPH continua, bases de usuarios |
| `TFM/datos_originales/` | `canasta_basica_regional.csv` | INDEC, informes semestrales de pobreza (CBT por región y mes) |
| `TFM/datos_originales/` | `ipc_indec.csv` | INDEC, IPC nivel general nacional |
| `TFM/datos_originales/` | `ppa_banco_mundial.csv` | Banco Mundial, PA.NUS.PRVT.PP |
| `TFM/datos_comparacion/pnadc/` | `PNADC_{trimestre}{año}.zip` (1T2024-2T2026) y `Dicionario_e_input.zip` | IBGE, PNAD Contínua trimestral |
| `TFM/datos_comparacion/ecv/`, `ees/` | `datos_2024.zip`, `datos_2025.zip`; `datos_2022.zip` | INE, Encuesta de Condiciones de Vida y Encuesta de Estructura Salarial |

La primera ejecución de la parte II reduce los microdatos de Brasil y España a una base común (`TFM/datos_procesados/base_comun_BR_ES.parquet`); las siguientes la leen en segundos.

## La aplicación

La aplicación está publicada en Streamlit Community Cloud (enlace al comienzo de este documento). Tiene cinco pestañas:

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

## Resultados principales (Argentina)

| Indicador | Resultado |
|---|---|
| Variables | 332 columnas originales + 5 construidas con todo el hogar → 97 candidatas → **26 seleccionadas** |
| Predicción a un año (1T2026) | LightGBM R² = 0,44 · regresión lineal 0,40 · origen móvil 0,44–0,55 |
| Principales determinantes (SHAP) | aglomerado, empleo registrado, jornada, educación, comprobante de pago, actividad y ocupación |
| Reparto de la importancia | 63% empresa y política laboral · 14% persona · 23% factores estructurales |
| Hijos y cuidados | no cambian el salario *por hora*; cada hijo de 0 a 5 años reduce unos 6 puntos la probabilidad de estar ocupada |
| Estabilidad | composición estable (PSI < 0,05); la prima del empleo registrado cayó del 34% al 19% entre 2024 y 2025 → reentrenar cada trimestre |

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
