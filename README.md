# ¿Cuánto *debería* cobrar? Determinantes del salario de las mujeres asalariadas en Argentina

**Trabajo Fin de Máster** · Máster en Data Science, Big Data & Business Analytics · Universidad Complutense de Madrid
**Autora:** Patricia Olguín · 2026

## De qué trata

El trabajo identifica qué características personales, laborales, del hogar y de la vivienda se asocian con salarios más
altos de las mujeres asalariadas en Argentina. También comprueba si esa estructura sirve para predecir el salario en el
futuro inmediato (un año o menos) y la convierte en una **herramienta de referencia salarial**. La herramienta puede usarla
una empresa, para auditar la equidad de su nómina, o un organismo público, para monitorear el mercado de trabajo femenino.

* **Datos:** microdatos de la Encuesta Permanente de Hogares (EPH, INDEC), bases de personas y de hogares, 1T2024–1T2026.
  El análisis parte de las más de 300 variables de ambas bases y selecciona las pertinentes con criterios explícitos.
  Con la base completa de personas (todos los miembros de cada hogar) se construyen además los **hijos propios**, la
  **educación de la pareja** y el **clima educativo del hogar**, que la EPH no pregunta directamente.
* **Variable objetivo:** logaritmo del salario horario relativo a la mediana de todos los asalariados del trimestre. Esta
  medida neutraliza la inflación y la volatilidad macroeconómica sin depender de un deflactor.
* **Modelos:** regresión lineal (ecuación de Mincer), Ridge, Lasso, árbol de decisión, Random Forest, LightGBM y
  Stacking. Todos los experimentos se registran en MLflow.
* **Validación fuera de tiempo:**

  | Conjunto | Período | Uso |
  |---|---|---|
  | Entrenamiento | 1T2024–2T2025 | ajuste de los modelos |
  | Validación | 3T–4T2025 | comparación de modelos, selección de variables y calibración de las bandas |
  | Prueba | 1T2026 | evaluación final, nunca vista durante el desarrollo |
* **Interpretación y producción:**
  * ecuación de Mincer con errores robustos;
  * valores SHAP;
  * validación con origen móvil y monitoreo de *drift* (PSI);
  * bandas salariales conformales al 90%;
  * canales del hogar: efecto de los hijos sobre el acceso al empleo, las horas y el salario por hora;
  * corrección del sesgo de selección con el modelo de Heckman (solo se observa el salario de las que trabajan);
  * mapa de la prima salarial de cada aglomerado, bruta y a igual perfil;
  * brecha de género contrafactual;
  * *welfare ratio* de Robert Allen;
  * aplicación web en Streamlit.

## Estructura del repositorio

| Carpeta | Contenido |
|---|---|
| `notebook/` | `TFM_salarios_mujeres_Argentina.ipynb`: el trabajo completo, incluido el capítulo comparativo con Brasil y España (sección 17) |
| `funciones_tfm.py` | funciones auxiliares que importa el notebook (lectura de la EPH, variables del hogar, depuración, métricas), organizadas como `NuestrasFunciones.py` del módulo de Minería de datos |
| `app/` | `app.py`: aplicación Streamlit (simulador de perfil, auditoría de nómina, desempeño del modelo, hogar y cuidados, territorio) |
| `modelos/` | modelo entrenado (`modelo_AR.joblib`), diccionario de variables, importancias SHAP, coeficientes, monitoreo, canales del hogar, salario potencial, prima por aglomerado y nómina de demostración |
| `resumen/` | memoria del trabajo (20 páginas + anexo) |
| `video/` | presentación de 5 minutos |
| `readme/` | copia de este documento |

Los microdatos **no se suben** al repositorio: se guardan en Google Drive (`MyDrive/TFM/datos_originales`), y el
archivo `.gitignore` lo impide.

## Cómo reproducir el trabajo en Google Colab

1. **Datos:** copie los 18 archivos de la EPH en `MyDrive/TFM/datos_originales/` con el nombre
   `eph_{año}_trim{trimestre}_{pers|hog}.csv`. Si falta alguno, el notebook intenta descargarlo del INDEC. Para el
   capítulo comparativo, copie los microdatos de Brasil y España en `MyDrive/TFM/datos_comparacion/` (la sección 17.1
   indica los nombres y los enlaces de descarga).
2. **Token de GitHub:** el notebook nunca contiene el token.
   1. En GitHub, cree un *fine-grained token* con permiso *Contents: Read and write* solo sobre este repositorio.
   2. En Colab, abra el panel 🔑 **Secretos**, cree el secreto `GITHUB_TOKEN`, pegue el token y active el acceso para
      el notebook.
3. Ejecute `notebook/TFM_salarios_mujeres_Argentina.ipynb` completo (*Entorno de ejecución → Ejecutar todas*). En la
   sección 16 guarda el modelo y sus tablas en `modelos/` y los sube a GitHub; después corre el capítulo comparativo
   (sección 17) y las conclusiones (sección 18).

## Cómo usar la aplicación

En línea, sin instalar nada: la aplicación se publica en [Streamlit Community Cloud](https://share.streamlit.io)
desde este repositorio (rama `main`, archivo `app/app.py`).

En una computadora con Python:

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

## Resultados principales (Argentina)

| Indicador | Resultado |
|---|---|
| Variables | 332 columnas originales (personas y hogares) + 5 construidas con todo el hogar → 143 candidatas tras la exclusión conceptual → 97 tras la depuración → **26 seleccionadas** |
| R² en validación (3T–4T2025) | LightGBM 0,52 · Stacking 0,51 · Random Forest 0,49 · Lasso/Ridge/MCO 0,49 |
| R² en datos futuros (1T2026) | LightGBM **0,44** · Regresión lineal 0,40 |
| Principales determinantes (SHAP) | aglomerado, empleo registrado, jornada, años de educación, comprobante de pago, actividad y ocupación |
| Efectos (Mincer) | +3,4% por año de educación · +25% empleo registrado · +3,3% por tramo de antigüedad · +0,7% por año de experiencia |
| Reparto de la importancia | 63% factores de empresa y política laboral · 14% de la persona · 23% estructurales |
| Hijos y cuidados | no cambian el salario *por hora*, pero cada hijo de 0 a 5 años reduce 6 puntos la probabilidad de estar ocupada y hacerse cargo de las tareas de la casa reduce un 12% las horas |
| Sesgo de selección (Heckman) | selección positiva de las asalariadas (lambda = 0,82); el retorno de la educación corregido sube del 10% al 13% por año; las mujeres que no trabajan tendrían un salario potencial un 29% menor |
| Territorio | a igual perfil, Patagonia +22% y Noreste/Noroeste −17% respecto del promedio del país; la ventaja de CABA (+45% bruta) baja al +13% a igual perfil |
| Estabilidad | población estable (PSI < 0,05); la prima del empleo registrado cayó del 34% al 19% entre 2024 y 2025 (Wald, p < 0,001) → reentrenar cada trimestre |
| Bandas del 90% | cobertura real en el 1T2026: 85% |
| Brecha de género (salario horario) | 4,7 log-puntos; no explicada 6,2 |

Los salarios son declarados en una encuesta y los efectos son asociaciones, no efectos causales.

## Fuentes

* INDEC, Encuesta Permanente de Hogares continua, microdatos y diseño de registro.
* IBGE, PNAD Contínua. INE, Encuesta de Condiciones de Vida y Encuesta de Estructura Salarial.
* Allen, R. C. (2001). The Great Divergence in European Wages and Prices. *Explorations in Economic History*.
* Mincer, J. (1974). *Schooling, Experience, and Earnings*. NBER.
* Romano, Y., Patterson, E. y Candès, E. (2019). Conformalized Quantile Regression. *NeurIPS*.
* Lundberg, S. y Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS*.
