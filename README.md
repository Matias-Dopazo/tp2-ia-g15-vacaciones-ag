# TP2 — Inteligencia Artificial — Grupo 15

**Optimización del cronograma anual de vacaciones en una empresa de manufactura con múltiples restricciones mediante Algoritmos Genéticos**

UTN — Facultad Regional Buenos Aires · Inteligencia Artificial · 2.º cuatrimestre 2026

## Descripción

Algoritmo Genético que asigna las vacaciones anuales de 500 empleados en 52 semanas, respetando:

- días de vacaciones por antigüedad (14, 21, 28 o 35), en módulos de 7 días;
- semana 1: el personal técnico no puede tomar vacaciones (parada de mantenimiento) y el no técnico debe tomarla;
- cobertura semanal mínima por certificación técnica (5 grupos) y por área operativa (4 grupos);
- preferencias de cada empleado como criterio secundario.

**Cromosoma:** un gen entero por empleado con la semana de inicio de su bloque de vacaciones; se decodifica a la matriz binaria 500 × 52 (1 = vacaciones, 0 = trabajo).

**Aptitud:** minimización lexicográfica de `J = (P, U, M, Q)`:
`P` = suma de faltantes al cuadrado, `U` = faltantes totales, `M` = peor faltante, `Q` = desvío ponderado respecto de la semana preferida.

**Operadores:** selección por torneo (k = 3), cruce por grupos completos, mutación de reubicación de inicio e intercambio, elitismo.

## Estructura

```
codigo/
  solucion_ag.py           AG, heurística voraz de comparación y generación de resultados
  verificar_resultados.py  validación independiente de los CSV y cronogramas
  regenerar_graficos.py    regenera los gráficos a partir de los logs
datos/
  empleados_caso.csv       500 empleados: grupo, antigüedad, días, semana preferida, prioridad
  demanda_semanal.csv      personas requeridas por grupo y semana
resultados/
  log_corridas.csv         log por generación de las 25 corridas
  parametros_corridas.csv  parámetros de las 5 configuraciones
  resumen_corridas.csv     resultado inicial y final de cada corrida
  mejor_cronograma.csv     mejor cronograma obtenido
  cobertura_mejor_cronograma.csv
  cronogramas_25_corridas.csv
  comparacion_voraz.csv, voraz_1..5.csv
graficos/                  evolución de la aptitud por configuración y gráficos auxiliares
```

## Ejecución

Requiere Python 3.10 o superior.

```
pip install -r requirements.txt
python codigo/solucion_ag.py
python codigo/verificar_resultados.py
```

`solucion_ag.py` regenera los datos, las 25 corridas (5 configuraciones × 5 semillas), los CSV y los gráficos. Los resultados son reproducibles por las semillas fijas.

## Resultados

| Config. | Población | Generaciones | Mutación/hijo | Elitismo | Corridas con P = 0 |
|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 320 | 0,85 | 3 | 5/5 |
| 2 | 64 | 320 | 0,45 | 3 | 2/5 |
| 3 | 96 | 320 | 0,85 | 3 | 5/5 |
| 4 | 64 | 450 | 0,85 | 3 | 5/5 |
| 5 | 64 | 320 | 0,85 | 8 | 5/5 |

- 22 de 25 corridas alcanzan cobertura completa (P = 0).
- Mejor solución: configuración 4, repetición 5, `J = (0, 0, 0, 2628)`.
- Heurística voraz de comparación: cobertura completa en 1 de 5 ejecuciones (mejor Q = 1554).

Los datos son un escenario de evaluación construido para el trabajo; no corresponden a una empresa real.

## Integrantes

Andreacchio Juan Cruz · Falco Marco · Mollon Rodrigo Javier · Su Facundo · Amore María Soledad · Padovani Franco · Dopazo Matías · Alonso Fiasche Ezequiel
