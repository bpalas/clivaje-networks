# clivaje-networks

Análisis de polarización en redes signadas (grafo con aristas positivas/negativas) y métricas de frontera para detectar actores nucleares, puentes y comportamientos anómalos. Incluye:

- Algoritmos de clustering (eigensign, búsqueda local tipo paper 2025 k=2, SCG).
- Métricas de evaluación y de frontera (sesgo de equilibrio, centralidad del núcleo, índice de frontera, etc.).
- Visualizaciones para explorar el núcleo, la frontera y la agencia de actores.


## Requisitos

- Python 3.10+
- Instalar dependencias:

```
pip install -r requirements.txt
```


## Estructura

- `src/pcd/`: paquete principal (preprocesamiento, algoritmos, análisis, visualización).
- `src/scripts/run_pcd.py`: ejecuta el clustering y genera un CSV con columna `CLUSTER`.
- `src/scripts/run_metrics.py`: script original de métricas y gráficos (usa variables globales).
- `src/scripts/run_metrics_cli.py`: envoltorio CLI para `run_metrics.py` (recomendado).
- `docs/metricas_frontera.md`: documentación de métricas de frontera.
- `docs/algoritmo_local_search_k2.md`: detalle del algoritmo de búsqueda local (k=2).
- `data/`: directorio sugerido para entradas; `data/datasets_cluster/` para salidas del clustering.
- `results/`: directorio sugerido para salidas de métricas/figuras.


## Datos de entrada

CSV con columnas mínimas:

- `FROM_NODE`: nombre del nodo origen
- `TO_NODE`: nombre del nodo destino
- `SIGN`: signo de la interacción (1, 0, -1 o strings como positive/neutral/negative)


## Uso

1) Generar dataset clusterizado (agrega columna `CLUSTER`):

```
python src/scripts/run_pcd.py --input "data/*.csv" --output-dir data/datasets_cluster \
  --algo local_search_paper_k2 --k 2 --ls-beta 0.005 --ls-max-iter 2000
```

Salidas en `data/datasets_cluster/`:

- `*_clustered_YYYYmmdd_HHMMSS.csv` (aristas con `CLUSTER`)
- `*_metrics_YYYYmmdd_HHMMSS.csv` (métricas de evaluación)
- `*_nodes_clusters_YYYYmmdd_HHMMSS.csv` (asignación de nodos)

2) Calcular métricas de frontera y visualizaciones:

Opción A (recomendada, CLI):

```
python src/scripts/run_metrics_cli.py --input data/datasets_cluster/tu_archivo_clustered_*.csv \
  --results-dir results
```

Opción B (script original):

- Editar `src/scripts/run_metrics.py` para que `INPUT_CSV_PATH` apunte a tu CSV.
- Ejecutar:

```
python src/scripts/run_metrics.py
```

Figuras y métricas quedan en `results/`.


## Notas de migración y paths

- Se corrigió la importación en `run_pcd.py` para usar `pcd.pipeline` agregando `src/` al `PYTHONPATH` en tiempo de ejecución.
- Se añadió `run_metrics_cli.py` para parametrizar la ruta de entrada y de salida sin tocar el script original.
- Si `run_metrics_cli.py` no encuentra `--input`, intentará usar el CSV más reciente en `data/datasets_cluster/`.


## Licencia

MIT. Ver `LICENSE`.
