import os
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
# quicktests/ -> scripts/ -> src/ -> repo root
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))

if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

try:
    import run_pcd
except Exception as e:
    logging.error(f"No se pudo importar run_pcd: {e}")
    sys.exit(1)

try:
    import run_metrics
except Exception as e:
    logging.error(f"No se pudo importar run_metrics: {e}")
    sys.exit(1)


def main():
    dataset_path = os.path.join(PROJECT_ROOT, 'data', 'raw', 'primarias2025.csv')
    if not os.path.exists(dataset_path):
        logging.error(f"No existe el dataset esperado: {dataset_path}")
        sys.exit(1)

    output_dir = os.path.join(PROJECT_ROOT, 'data', 'datasets_cluster')
    results_dir = os.path.join(PROJECT_ROOT, 'results', 'Primarias_2025')
    os.makedirs(results_dir, exist_ok=True)

    config = {
        "name": "local_search_paper_k2 (beta=0.005)",
        "algorithm_type": "local_search_paper_k2",
        "k": 2,
        "ls_beta": 0.005,
        "ls_max_iter": 2000,
    }

    logging.info("Ejecutando clustering para Primarias_2025...")
    r = run_pcd.ejecutar_clustering_individual(
        dataset_path=dataset_path,
        dataset_name='Primarias_2025',
        config_clustering=config,
        output_dir=output_dir,
    )
    if not r:
        logging.error("El clustering no produjo resultados.")
        sys.exit(1)

    clustered_csv = r['dataset_path']
    logging.info(f"CSV clusterizado: {clustered_csv}")

    logging.info("Calculando métricas de frontera y visualizaciones...")
    run_metrics.run_frontier_metrics(clustered_csv, results_dir)


if __name__ == '__main__':
    main()
