import os
import sys
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

def _find_latest_clustered_csv(base_dir: str) -> str | None:
    import glob
    datasets_dir = os.path.join(base_dir, 'data', 'datasets_cluster')
    patterns = [
        os.path.join(datasets_dir, '*_clustered_*.csv'),
        os.path.join(datasets_dir, '*.csv'),
    ]
    candidates = []
    for pat in patterns:
        candidates.extend(glob.glob(pat))
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def parse_args():
    parser = argparse.ArgumentParser(description="CLI para ejecutar métricas y visualizaciones de frontera")
    parser.add_argument(
        "--input",
        help="Ruta al CSV ya clusterizado (columnas: FROM_NODE, TO_NODE, SIGN, CLUSTER). Si se omite, busca el más reciente en data/datasets_cluster."
    )
    parser.add_argument(
        "--results-dir",
        default=os.path.join(PROJECT_ROOT, 'results'),
        help="Directorio donde guardar resultados (por defecto: results)."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Importar el script original y setear parámetros globales
    sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src', 'scripts'))
    import run_metrics as rm  # type: ignore

    input_path = args.input
    if not input_path:
        input_path = _find_latest_clustered_csv(PROJECT_ROOT)
        if not input_path:
            print("No se encontró un CSV en data/datasets_cluster. Especifica --input.")
            sys.exit(1)

    rm.INPUT_CSV_PATH = os.path.normpath(input_path)
    rm.RESULTS_DIR = os.path.normpath(args.results_dir)

    os.makedirs(rm.RESULTS_DIR, exist_ok=True)
    rm.main()


if __name__ == "__main__":
    main()

