import os
import sys
import glob
import argparse
from datetime import datetime


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))


def parse_args():
    parser = argparse.ArgumentParser(description="Pipeline: clustering PCD k=2 + métricas SH")
    parser.add_argument(
        "--input",
        nargs="+",
        help="Ruta(s) CSV crudo(s) con columnas FROM_NODE, TO_NODE, SIGN. Acepta comodines entre comillas."
    )
    parser.add_argument(
        "--clustered-input",
        help="Opción alternativa: pasar directamente un CSV ya clusterizado (FROM_NODE, TO_NODE, SIGN, CLUSTER)."
    )
    parser.add_argument(
        "--results-dir",
        default=os.path.join(PROJECT_ROOT, 'results'),
        help="Directorio base de resultados (por defecto: results)"
    )
    parser.add_argument("--name", help="Nombre base del dataset (si se procesa un único archivo)")
    parser.add_argument("--topk", type=int, default=20, help="Top-k para ranking SH")

    # Hiperparámetros LS k=2
    parser.add_argument("--ls-beta", type=float, default=0.005, help="Parámetro beta de Local Search")
    parser.add_argument("--ls-max-iter", type=int, default=2000, help="Iteraciones máximas de Local Search")
    return parser.parse_args()


def main():
    args = parse_args()

    # Preparar imports de los scripts existentes
    sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src', 'scripts'))
    try:
        import run_pcd as rp  # type: ignore
        import run_metrics as rm  # type: ignore
    except Exception as e:
        print(f"Error importando scripts: {e}. Ejecuta desde la raíz del repo.")
        sys.exit(1)

    os.makedirs(args.results_dir, exist_ok=True)

    # Caso 1: usar directamente un CSV ya clusterizado
    if args.clustered_input:
        clustered_path = os.path.normpath(args.clustered_input)
        dataset_name = args.name or os.path.splitext(os.path.basename(clustered_path))[0]
        out_dir = os.path.join(args.results_dir, dataset_name)
        os.makedirs(out_dir, exist_ok=True)
        print(f"[Pipeline] Ejecutando métricas de frontera sobre: {clustered_path}")
        rm.run_frontier_metrics(clustered_path, out_dir, top_k=int(max(1, args.topk)))
        print(f"[Listo] Resultados en: {out_dir}")
        return

    # Caso 2: ejecutar clustering primero
    inputs = []
    if args.input:
        for pattern in args.input:
            expanded = glob.glob(pattern)
            if expanded:
                inputs.extend(expanded)
            else:
                print(f"Advertencia: no se encontraron archivos para el patrón: {pattern}")

    if not inputs:
        print("No hay entradas. Usa --input para CSV crudos o --clustered-input para CSV clusterizado.")
        sys.exit(1)

    for ipath in inputs:
        dataset_name = args.name if (args.name and len(inputs) == 1) else os.path.splitext(os.path.basename(ipath))[0]
        print(f"\n=== Procesando dataset: {dataset_name} ===")

        # Config clustering (PCD k=2 con neutrales)
        config = {
            "name": f"local_search_paper_k2 (beta={args.ls_beta})",
            "algorithm_type": "local_search_paper_k2",
            "k": 2,
            "ls_beta": args.ls_beta,
            "ls_max_iter": args.ls_max_iter,
        }

        # Directorio donde run_pcd guarda los clusterizados
        cluster_out_dir = os.path.join(PROJECT_ROOT, 'data', 'datasets_cluster')
        os.makedirs(cluster_out_dir, exist_ok=True)

        # 1) Clustering
        res = rp.ejecutar_clustering_individual(
            dataset_path=os.path.normpath(ipath),
            dataset_name=dataset_name,
            config_clustering=config,
            output_dir=cluster_out_dir,
        )
        if not res or 'dataset_path' not in res:
            print(f"Error en clustering para {dataset_name}")
            continue

        clustered_csv = res['dataset_path']

        # 2) Métricas de frontera + ranking SH
        out_dir = os.path.join(args.results_dir, dataset_name)
        os.makedirs(out_dir, exist_ok=True)
        print(f"[Pipeline] Ejecutando métricas de frontera y SH ranking (top-{args.topk})...")
        rm.run_frontier_metrics(clustered_csv, out_dir, top_k=int(max(1, args.topk)))
        print(f"[Listo] Resultados en: {out_dir}")


if __name__ == "__main__":
    main()

