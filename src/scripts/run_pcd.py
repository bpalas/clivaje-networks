import sys
import os
import argparse
import glob
import pandas as pd
import numpy as np
import logging
from datetime import datetime

# --- Configuración de logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# --- Rutas del proyecto e imports del paquete ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

try:
    from pcd import pipeline
except (ImportError, FileNotFoundError) as e:
    logging.error(f"No se pudo importar 'pcd.pipeline': {e}. Asegúrate de ejecutar desde la raíz del repo y que 'src/pcd' exista.")
    sys.exit(1)


def ejecutar_clustering_individual(dataset_path: str, dataset_name: str, config_clustering: dict, output_dir: str):
    """
    Ejecuta clustering en un dataset individual y guarda los resultados.

    Args:
        dataset_path: Ruta al archivo CSV del dataset.
        dataset_name: Nombre identificador del dataset.
        config_clustering: Configuración del algoritmo de clustering.
        output_dir: Directorio de salida para resultados.

    Returns:
        dict: Métricas e información de archivos generados.
    """
    logging.info(f"Iniciando análisis individual para: {dataset_name}")

    # Cargar dataset
    try:
        df_dataset = pd.read_csv(dataset_path, sep=',', on_bad_lines='skip')
        logging.info(f"Dataset cargado: {len(df_dataset)} registros")
    except FileNotFoundError:
        logging.error(f"No se encontró el archivo: {dataset_path}")
        return None
    except Exception as e:
        logging.error(f"Error cargando dataset: {e}")
        return None

    # Validar columnas requeridas
    required_cols = ['FROM_NODE', 'TO_NODE', 'SIGN']
    missing_cols = [col for col in required_cols if col not in df_dataset.columns]
    if missing_cols:
        logging.error(f"Faltan columnas requeridas: {missing_cols}")
        return None

    # Configuración por defecto para el pipeline
    default_cols = {
        'from_node_col': 'FROM_NODE',
        'to_node_col': 'TO_NODE',
        'sign_col': 'SIGN'
    }

    # Ejecutar análisis de clustering
    try:
        df_nodes_results, _, paper_metrics, _, _ = pipeline.ejecutar_analisis_polarizacion(
            df_input=df_dataset.copy(),
            config=config_clustering.copy(),
            default_cols=default_cols,
            calculate_intra_cluster_cc=True
        )

        if df_nodes_results is None or df_nodes_results.empty:
            logging.error("No se generaron resultados del clustering")
            return None

    except Exception as e:
        logging.error(f"Error en el análisis de clustering: {e}")
        return None

    # Preparar dataset original con clusters
    df_original_con_clusters = df_dataset.copy()

    # Normalizar nombres de nodos para el mapeo
    df_original_con_clusters['FROM_NODE_norm'] = df_original_con_clusters['FROM_NODE'].str.lower().str.strip()
    df_original_con_clusters['TO_NODE_norm'] = df_original_con_clusters['TO_NODE'].str.lower().str.strip()
    df_nodes_results['NODE_NAME_norm'] = df_nodes_results['NODE_NAME'].str.lower().str.strip()

    # Crear mapeo de nodos a clusters
    mapa_nodo_cluster = pd.Series(
        df_nodes_results.CLUSTER_ASSIGNMENT.values,
        index=df_nodes_results.NODE_NAME_norm
    ).to_dict()

    # Asignar clusters a los nodos FROM y TO
    df_original_con_clusters['FROM_CLUSTER'] = df_original_con_clusters['FROM_NODE_norm'].map(mapa_nodo_cluster).fillna(0)
    df_original_con_clusters['TO_CLUSTER'] = df_original_con_clusters['TO_NODE_norm'].map(mapa_nodo_cluster).fillna(0)

    # Usamos el cluster del nodo FROM como cluster principal de la arista; si FROM no tiene y TO sí, usamos TO
    df_original_con_clusters['CLUSTER'] = df_original_con_clusters['FROM_CLUSTER']
    mask_from_sin_cluster = (df_original_con_clusters['FROM_CLUSTER'] == 0) & (df_original_con_clusters['TO_CLUSTER'] != 0)
    df_original_con_clusters.loc[mask_from_sin_cluster, 'CLUSTER'] = df_original_con_clusters.loc[mask_from_sin_cluster, 'TO_CLUSTER']

    # Mantener solo columnas requeridas
    columnas_finales = ['FROM_NODE', 'TO_NODE', 'SIGN', 'CLUSTER']
    df_resultado_final = df_original_con_clusters[columnas_finales].copy()
    df_resultado_final['CLUSTER'] = df_resultado_final['CLUSTER'].astype(int)

    # Crear directorio de salida
    os.makedirs(output_dir, exist_ok=True)

    # Timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Guardar dataset con clusters
    output_dataset_path = os.path.join(output_dir, f"{dataset_name}_clustered_{timestamp}.csv")
    df_resultado_final.to_csv(output_dataset_path, index=False, sep=',', encoding='utf-8-sig')
    logging.info(f"Dataset con clusters guardado: {output_dataset_path}")

    # Preparar y guardar métricas
    metricas_con_info = paper_metrics.copy()
    metricas_con_info['Dataset'] = dataset_name
    metricas_con_info['Algoritmo'] = config_clustering['name']
    metricas_con_info['Timestamp'] = timestamp
    metricas_con_info['Total_Edges'] = len(df_resultado_final)
    metricas_con_info['Edges_Cluster_1'] = len(df_resultado_final[df_resultado_final['CLUSTER'] == 1])
    metricas_con_info['Edges_Cluster_-1'] = len(df_resultado_final[df_resultado_final['CLUSTER'] == -1])
    metricas_con_info['Edges_Cluster_0'] = len(df_resultado_final[df_resultado_final['CLUSTER'] == 0])

    output_metrics_path = os.path.join(output_dir, f"{dataset_name}_metrics_{timestamp}.csv")
    df_metricas = pd.DataFrame([metricas_con_info])
    df_metricas.to_csv(output_metrics_path, index=False, sep=',', encoding='utf-8-sig')
    logging.info(f"Métricas guardadas: {output_metrics_path}")

    # Guardar información de nodos con clusters (opcional)
    output_nodes_path = os.path.join(output_dir, f"{dataset_name}_nodes_clusters_{timestamp}.csv")
    df_nodes_final = df_nodes_results[['NODE_NAME', 'CLUSTER_ASSIGNMENT']].copy()
    df_nodes_final.rename(columns={'CLUSTER_ASSIGNMENT': 'CLUSTER'}, inplace=True)
    df_nodes_final.to_csv(output_nodes_path, index=False, sep=',', encoding='utf-8-sig')
    logging.info(f"Asignación de nodos guardada: {output_nodes_path}")

    # Resumen en consola
    print(f"\n{'='*60}")
    print(f"RESUMEN DEL ANÁLISIS: {dataset_name}")
    print(f"{'='*60}")
    print(f"Algoritmo utilizado: {config_clustering['name']}")
    print(f"Total de aristas: {len(df_resultado_final):,}")
    print(f"Aristas en Cluster 1: {metricas_con_info['Edges_Cluster_1']:,}")
    print(f"Aristas en Cluster -1: {metricas_con_info['Edges_Cluster_-1']:,}")
    print(f"Aristas neutrales (Cluster 0): {metricas_con_info['Edges_Cluster_0']:,}")
    print(f"Total de nodos únicos: {len(df_nodes_final):,}")
    print(f"Nodos en Cluster 1: {len(df_nodes_final[df_nodes_final['CLUSTER'] == 1]):,}")
    print(f"Nodos en Cluster -1: {len(df_nodes_final[df_nodes_final['CLUSTER'] == -1]):,}")
    print(f"Archivo principal: {os.path.basename(output_dataset_path)}")
    print(f"{'='*60}")

    return {
        'dataset_path': output_dataset_path,
        'metrics_path': output_metrics_path,
        'nodes_path': output_nodes_path,
        'metrics': metricas_con_info
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Ejecuta el clustering y genera datasets con columna CLUSTER.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="Ruta(s) a archivo(s) CSV a procesar. Acepta comodines entre comillas, p.ej. 'data/*.csv'."
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(PROJECT_ROOT, 'data', 'datasets_cluster'),
        help="Directorio donde guardar resultados (por defecto: data/datasets_cluster)."
    )
    parser.add_argument(
        "--name",
        help="Nombre base para el dataset cuando se procesa un único archivo (opcional)."
    )
    parser.add_argument(
        "--algo",
        default="local_search_paper_k2",
        choices=["local_search_paper_k2", "eigensign", "random_eigensign", "scg"],
        help="Algoritmo de clustering a utilizar."
    )
    parser.add_argument("--k", type=int, default=2, help="Número de clusters (cuando aplique).")
    parser.add_argument("--ls-beta", type=float, default=0.005, help="Parámetro beta para Local Search.")
    parser.add_argument("--ls-max-iter", type=int, default=2000, help="Iteraciones para Local Search.")
    return parser.parse_args()


def main():
    args = parse_args()

    # Configuración del algoritmo de clustering
    CLUSTERING_CONFIG = {
        "name": f"{args.algo} (beta={args.ls_beta})" if args.algo == "local_search_paper_k2" else args.algo,
        "algorithm_type": args.algo,
        "k": args.k,
        "ls_beta": args.ls_beta,
        "ls_max_iter": args.ls_max_iter
    }

    # Expandir comodines de entradas si se entregaron
    inputs = []
    if args.input:
        for pattern in args.input:
            expanded = glob.glob(pattern)
            if expanded:
                inputs.extend(expanded)
            else:
                logging.warning(f"No se encontraron archivos para el patrón: {pattern}")

    if not inputs:
        logging.error("No se proporcionaron entradas. Usa --input para especificar archivos CSV.")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    logging.info("=== INICIANDO ANÁLISIS INDIVIDUAL DE CLUSTERING ===")
    resultados = []

    for input_path in inputs:
        dataset_name = args.name if (args.name and len(inputs) == 1) else os.path.splitext(os.path.basename(input_path))[0]
        logging.info(f"Procesando: {dataset_name}")
        resultado = ejecutar_clustering_individual(
            dataset_path=input_path,
            dataset_name=dataset_name,
            config_clustering=CLUSTERING_CONFIG,
            output_dir=args.output_dir
        )

        if resultado:
            resultados.append(resultado)
            logging.info(f"OK. Completado: {dataset_name}")
        else:
            logging.error(f"Error procesando: {dataset_name}")

    # Resumen final
    if resultados:
        print(f"\n{'='*80}")
        print("ANÁLISIS COMPLETADO EXITOSAMENTE")
        print(f"{'='*80}")
        print(f"Datasets procesados: {len(resultados)}")
        print(f"Archivos generados en: {args.output_dir}")
        print(f"{'='*80}")
    else:
        print("\nNo se procesaron datasets. Verifica rutas y formato de columnas (FROM_NODE, TO_NODE, SIGN).")


if __name__ == '__main__':
    main()

