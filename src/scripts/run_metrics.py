"""
Métricas de frontera y visualizaciones para redes signadas ya clusterizadas.

Uso CLI:
  python src/scripts/run_metrics.py --input data/datasets_cluster/archivo_clustered_*.csv --results-dir results

También puede ejecutarse sin --input: buscará el CSV más reciente en data/datasets_cluster/.
"""

import os
import sys
import argparse
import glob
import warnings
warnings.filterwarnings('ignore')
import html as html_lib

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# Contexto de proyecto
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

# Variables globales que puede setear run_metrics_cli (compatibilidad)
INPUT_CSV_PATH = None
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')


def _find_latest_clustered_csv(base_dir: str) -> str | None:
    datasets_dir = os.path.join(base_dir, 'data', 'datasets_cluster')
    patterns = [os.path.join(datasets_dir, '*_clustered_*.csv'), os.path.join(datasets_dir, '*.csv')]
    candidates = []
    for pat in patterns:
        candidates.extend(glob.glob(pat))
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


# --- Carga y preparación ---
def load_and_prepare_data(filepath: str):
    filepath = os.path.normpath(filepath)
    print(f"Cargando datos desde: {filepath}")
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo en la ruta: {filepath}")
        return None, None

    # Validaciones mínimas
    for col in ['FROM_NODE', 'TO_NODE', 'SIGN']:
        if col not in df.columns:
            print(f"Error: Falta columna requerida: {col}")
            return None, None
    if 'CLUSTER' not in df.columns:
        print("Error: No se encontró columna CLUSTER. Ejecuta el clustering con run_pcd.py primero.")
        return None, None

    # Mapear SIGN si es texto
    sign_map = {'positive': 1, 'negative': -1, 'neutral': 0}
    if df['SIGN'].dtype == object:
        df['SIGN'] = df['SIGN'].str.lower().map(sign_map).fillna(0).astype(int)

    # Info de nodos (por CLUSTER más reciente disponible por arista)
    node_info_from = df[['FROM_NODE', 'CLUSTER']].rename(columns={'FROM_NODE': 'node'})
    node_info_to = df[['TO_NODE', 'CLUSTER']].rename(columns={'TO_NODE': 'node'})
    node_info = pd.concat([node_info_from, node_info_to]).drop_duplicates(subset='node').set_index('node')
    node_info['CLUSTER'] = node_info['CLUSTER'].astype(int)

    print("OK. Datos cargados y preparados.")
    return df, node_info


# --- Cálculo de métricas ---
def calculate_structural_positioning_metrics(G: nx.Graph, node_info: pd.DataFrame):
    print("Calculando métricas de posicionamiento estructural...")

    # Sesgo de Equilibrio S_eq
    s_eq = {}
    for node in G.nodes():
        d_c1 = sum(1 for neighbor in G.neighbors(node) if G.nodes[neighbor].get('cluster') == 1)
        d_c2 = sum(1 for neighbor in G.neighbors(node) if G.nodes[neighbor].get('cluster') == -1)
        total = d_c1 + d_c2
        s_eq[node] = (d_c1 - d_c2) / total if total > 0 else 0.0

    # Centralidad de autovector en el núcleo polarizado
    polarized_nodes = node_info[node_info['CLUSTER'].isin([-1, 1])].index.tolist()
    eigenvector_centrality = {}
    if polarized_nodes:
        G_nucleus = G.subgraph(polarized_nodes).copy()
        for u, v, data in G_nucleus.edges(data=True):
            G_nucleus[u][v]['weight'] = data.get('SIGN', 0)
        try:
            eigenvector_centrality = nx.eigenvector_centrality_numpy(G_nucleus, weight='weight')
        except Exception:
            print("Advertencia: Centralidad de autovector no convergió. Se usarán ceros.")
            eigenvector_centrality = {node: 0.0 for node in G_nucleus.nodes()}
    for node in G.nodes():
        eigenvector_centrality.setdefault(node, 0.0)

    return s_eq, eigenvector_centrality


def calculate_anomaly_metrics(G: nx.Graph, s_eq: dict):
    print("Calculando métricas de comportamiento anómalo...")
    conexiones_anomalas = {}
    indice_frontera = {}
    d_anom = {}
    p_anom = {}

    for node in G.nodes():
        node_cluster = G.nodes[node].get('cluster')
        total_connections = G.degree(node)

        # Conexiones anómalas
        if total_connections > 0:
            conexiones_anomalas[node] = total_connections * (1 - abs(s_eq[node])) / 2
        else:
            conexiones_anomalas[node] = 0.0

        # Índice de frontera
        if total_connections > 0:
            propor = conexiones_anomalas[node] / total_connections
            indice_frontera[node] = propor * np.log(1 + conexiones_anomalas[node])
        else:
            indice_frontera[node] = 0.0

        # Grado anómalo externo y proporción
        inter_pos = 0
        inter_tot = 0
        for nbr, edata in G[node].items():
            nbr_cluster = G.nodes[nbr].get('cluster')
            if node_cluster == 0 or nbr_cluster == 0:
                continue
            if node_cluster * nbr_cluster < 0:  # inter-cluster
                inter_tot += 1
                if edata.get('SIGN', 0) > 0:
                    inter_pos += 1
        d_anom[node] = inter_pos
        p_anom[node] = (inter_pos / inter_tot) if inter_tot > 0 else 0.0

    return conexiones_anomalas, indice_frontera, d_anom, p_anom


def calculate_all_metrics(G: nx.Graph, node_info: pd.DataFrame):
    s_eq, eigenvector_centrality = calculate_structural_positioning_metrics(G, node_info)
    conexiones_anomalas, indice_frontera, d_anom, p_anom = calculate_anomaly_metrics(G, s_eq)
    print("OK. Todas las métricas calculadas.")
    return {
        's_eq': s_eq,
        'eigenvector_centrality': eigenvector_centrality,
        'conexiones_anomalas': conexiones_anomalas,
        'indice_frontera': indice_frontera,
        'd_anom': d_anom,
        'p_anom': p_anom,
    }


# --- Visualizaciones ---
def generate_enhanced_visualizations(df_metrics: pd.DataFrame, G: nx.Graph, results_dir: str):
    print("\nGenerando visualizaciones...")

    os.makedirs(results_dir, exist_ok=True)
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_context("talk")
    palette = {-1: '#3498db', 1: '#e74c3c', 0: '#95a5a6'}
    df_polarized = df_metrics[df_metrics['CLUSTER'].isin([-1, 1])].copy()
    df_neutral = df_metrics[df_metrics['CLUSTER'] == 0].copy()

    if not df_polarized.empty:
        # 1) Mapa S_eq vs Índice de frontera
        fig, ax = plt.subplots(figsize=(16, 10))
        sns.scatterplot(data=df_polarized, x='s_eq', y='indice_frontera',
                        hue='CLUSTER', size='degree_total', sizes=(50, 500),
                        palette=palette, alpha=0.7, ax=ax)
        ax.axvline(0, color='black', linestyle='--', linewidth=2, alpha=0.7, label='Frontera Perfecta')
        ax.add_patch(Rectangle((-0.2, 0), 0.4, ax.get_ylim()[1], alpha=0.1, facecolor='yellow', label='Zona de Frontera'))
        ax.set_title('Mapa del Sesgo de Equilibrio vs Índice de Frontera', fontsize=20, weight='bold')
        ax.set_xlabel('Sesgo de Equilibrio (S_eq)')
        ax.set_ylabel('Índice de Frontera')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, '1_mapa_sesgo_equilibrio.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("OK. Gráfico 1 guardado")

        # 2) Distribuciones
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
        sns.histplot(data=df_polarized, x='s_eq', hue='CLUSTER', bins=30, kde=True, palette=palette, ax=ax1)
        ax1.axvline(0, color='black', linestyle='--', linewidth=2)
        ax1.set_title('Distribución de S_eq', fontsize=16, weight='bold')
        sns.histplot(data=df_polarized, x='indice_frontera', hue='CLUSTER', bins=30, kde=True, palette=palette, ax=ax2)
        ax2.set_title('Distribución del Índice de Frontera', fontsize=16, weight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, '2_espectro_frontera.png'), dpi=300)
        plt.close()
        print("OK. Gráfico 2 guardado")

        # 3) Matriz de correlación
        metrics_cols = ['s_eq', 'eigenvector_centrality', 'conexiones_anomalas', 'indice_frontera', 'd_anom', 'p_anom', 'degree_total']
        corr = df_polarized[metrics_cols].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='RdBu_r', center=0, square=True, fmt='.2f', cbar_kws={'shrink': 0.8}, ax=ax)
        ax.set_title('Matriz de Correlación entre Métricas', fontsize=16, weight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, '3_matriz_correlacion.png'), dpi=300)
        plt.close()
        print("OK. Gráfico 3 guardado")

        # 4) PCA
        features = ['s_eq', 'eigenvector_centrality', 'conexiones_anomalas', 'indice_frontera', 'd_anom', 'p_anom', 'degree_total']
        df_pca_input = df_metrics[features].replace([np.inf, -np.inf], 0).fillna(0)
        scaler = StandardScaler()
        scaled = scaler.fit_transform(df_pca_input)
        pca = PCA(n_components=2)
        comps = pca.fit_transform(scaled)
        df_metrics['pca1'] = comps[:, 0]
        df_metrics['pca2'] = comps[:, 1]
        fig, ax = plt.subplots(figsize=(16, 12))
        sns.scatterplot(data=df_metrics, x='pca1', y='pca2', hue='CLUSTER', size='degree_total', sizes=(50, 1200), palette=palette, alpha=0.8, ax=ax)
        ax.set_title('Proyección PCA del Ecosistema', fontsize=20, weight='bold')
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, '4_pca_ecosistema.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("OK. Gráfico 4 guardado")
    else:
        print("No hay nodos polarizados para visualizar.")

    # Visualización adicional: Nodos neutrales S_eq vs Grado Total
    if not df_neutral.empty:
        fig, ax = plt.subplots(figsize=(16, 10))
        sns.scatterplot(
            data=df_neutral,
            x='s_eq',
            y='degree_total',
            size='degree_total',
            sizes=(50, 1200),
            color=palette[0],
            alpha=0.7,
            ax=ax
        )
        ax.axvline(0, color='black', linestyle='--', linewidth=2, alpha=0.7, label='Balance Perfecto')
        ax.set_title('Rol de Nodos Neutrales: Inclinación vs. Actividad General', fontsize=20, weight='bold')
        ax.set_xlabel('Sesgo de Equilibrio ($S_{eq}$ - Inclinación estructural)', fontsize=14)
        ax.set_ylabel('Grado Total (Actividad general en la red)', fontsize=14)
        top_neutral_active = df_neutral.nlargest(10, 'degree_total')
        for node_name, row in top_neutral_active.iterrows():
            ax.annotate(node_name, (row['s_eq'], row['degree_total']), xytext=(5, 5), textcoords='offset points')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'neutrales_inclinacion_vs_actividad.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("OK. Gráfico neutrales_inclinacion_vs_actividad guardado")


def generate_html_report(df_metrics: pd.DataFrame, results_dir: str, title: str = "Reporte de Métricas de Frontera") -> str:
    """
    Genera un reporte HTML sencillo con KPIs, tablas y enlaces a figuras.
    Devuelve la ruta del archivo HTML generado.
    """
    os.makedirs(results_dir, exist_ok=True)

    # KPIs básicos
    counts = df_metrics['CLUSTER'].value_counts().to_dict()
    total_nodos = int(df_metrics.shape[0])

    # Tops
    top_frontier = df_metrics.sort_values('indice_frontera', ascending=False).head(15)
    top_anom = df_metrics.sort_values('d_anom', ascending=False).head(15)
    top_degree_c1 = (
        df_metrics[df_metrics['CLUSTER'] == 1]
        .sort_values('degree_total', ascending=False)
        .head(10)
    )
    top_degree_c2 = (
        df_metrics[df_metrics['CLUSTER'] == -1]
        .sort_values('degree_total', ascending=False)
        .head(10)
    )

    def df_to_html_table(df: pd.DataFrame, cols: list[str]) -> str:
        safe_cols = [c for c in cols if c in df.columns]
        return df[safe_cols].round(3).to_html(index=True, classes='table', border=0)

    imgs = [
        '1_mapa_sesgo_equilibrio.png',
        '2_espectro_frontera.png',
        '3_matriz_correlacion.png',
        '4_pca_ecosistema.png',
        'neutrales_inclinacion_vs_actividad.png',
    ]

    # Renderizado simple de Markdown a HTML para el anexo
    def md_to_html(md_text: str) -> str:
        lines = md_text.splitlines()
        html_lines = []
        in_ul = False
        for raw in lines:
            line = raw.rstrip('\n')
            if line.strip() == '':
                if in_ul:
                    html_lines.append('</ul>')
                    in_ul = False
                html_lines.append('<br>')
                continue
            if line.startswith('### '):
                if in_ul:
                    html_lines.append('</ul>')
                    in_ul = False
                html_lines.append(f"<h3>{html_lib.escape(line[4:])}</h3>")
            elif line.startswith('## '):
                if in_ul:
                    html_lines.append('</ul>')
                    in_ul = False
                html_lines.append(f"<h2>{html_lib.escape(line[3:])}</h2>")
            elif line.startswith('# '):
                if in_ul:
                    html_lines.append('</ul>')
                    in_ul = False
                html_lines.append(f"<h1>{html_lib.escape(line[2:])}</h1>")
            elif line.lstrip().startswith('- '):
                if not in_ul:
                    html_lines.append('<ul>')
                    in_ul = True
                content = line.lstrip()[2:]
                html_lines.append(f"<li>{html_lib.escape(content)}</li>")
            else:
                if in_ul:
                    html_lines.append('</ul>')
                    in_ul = False
                html_lines.append(f"<p>{html_lib.escape(line)}</p>")
        if in_ul:
            html_lines.append('</ul>')
        return '\n'.join(html_lines)

    # Cargar docs/metricas_frontera.md si existe
    annex_html = ""
    md_path = os.path.join(PROJECT_ROOT, 'docs', 'metricas_frontera.md')
    if os.path.exists(md_path):
        try:
            with open(md_path, 'r', encoding='utf-8') as f:
                md_text = f.read()
            annex_html = md_to_html(md_text)
        except Exception as e:
            annex_html = f"<p class=\"muted\">No se pudo cargar metricas_frontera.md: {html_lib.escape(str(e))}</p>"
    else:
        annex_html = "<p class=\"muted\">No se encontró docs/metricas_frontera.md.</p>"

    html = f"""
<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; }}
    h1, h2 {{ margin: 0.6em 0; }}
    .kpis {{ display: flex; gap: 16px; flex-wrap: wrap; margin: 12px 0 24px; }}
    .kpi {{ border: 1px solid #ddd; padding: 12px 16px; border-radius: 8px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr)); gap: 16px; }}
    .card {{ border: 1px solid #eee; border-radius: 8px; padding: 12px; }}
    img {{ max-width: 100%; height: auto; border: 1px solid #eee; border-radius: 6px; }}
    .table {{ border-collapse: collapse; width: 100%; }}
    .table th, .table td {{ border: 1px solid #eee; padding: 6px 8px; text-align: center; }}
    .muted {{ color: #666; font-size: 0.95em; }}
  </style>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap" rel="stylesheet">
  <style> body {{ font-family: 'Inter', Arial, sans-serif; }} </style>
  </head>
<body>
  <h1>{title}</h1>
  <div class="kpis">
    <div class="kpi"><b>Total de nodos</b><br>{total_nodos}</div>
    <div class="kpi"><b>Cluster 1</b><br>{counts.get(1, 0)}</div>
    <div class="kpi"><b>Cluster -1</b><br>{counts.get(-1, 0)}</div>
    <div class="kpi"><b>Neutrales (0)</b><br>{counts.get(0, 0)}</div>
  </div>

  <h2>Tablas destacadas</h2>
  <div class="card" style="overflow-x:auto;">
    <h3>Top 15 Índice de Frontera</h3>
    {df_to_html_table(top_frontier, ['s_eq','indice_frontera','d_anom','p_anom','degree_total','CLUSTER'])}
  </div>
  <div class="card" style="overflow-x:auto; margin-top:16px;">
    <h3>Top 15 Grado Anómalo Externo</h3>
    {df_to_html_table(top_anom, ['d_anom','p_anom','s_eq','indice_frontera','degree_total','CLUSTER'])}
  </div>
  <div class="card" style="overflow-x:auto; margin-top:16px;">
    <h3>Top 10 Grado por Cluster 1</h3>
    {df_to_html_table(top_degree_c1, ['degree_total','s_eq','indice_frontera','d_anom','p_anom','CLUSTER'])}
  </div>
  <div class="card" style="overflow-x:auto; margin-top:16px;">
    <h3>Top 10 Grado por Cluster -1</h3>
    {df_to_html_table(top_degree_c2, ['degree_total','s_eq','indice_frontera','d_anom','p_anom','CLUSTER'])}
  </div>

  <h2>Figuras</h2>
  <div class="grid">
    {''.join([f'<div class="card"><img src="{img}" alt="{img}"><div class="muted">{img}</div></div>' for img in imgs])}
  </div>

  <p class="muted">Generado automáticamente por run_metrics.py</p>
</body>
</html>
"""

    # Inyectar MathJax para que renderice fórmulas LaTeX del anexo
    mathjax_head = (
        "  <!-- MathJax for LaTeX rendering -->\n"
        "  <script>\n"
        "    window.MathJax = {\n"
        "      tex: { inlineMath: [['$', '$'], ['\\\\(', '\\\\)']], displayMath: [['$$','$$'], ['\\\\[','\\\\]']] },\n"
        "      svg: { fontCache: 'global' }\n"
        "    };\n"
        "  </script>\n"
        "  <script id=\"MathJax-script\" defer src=\"https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js\"></script>\n"
    )
    html = html.replace("</head>", mathjax_head + "</head>")

    # Insertar anexo antes del cierre del body
    html = html.replace(
        "</body>",
        f"""
  <h2 style=\"margin-top:28px;\">Anexo: Métricas y Fórmulas</h2>
  <div class=\"card\"> 
    <div class=\"md-content\">
      {annex_html}
    </div>
  </div>

</body>"""
    )

    out_path = os.path.join(results_dir, 'reporte_frontera.html')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"Reporte HTML generado: {out_path}")
    return out_path


# --- Orquestación ---
def run_frontier_metrics(input_csv_path: str, results_dir: str):
    os.makedirs(results_dir, exist_ok=True)

    df_edges, node_info = load_and_prepare_data(input_csv_path)
    if df_edges is None:
        return None

    # Construir grafo
    G = nx.from_pandas_edgelist(df_edges, 'FROM_NODE', 'TO_NODE', edge_attr=['SIGN'])
    nx.set_node_attributes(G, node_info['CLUSTER'].to_dict(), 'cluster')

    # Calcular métricas
    metrics = calculate_all_metrics(G, node_info)

    # DataFrame final
    metrics_df = node_info.copy().reset_index()
    for metric_name, metric_dict in metrics.items():
        metrics_df[metric_name] = metrics_df['node'].map(metric_dict).fillna(0)
    degree = dict(G.degree())
    metrics_df['degree_total'] = metrics_df['node'].map(degree).fillna(0).astype(int)
    metrics_df = metrics_df.set_index('node')

    # Guardar métricas
    output_csv_path = os.path.join(results_dir, 'metricas_frontera_completas.csv')
    metrics_df.reset_index().to_csv(output_csv_path, index=False)
    print(f"Métricas guardadas en: {output_csv_path}")

    # Vista previa
    print("\n--- Vista previa de métricas ---")
    preview_cols = ['CLUSTER', 's_eq', 'eigenvector_centrality', 'indice_frontera', 'd_anom', 'p_anom']
    print(metrics_df[preview_cols].head(10).round(3).to_string())

    # Visualizaciones
    generate_enhanced_visualizations(metrics_df.copy(), G.copy(), results_dir)
    # Reporte HTML
    generate_html_report(metrics_df.copy(), results_dir, title="Reporte de Métricas de Frontera")

    print(f"\nAnálisis completo. Revisa resultados en: {results_dir}")
    return output_csv_path


def parse_args():
    parser = argparse.ArgumentParser(description="Métricas de frontera para CSV clusterizado")
    parser.add_argument("--input", help="Ruta al CSV clusterizado (FROM_NODE, TO_NODE, SIGN, CLUSTER)")
    parser.add_argument("--results-dir", default=RESULTS_DIR, help="Directorio para guardar resultados")
    return parser.parse_args()


def main():
    # Permitir que run_metrics_cli setee variables globales; si no, usar argparse
    args = parse_args()
    input_path = args.input or INPUT_CSV_PATH
    results_dir = args.results_dir or RESULTS_DIR

    if not input_path:
        # Buscar el más reciente en data/datasets_cluster
        input_path = _find_latest_clustered_csv(PROJECT_ROOT)
        if not input_path:
            print("No se encontró archivo clusterizado en data/datasets_cluster. Usa --input.")
            sys.exit(1)

    run_frontier_metrics(input_path, results_dir)


if __name__ == "__main__":
    main()
