# Algoritmo de Búsqueda Local (k = 2) para Redes Signadas

Este documento describe el algoritmo de búsqueda local implementado en `src/pcd/algorithms.py` (`run_local_search_paper_k2`) para particionar redes signadas en dos comunidades polarizadas, permitiendo además un conjunto de nodos neutrales.

## Objetivo del Modelo

Dado un grafo signado no dirigido con matriz de adyacencia `A_s ∈ R^{n×n}` (pesos en {−1, 0, 1}), buscamos una asignación discreta por nodo `x ∈ {−1, 0, 1}^n` donde:

- `x_i = 1` pertenece al Clúster 1 (polo 1)
- `x_i = −1` pertenece al Clúster 2 (polo 2)
- `x_i = 0` es neutral (no participa del núcleo polarizado)

La función objetivo balancea dos criterios: coherencia con los signos de las aristas (polarización) y una regularización que penaliza soluciones triviales o degeneradas.

## Función Objetivo (Paper k=2)

Definimos los conteos sobre el subgrafo inducido por los nodos no neutrales (`x_i ≠ 0`):

- `N_intra_pos`: número (o peso) de aristas positivas dentro de cada clúster
- `N_intra_neg`: número de aristas negativas dentro de clúster
- `N_inter_pos`: número de aristas positivas entre clústeres opuestos
- `N_inter_neg`: número de aristas negativas entre clústeres opuestos

La parte de “polaridad” (coherencia estructural) del objetivo es:

$$ \mathrm{POL} = (N_{intra}^{+} - N_{intra}^{-}) + \alpha\, (N_{inter}^{-} - N_{inter}^{+}) $$

Esta premiación favorece “amigos dentro” y “enemigos fuera” (estructura bipolar). Para evitar que el algoritmo fuerce tamaños extremos (por ejemplo, meter casi todos a un clúster), se añade una regularización sobre los tamaños de los clústeres no neutrales:

$$ \mathrm{REG} = \beta\, (|S_1|^2 + |S_2|^2) $$

La función objetivo total a maximizar es:

$$ \max_{x \in \{-1,0,1\}^n} \; \mathrm{POL}(x; A_s, \alpha) - \mathrm{REG}(x; \beta) $$

donde `|S_1|` y `|S_2|` son los tamaños de los clústeres 1 y 2.

- `α` (alpha): pondera la coherencia entre polos (recompensa aristas negativas inter-clúster y penaliza positivas entre clústeres).
- `β` (beta): controla la penalización por tamaños (ayuda a evitar soluciones sesgadas por tamaño).

En el repositorio, el cálculo de esta métrica está implementado en `src/pcd/analysis.py::calculate_objective_paper_k2` para fines de evaluación.

## Esquema del Algoritmo (k = 2)

El algoritmo es una búsqueda local eficiente basada en actualizaciones coordinadas por nodo con pre-cómputo incremental.

1) Inicialización aleatoria `x_internal ∈ {0,1,2}^n` (0=neutral, 1=C1, 2=C2).

2) Construcción de matriz indicador `X ∈ R^{n×2}` y pre-cómputo:

- `X[i, c-1] = 1` si `x_internal[i] = c` (c ∈ {1,2})
- `M = 2 A_s X` (dos columnas: acumulados hacia C1 y hacia C2). Permite evaluar movimientos en O(grado(i)).

3) Bucle de mejora (hasta `ls_max_iter`):

- Se permuta aleatoriamente el orden de los nodos.
- Para cada nodo `i` con asignación original `orig ∈ {0,1,2}` se computa un “gradiente” aproximado `G[c]` para `c ∈ {0,1,2}`:

  - Sea `η_i = sum_j M[i, j]`, `s_m_size` el tamaño del clúster m (m ∈ {1,2}), `is_in_cluster` indicador si `i ∈ S_m`.
  - Para m ∈ {1,2}:  
    `G[m] = −β + (1 + α)·M[i, m-1] − (2β)·(s_m_size − is_in_cluster) − α·η_i`
  - Se fija `G[0] = 0` (neutral), actuando como opción de “no polarizarse”.

- Se elige `best = argmax_c G[c]` y si `best ≠ orig` se actualiza `x_internal[i] = best` y se corrige `M` en O(grado(i)):

  - `M[:, orig-1] -= 2·A_s[:, i]` si `orig > 0`  
  - `M[:, best-1] += 2·A_s[:, i]` si `best > 0`

- Si en una pasada completa no hay cambios, se declara convergencia.

4) Mapeo final a `x ∈ {−1,0,1}^n`: 1→`+1`, 2→`−1`, 0→`0`.

Este esquema realiza pasos de mejora “codiciosos” basados en un gradiente local consistente con la función objetivo y aprovecha un pre-cómputo `M` para actualizaciones rápidas.

## Complejidad y Escalabilidad

- Inicialización: `O(n + nnz(A_s))`
- Por iteración: `O(∑_i deg(i)) = O(nnz(A_s))` para evaluar y actualizar, más términos `O(n)` por contabilidad de tamaños.  
- Total: `O(max_iter · nnz(A_s))` aproximadamente (para k=2).  
- Memoria: `A_s` dispersa + `M ∈ R^{n×2}` y estructuras ligeras.

Funciona bien en grafos medianos a grandes siempre que `A_s` sea dispersa. El parámetro `ls_max_iter` limita el tiempo.

## Hiperparámetros y Sugerencias

- `k = 2`: dos polos (modelo de clivaje binario). Para `k > 2` se recomienda el algoritmo SCG incluido.
- `α` (alpha): por defecto `1.0` para k=2 (equivalente a 1/(k−1)). Aumentarlo hace más estricta la penalización de “amistades cruzadas”.
- `β` (beta): regula el tamaño de los clústeres. Valores pequeños (p.ej. 0.001–0.02) suelen funcionar bien; en los quicktests se usa `0.005`.
- `ls_max_iter`: límite superior de iteraciones; típicamente converge antes.

Consejo práctico: si observas clústeres muy desbalanceados, prueba aumentar ligeramente `β`. Si ves demasiados neutrales, puede ayudar reducir `β` o aumentar `α`.

## Integración en el Repo

- Implementación: `src/pcd/algorithms.py::run_local_search_paper_k2`
- Cálculo de métricas de evaluación: `src/pcd/analysis.py` (incluye `calculate_objective_paper_k2`)
- CLI de clustering: `src/scripts/run_pcd.py`

Ejemplo de uso (CLI):

```
python src/scripts/run_pcd.py \
  --input "data/raw/primarias2025.csv" \
  --output-dir data/datasets_cluster \
  --algo local_search_paper_k2 --k 2 --ls-beta 0.005 --ls-max-iter 2000
```

El script genera:

- `*_clustered_YYYYmmdd_HHMMSS.csv`: aristas con columna `CLUSTER`
- `*_metrics_YYYYmmdd_HHMMSS.csv`: métricas del experimento
- `*_nodes_clusters_YYYYmmdd_HHMMSS.csv`: asignación final de nodos

## Diferencias con Métodos Espectrales (Eigensign) y SCG

- Eigensign: usa el autovector principal para inducir una asignación por umbral. Es rápido y estable, pero no incorpora una regularización explícita por tamaños ni neutrales durante la optimización.
- SCG (k ≥ 2): diseñado para múltiples comunidades y usa redondeos espectrales especializados. Para k=2, la búsqueda local ofrece una alternativa con control explícito de neutrales y del término de regularización.

## Pseudocódigo Resumido

```
Input: A_s (csr matriz dispersa), n, α, β, max_iter
x ← Random in {0,1,2}^n; construir X y M = 2 A_s X
for t in 1..max_iter:
  changed ← false
  for i in RandomPermutation(1..n):
    orig ← x[i]
    η_i ← sum(M[i, :])
    para m=1..2:
      s_m_size ← |{j : x[j] = m}|
      is_in ← 1 si orig = m sino 0
      G[m] ← −β + (1+α)·M[i,m-1] − 2β·(s_m_size − is_in) − α·η_i
    G[0] ← 0
    best ← argmax_c G[c]
    si best ≠ orig:
      si orig>0: M[:,orig-1] ← M[:,orig-1] − 2·A_s[:,i]
      si best>0: M[:,best-1] ← M[:,best-1] + 2·A_s[:,i]
      x[i] ← best; changed ← true
  si no changed: break
return mapear {0,1,2}→{0, +1, −1}
```

## Consideraciones Prácticas

- Normalización y pesos: por defecto `A_s` usa “binary_sum_signs_actual” (signo mayoritario por par de nodos), robusto a ruido.
- Neutrales: pueden ser numerosos si la señal es débil; ajustar `α`/`β` para controlar la fracción neutral.
- Grafos desconectados: el algoritmo funciona por componente; la convergencia puede ser más rápida.
- Reproducibilidad: la inicialización es aleatoria; fija la semilla global si necesitas resultados repetibles.

## Referencias

- (Aronsson et al., 2025) – Formulación de la función objetivo y esquema de búsqueda local para k=2 (referencia interna del proyecto).

